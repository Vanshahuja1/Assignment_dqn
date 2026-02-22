# modules/environment.py
import numpy as np
import random
from params import *
from initializer import initialize_topology
from association import associate_users_to_bs

class Session:
    def __init__(self, id, operator_id, service_type, arrival_slot, source_edge_id):
        self.id = id
        self.operator_id = operator_id
        self.service_type = service_type
        self.arrival_slot = arrival_slot
        self.source_edge_id = source_edge_id
        
        self.data_rate = SERVICE_RATES_KBPS[service_type]
        self.latency_req = SERVICE_LATENCY_REQ_MS[service_type]
        self.cpu_cost = SERVICE_CPU_COST_MC[service_type]
        
        self.served_slot = None
        self.forwarded = False
        self.forward_count = 0
        self.total_propagation_delay = 0

class EdgeEnvironment:
    def __init__(self, num_users=500):
        # Initialize Topology
        self.bs_list, self.edge_list, self.bs_to_edge, self.users = initialize_topology(num_users)
        
        # Edge internal state: Queues
        # 6 queues per edge: [L_U, L_E, L_M, R_U, R_E, R_M]
        self.local_queues = {e.id: {st: [] for st in SERVICE_TYPES} for e in self.edge_list}
        self.roaming_queues = {e.id: {st: [] for st in SERVICE_TYPES} for e in self.edge_list}
        
        # Performance tracking
        self.prev_cpu_utilization = {e.id: 0.0 for e in self.edge_list}
        self.slot_count = 0
        self.session_id_counter = 0

    def reset_episode(self):
        self.slot_count = 0
        self.session_id_counter = 0
        self.prev_cpu_utilization = {e.id: 0.0 for e in self.edge_list}
        self.local_queues = {e.id: {st: [] for st in SERVICE_TYPES} for e in self.edge_list}
        self.roaming_queues = {e.id: {st: [] for st in SERVICE_TYPES} for e in self.edge_list}
        
        # Re-initialize users (random positions every episode as per PDF)
        # initializer.initialize_topology already does this if we call it again
        # but for efficiency, let's just re-randomize positions
        for user in self.users:
            user.location = (random.uniform(0, AREA_WIDTH), random.uniform(0, AREA_HEIGHT))
            user.associated_bs = None
            user.sinr_table = {}
            
        return self._get_all_states()

    def _get_state(self, edge_id):
        """Eq 60-62: State vector normalization."""
        l_queues = [len(self.local_queues[edge_id][st]) for st in SERVICE_TYPES]
        r_queues = [len(self.roaming_queues[edge_id][st]) for st in SERVICE_TYPES]
        prev_util = self.prev_cpu_utilization[edge_id]
        
        state = np.array(l_queues + r_queues + [prev_util], dtype=np.float32)
        # Normalize by Q_MAX (except the last utilization term)
        state[:6] = state[:6] / Q_MAX
        return state

    def _get_all_states(self):
        return {e.id: self._get_state(e.id) for e in self.edge_list}

    def _dist(self, p, q):
        return np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

    def step(self, actions):
        """
        actions: dict {edge_id: alpha_vector (len 6)}
        """
        self.slot_count += 1
        
        # Enhanced Metrics Structure
        slot_metrics = {
            e.id: {
                "tp_URLLC": 0, "tp_eMBB": 0, "tp_mMTC": 0,
                "lat_URLLC": [], "lat_eMBB": [], "lat_mMTC": [],
                "vio_URLLC": 0, "vio_eMBB": 0, "vio_mMTC": 0,
                "served_count": 0, "cpu_used": 0, "mem_vio": 0, "cpu_waste": 0
            } for e in self.edge_list
        }
        
        # 1. PSA Association (Algorithm 2)
        associate_users_to_bs(self.users, self.bs_list)
        
        # 2. Build per-edge arrivals
        for e_id in self.roaming_queues:
            for st in SERVICE_TYPES:
                self.roaming_queues[e_id][st] = []

        for user in self.users:
            if random.random() < P_SESSION_REQUEST and user.associated_bs is not None:
                service_type = random.choices(SERVICE_TYPES, weights=SERVICE_PI, k=1)[0]
                edge_id = self.bs_to_edge[user.associated_bs]
                target_edge = next(e for e in self.edge_list if e.id == edge_id)
                session = Session(self.session_id_counter, user.operator_id, service_type, self.slot_count, edge_id)
                self.session_id_counter += 1
                
                assoc_bs_obj = next(b for b in self.bs_list if b.id == user.associated_bs)
                prop_delay = 1000 * self._dist(assoc_bs_obj.location, target_edge.location) / V_F
                session.total_propagation_delay += prop_delay
                
                if (user.operator_id == target_edge.operator_id):
                    self.local_queues[edge_id][service_type].append(session)
                else:
                    self.roaming_queues[edge_id][service_type].append(session)

        # 3. Serve-or-Forward
        forwarded_sessions_this_slot = []
        for edge in self.edge_list:
            alpha = actions[edge.id]
            cpu_budget = edge.cpu_budget_mc
            q_idx = [('local',st) for st in SERVICE_TYPES] + [('roaming',st) for st in SERVICE_TYPES]
            
            for i, (q_type, st) in enumerate(q_idx):
                q_cpu = alpha[i] * cpu_budget
                q = self.local_queues[edge.id][st] if q_type == 'local' else self.roaming_queues[edge.id][st]
                capacity = int(np.floor(q_cpu / SERVICE_CPU_COST_MC[st]))
                num_served = min(len(q), capacity)
                
                for _ in range(num_served):
                    session = q.pop(0)
                    session.served_slot = self.slot_count
                    q_delay = (session.served_slot - session.arrival_slot) * DT * 1000
                    p_delay = 1000 * DT * (session.cpu_cost) / (q_cpu) if q_cpu > 0 else 0
                    total_lat = session.total_propagation_delay + q_delay + p_delay
                    
                    slot_metrics[edge.id][f"tp_{st}"] += session.data_rate
                    slot_metrics[edge.id][f"lat_{st}"].append(total_lat)
                    if total_lat > session.latency_req:
                        slot_metrics[edge.id][f"vio_{st}"] += 1
                    
                    slot_metrics[edge.id]["served_count"] += 1
                    slot_metrics[edge.id]["cpu_used"] += session.cpu_cost
                
                if q_type == 'roaming':
                    while q: forwarded_sessions_this_slot.append(q.pop(0))

            self.prev_cpu_utilization[edge.id] = slot_metrics[edge.id]["cpu_used"] / cpu_budget
                
        # 4. System Coupling (Forwarded Roaming)
        for sess in forwarded_sessions_this_slot:
            own_edges = [e for e in self.edge_list if e.operator_id == sess.operator_id]
            target_edge = own_edges[0]
            current_edge = next(e for e in self.edge_list if e.id == sess.source_edge_id)
            e2e_dist = self._dist(current_edge.location, target_edge.location)
            sess.total_propagation_delay += (1000 * e2e_dist / V_F)
            sess.forward_count += 1
            self.local_queues[target_edge.id][sess.service_type].append(sess)

        # 5. Reward Calculation (Eq. 56)
        rewards = {}
        for edge in self.edge_list:
            m = slot_metrics[edge.id]
            
            # Simple normalization for reward calculation
            current_tp = sum(m[f"tp_{st}"] for st in SERVICE_TYPES)
            norm_tp = current_tp / (250 * 500) 
            
            total_vios = sum(m[f"vio_{st}"] for st in SERVICE_TYPES)
            avg_vio = total_vios / m["served_count"] if m["served_count"] > 0 else 0
            
            # CPU Waste (Eq. 54)
            total_work = sum(len(self.local_queues[edge.id][st]) for st in SERVICE_TYPES) + \
                         sum(len(self.roaming_queues[edge.id][st]) for st in SERVICE_TYPES)
            if total_work > 0:
                m["cpu_waste"] = 1.0 - self.prev_cpu_utilization[edge.id]
            
            # Memory accounting (Eq. 25)
            curr_mem = (sum(len(self.local_queues[edge.id][st]) for st in SERVICE_TYPES) + 
                        sum(len(self.roaming_queues[edge.id][st]) for st in SERVICE_TYPES)) * M0_MB
            m["mem_vio"] = max(0, curr_mem / edge.mem_budget_mb - 1)

            reward = W1_THROUGHPUT * norm_tp - \
                     W2_LATENCY * avg_vio - \
                     W3_CPU_WASTE * m["cpu_waste"] - \
                     W4_MEM_VIO * m["mem_vio"]
            rewards[edge.id] = reward

        return self._get_all_states(), rewards, slot_metrics
