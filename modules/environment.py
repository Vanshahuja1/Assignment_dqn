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
        slot_metrics = {e.id: {"throughput": 0, "latency_vio": 0, "served_count": 0, "cpu_waste": 0, "mem_vio": 0} for e in self.edge_list}
        
        # 1. PSA Association (Step 1 of Algorithm 2)
        # Re-associate if users moved? (Assumed fixed in slot but changes per episode)
        # For simplicity, we re-associate if it's a new slot or episode.
        associate_users_to_bs(self.users, self.bs_list)
        
        # 2. Build per-edge arrivals (Step 2 of Algorithm 2)
        # Clear roaming queues (they represent only current-slot arrivals)
        for e_id in self.roaming_queues:
            for st in SERVICE_TYPES:
                self.roaming_queues[e_id][st] = []

        new_sessions = []
        for user in self.users:
            if random.random() < P_SESSION_REQUEST and user.associated_bs is not None:
                service_type = random.choices(SERVICE_TYPES, weights=SERVICE_PI, k=1)[0]
                edge_id = self.bs_to_edge[user.associated_bs]
                
                # Check if session is local or roaming
                target_edge = next(e for e in self.edge_list if e.id == edge_id)
                is_local = (user.operator_id == target_edge.operator_id)
                
                session = Session(self.session_id_counter, user.operator_id, service_type, self.slot_count, edge_id)
                self.session_id_counter += 1
                
                # BS->Edge propagation delay (Eq 36)
                assoc_bs_obj = next(b for b in self.bs_list if b.id == user.associated_bs)
                prop_delay = 1000 * self._dist(assoc_bs_obj.location, target_edge.location) / V_F
                session.total_propagation_delay += prop_delay
                
                if is_local:
                    self.local_queues[edge_id][service_type].append(session)
                else:
                    self.roaming_queues[edge_id][service_type].append(session)
                new_sessions.append(session)

        # 3. Serve-or-Forward Rule (Step 4 of Algorithm 2)
        served_sessions_this_slot = []
        forwarded_sessions_this_slot = []

        for edge in self.edge_list:
            alpha = actions[edge.id]
            cpu_budget = edge.cpu_budget_mc
            
            # Record initial memory usage (Eq. 25)
            # Me,used(t) depends on (i) local backlog, (ii) new local arrivals, (iii) current-slot roaming
            # This is basically len(local_queues) + len(roaming_queues)
            current_mem_sessions = sum(len(self.local_queues[edge.id][st]) for st in SERVICE_TYPES) + \
                                   sum(len(self.roaming_queues[edge.id][st]) for st in SERVICE_TYPES)
            mem_used = current_mem_sessions * M0_MB
            slot_metrics[edge.id]["mem_vio"] = max(0, mem_used / edge.mem_budget_mb - 1)

            cpu_used_edge = 0
            
            # Map Alpha to queues: [L_U, L_E, L_M, R_U, R_E, R_M]
            queue_indices = [
                ('local', 'URLLC'), ('local', 'eMBB'), ('local', 'mMTC'),
                ('roaming', 'URLLC'), ('roaming', 'eMBB'), ('roaming', 'mMTC')
            ]
            
            for i, (q_type, st) in enumerate(queue_indices):
                q_cpu_budget = alpha[i] * cpu_budget
                q_to_serve = self.local_queues[edge.id][st] if q_type == 'local' else self.roaming_queues[edge.id][st]
                
                # Capacities (Eq. 14)
                capacity = int(np.floor(q_cpu_budget / SERVICE_CPU_COST_MC[st]))
                num_served = min(len(q_to_serve), capacity)
                
                # Actual serving
                for _ in range(num_served):
                    session = q_to_serve.pop(0)
                    session.served_slot = self.slot_count
                    
                    # Latency Logic (Eq. 38, 40, 42)
                    queueing_delay = (session.served_slot - session.arrival_slot) * DT * 1000
                    # Processing delay (Eq. 40) - simplified to exact cost if served
                    proc_delay = 1000 * DT * (session.cpu_cost * 1) / (alpha[i] * cpu_budget) if alpha[i] > 0 else 0
                    
                    total_latency = session.total_propagation_delay + queueing_delay + proc_delay
                    
                    # Metrics
                    slot_metrics[edge.id]["throughput"] += session.data_rate
                    if total_latency > session.latency_req:
                        # Normalized violation (Eq. 51)
                        slot_metrics[edge.id]["latency_vio"] += (total_latency - session.latency_req) / session.latency_req
                    
                    slot_metrics[edge.id]["served_count"] += 1
                    cpu_used_edge += session.cpu_cost
                
                # If roaming and not served -> forward (Eq. 16)
                if q_type == 'roaming':
                    # Remaining in roaming queue must be forwarded
                    while q_to_serve:
                        sess = q_to_serve.pop(0)
                        forwarded_sessions_this_slot.append(sess)

            # Record CPU utilization for next state
            self.prev_cpu_utilization[edge.id] = cpu_used_edge / cpu_budget
            
            # CPU Waste Penalty (Eq. 54)
            # Activates only when work is present
            total_work = sum(len(self.local_queues[edge.id][st]) for st in SERVICE_TYPES) + \
                         sum(len(self.roaming_queues[edge.id][st]) for st in SERVICE_TYPES)
            if total_work > 0:
                slot_metrics[edge.id]["cpu_waste"] = 1.0 - self.prev_cpu_utilization[edge.id]

        # 4. System Coupling (Step 5 of Algorithm 2)
        # Forwarded roaming sessions arrive in the destination edge's local queues in NEXT slot
        for sess in forwarded_sessions_this_slot:
            # Deterministic mapping f(e, o) -> eo
            # In our setup, bs_to_edge maps BS to edge of SAME operator.
            # Roaming sessions belong to operator X but arrived at edge of operator Y.
            # They should be forwarded to an edge of operator X.
            own_edges = [e for e in self.edge_list if e.operator_id == sess.operator_id]
            # Simple policy: forward to the FIRST edge of their own operator
            target_edge = own_edges[0]
            
            # Add edge-to-edge propagation delay (Eq 37)
            # (Assuming distance between current edge and target edge)
            current_edge = next(e for e in self.edge_list if e.id == sess.source_edge_id)
            e2e_dist = self._dist(current_edge.location, target_edge.location)
            sess.total_propagation_delay += (1000 * e2e_dist / V_F)
            sess.forward_count += 1
            
            # Add to target local queue (Algorithm 2, line 17)
            self.local_queues[target_edge.id][sess.service_type].append(sess)

        # 5. Reward Calculation (Eq. 56)
        rewards = {}
        for edge in self.edge_list:
            m = slot_metrics[edge.id]
            # Normalized throughput (Eq. 50) - simplified to avg service rate scale
            norm_throughput = m["throughput"] / (250 * 500) # heuristic norm
            
            # Avg violation per served session (Eq 52)
            avg_vio = m["latency_vio"] / m["served_count"] if m["served_count"] > 0 else 0
            
            reward = W1_THROUGHPUT * norm_throughput - \
                     W2_LATENCY * avg_vio - \
                     W3_CPU_WASTE * m["cpu_waste"] - \
                     W4_MEM_VIO * m["mem_vio"]
            rewards[edge.id] = reward

        return self._get_all_states(), rewards, slot_metrics
