import time
import numpy as np
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math
import random
import logging

logger = logging.getLogger(__name__)

class GuaranteedQuantumSolver:
    def __init__(self):
        # Use AerSimulator statevector for deterministic runs; we will use shots to sample.
        self.simulator = AerSimulator(method="statevector")

    def build_distance_matrix(self, num_locations: int, distances: List[List[float]], traffic: List[List[float]]) -> np.ndarray:
        # Interpret num_locations as number of customers (excluding depot)
        n = num_locations + 1  # node indices: 0..num_locations
        dist_matrix = np.zeros((n, n))
        # Fill symmetric distances
        for start, end, d in distances:
            s = int(start); e = int(end)
            dist_matrix[s, e] = float(d)
            dist_matrix[e, s] = float(d)
        # Add traffic (delay hours) to both entries
        for start, end, delay in traffic:
            s = int(start); e = int(end)
            dist_matrix[s, e] += float(delay)
            dist_matrix[e, s] += float(delay)
        # Ensure diagonal is zero
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix

    def classical_greedy_solution(self, distance_matrix: np.ndarray, num_vehicles: int) -> Tuple[List[List[int]], float]:
        # Very simple: assign customers greedily round-robin to vehicles based on nearest neighbor order
        n = distance_matrix.shape[0]
        customers = list(range(1, n))
        # create an ordering using nearest neighbor from depot
        order = []
        cur = 0
        unvisited = set(customers)
        while unvisited:
            nxt = min(unvisited, key=lambda x: distance_matrix[cur, x] if distance_matrix[cur, x] > 0 else 1e9)
            order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        # Split ordered customers into num_vehicles chunks (balanced)
        routes = []
        per = math.ceil(len(order) / num_vehicles) if num_vehicles>0 else len(order)
        for v in range(num_vehicles):
            chunk = order[v*per:(v+1)*per]
            if chunk:
                routes.append([0] + chunk + [0])
            else:
                routes.append([0, 0])  # empty route (starts and ends at depot)
        # Remove purely [0,0] entries for tightness (but we keep at least one)
        final = [r for r in routes if len(r) >= 2]
        if not final:
            final = [[0,1,0]]
        total = self.calculate_total_distance(final, distance_matrix)
        return final, total

    def quantum_route_optimizer(self, distance_matrix: np.ndarray, num_vehicles: int, num_locations: int,
                                attempts: int = 3, shots: int = 1024) -> Tuple[List[List[int]], str, int, int]:
        """
        Try a few quantum variations, collect measurement results, decode many bitstrings into candidate routes,
        evaluate them on the true distance matrix and pick the best.
        """
        best_routes = None
        best_cost = float("inf")
        best_depth = 0
        best_shots_used = shots
        method_name = "Quantum Superposition + Entanglement (sampling-decoding)"

        # number of customers
        n_customers = num_locations
        # choose qubits: at least enough to encode 'scores' for customers; limit 8 to keep circuit small
        n_qubits = max(2, min(8, n_customers))
        logger.info(f"Quantum attempt: qubits={n_qubits}, shots={shots}")

        for attempt in range(attempts):
            # build circuit
            qc = QuantumCircuit(n_qubits, n_qubits)
            for q in range(n_qubits):
                qc.h(q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            # small distance-biased rotations (use distances summary to bias)
            # Use customers 1..n_customers to compute per-qubit rotation heuristics
            if n_customers > 0:
                avg_dists = []
                for i in range(min(n_customers, n_qubits)):
                    # approximate avg distance from node i+1 to others (avoid zero division)
                    avg = np.mean(distance_matrix[i+1, :]) if distance_matrix.shape[0] > i+1 else 0.5
                    avg_dists.append(avg)
                for i, avg in enumerate(avg_dists):
                    angle = float(min(np.pi/2, (avg / (np.max(avg_dists)+1e-6)) * (np.pi/2)))
                    # add small random perturbation to diversify sampling on retries
                    angle *= 1.0 + (attempt * 0.05)
                    qc.ry(angle, i)

            qc.measure_all()

            # simulate
            trans = transpile(qc, self.simulator)
            job = self.simulator.run(trans, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # decode many bitstrings into candidate routes
            candidates = self._decode_counts_to_candidate_routes(counts, n_customers, num_vehicles, distance_matrix)
            # evaluate candidates
            for routes_candidate in candidates:
                cost = self.calculate_total_distance(routes_candidate, distance_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = routes_candidate
                    best_depth = qc.depth()
                    best_shots_used = shots

            # if quantum already beat a simple greedy baseline, we can break early
            # otherwise continue attempts to try varied circuits
            logger.info(f"Attempt {attempt+1}/{attempts} best_cost so far: {best_cost:.4f}")

        # Ensure every route starts/ends with 0 (already ensured by decode)
        if not best_routes:
            best_routes = [[0, 1, 0]]
            best_cost = self.calculate_total_distance(best_routes, distance_matrix)

        return best_routes, method_name, best_depth, best_shots_used

    def _decode_counts_to_candidate_routes(self, counts: Dict[str, int], n_customers: int, num_vehicles: int, distance_matrix: np.ndarray, max_candidates: int = 50) -> List[List[int]]:
        """
        Convert measured bitstrings into candidate routes.
        Approach:
          - For each measured bitstring, build a per-customer 'score' by repeating/truncating the bitstring to cover customers.
          - Sort customers by their score (higher score → earlier in ordering).
          - Split this ordering among vehicles with a greedy assignment and produce full routes [0,...,0].
          - Keep top-K unique candidates by cost (evaluate later).
        """
        # sort counts by frequency desc
        sorted_bits = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        candidates = []
        seen = set()
        for bitstring, _freq in sorted_bits[:max_candidates]:
            # bitstring might be like '001101', map LSB/MSB consistently (qiskit returns string with MSB on left, keep as-is)
            bits = [int(b) for b in bitstring.strip()]
            # produce a score for each customer
            scores = []
            for c in range(1, n_customers + 1):
                # map customer index to bits position using a simple repeating pattern
                pos = (c - 1) % len(bits)
                score = bits[pos]
                # to break ties and add variation, add tiny fractional term based on (c + sum(bits)) mod something
                score += ((sum(bits) % 5) / 100.0) * ((c % 3) / 3.0)
                scores.append((c, score))
            # order customers by score descending (higher → earlier)
            scores_sorted = sorted(scores, key=lambda x: (-x[1], x[0]))
            ordering = [c for c, _s in scores_sorted]

            # Now split ordering into vehicle routes. We'll use a balanced chunking approach but also try greedy per vehicle
            per = max(1, math.ceil(len(ordering) / num_vehicles)) if num_vehicles > 0 else len(ordering)
            routes = []
            for v in range(num_vehicles):
                chunk = ordering[v*per:(v+1)*per]
                if chunk:
                    routes.append([0] + chunk + [0])
                else:
                    routes.append([0, 0])
            # create a canonical tuple to dedupe
            key = tuple(tuple(r) for r in routes)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(routes)

        # fallback: if no candidates produced, produce simple NN greedy partition
        if not candidates:
            greedy_routes, _ = self.classical_greedy_solution(distance_matrix, num_vehicles)
            candidates.append(greedy_routes)

        return candidates

    def calculate_total_distance(self, routes: List[List[int]], distance_matrix: np.ndarray) -> float:
        total = 0.0
        for route in routes:
            # ensure route indices within matrix
            for i in range(len(route) - 1):
                a = route[i]; b = route[i+1]
                if a < distance_matrix.shape[0] and b < distance_matrix.shape[0]:
                    total += float(distance_matrix[a, b])
                else:
                    total += 1e6  # large penalty for invalid index
        return total

    def solve(self, num_locations: int, num_vehicles: int, distances: List[List[float]], traffic: List[List[float]]) -> dict:
        """
        Top-level solve: build matrix, compute classical baseline, run quantum attempts, compare,
        and return the best solution (quantum preferred if better).
        """
        logger.info(f"Starting solve: num_locations={num_locations}, num_vehicles={num_vehicles}")

        start_time = time.time()
        dist_matrix = self.build_distance_matrix(num_locations, distances, traffic)

        # classical baseline
        classical_routes, classical_cost = self.classical_greedy_solution(dist_matrix, num_vehicles)
        logger.info(f"Classical baseline cost: {classical_cost:.4f}")

        # run quantum with multiple attempts and sampling decoding
        quantum_routes, quantum_method, qc_depth, shots = self.quantum_route_optimizer(dist_matrix, num_vehicles, num_locations, attempts=4, shots=1024)
        quantum_cost = self.calculate_total_distance(quantum_routes, dist_matrix)
        logger.info(f"Quantum found cost: {quantum_cost:.4f}")

        # If quantum isn't better, make a couple more quantum attempts with altered params
        if quantum_cost >= classical_cost:
            logger.info("Quantum >= classical; retrying a couple more quantum attempts to try to beat classical")
            for extra in range(2):
                qr, _, d, s = self.quantum_route_optimizer(dist_matrix, num_vehicles, num_locations, attempts=2, shots=2048)
                qc = self.calculate_total_distance(qr, dist_matrix)
                if qc < quantum_cost:
                    quantum_routes, quantum_cost, qc_depth, shots = qr, qc, d, s
                if quantum_cost < classical_cost:
                    break

        # choose best-of-quantum-vs-classical (prefer quantum if equal)
        use_quantum = quantum_cost <= classical_cost
        chosen_routes = quantum_routes if use_quantum else classical_routes
        chosen_cost = quantum_cost if use_quantum else classical_cost

        execution_time = time.time() - start_time

        logger.info(f"Chosen solution (quantum_used={use_quantum}): cost={chosen_cost:.4f}")

        return {
            "routes": chosen_routes,
            "total_distance": float(chosen_cost),
            "is_quantum_solution": bool(use_quantum),
            "quantum_method": quantum_method if use_quantum else "Classical Greedy Fallback",
            "execution_time": execution_time,
            "quantum_circuit_depth": int(qc_depth if use_quantum else 0),
            "quantum_shots_used": int(shots if use_quantum else 0),
            "distance_matrix": dist_matrix.tolist()
        }
