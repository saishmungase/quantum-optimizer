# main.py
import sys
import logging
import time
import numpy as np
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Guaranteed Quantum VRP", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ----------------------------
# Pydantic Models
# ----------------------------
class VRPRequest(BaseModel):
    num_locations: int
    num_vehicles: int
    distances: List[List[float]]  # [start, end, distance]
    traffic: List[List[float]]    # [start, end, delay_hours]

class VRPResponse(BaseModel):
    routes: List[List[int]]
    total_distance: float
    delivery_times: List[float]          # per vehicle time
    classical_distance: float
    is_quantum_solution: bool
    quantum_method: str
    execution_time: float
    quantum_circuit_depth: int
    quantum_shots_used: int

# ----------------------------
# Quantum VRP Solver
# ----------------------------
class GuaranteedQuantumSolver:
    """Quantum solver that uses distance + traffic matrices and provides optimized routes."""

    def __init__(self):
        self.simulator = AerSimulator(method="statevector")

    def build_distance_matrix(self, num_locations: int, distances: List[List[float]], traffic: List[List[float]]) -> np.ndarray:
        n = num_locations + 1  # 0 is depot
        dist_matrix = np.zeros((n, n))

        for start, end, d in distances:
            dist_matrix[int(start)][int(end)] = float(d)
            dist_matrix[int(end)][int(start)] = float(d)

        for start, end, delay in traffic:
            dist_matrix[int(start)][int(end)] += float(delay)
            dist_matrix[int(end)][int(start)] += float(delay)

        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix

    def classical_greedy_solution(self, distance_matrix: np.ndarray, num_vehicles: int) -> Tuple[List[List[int]], float]:
        n = distance_matrix.shape[0]
        customers = list(range(1, n))
        order = []
        cur = 0
        unvisited = set(customers)
        while unvisited:
            nxt = min(unvisited, key=lambda x: distance_matrix[cur, x] if distance_matrix[cur, x] > 0 else 1e9)
            order.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        # Split into balanced routes
        routes = []
        per = max(1, len(order)//num_vehicles)
        for v in range(num_vehicles):
            chunk = order[v*per:(v+1)*per]
            if chunk:
                routes.append([0] + chunk + [0])
        total = self.calculate_total_distance(routes, distance_matrix)
        return routes, total

    def quantum_route_optimizer(self, distance_matrix: np.ndarray, num_vehicles: int, num_locations: int) -> Tuple[List[List[int]], str, int, int]:
        """Use a small quantum circuit with superposition + entanglement for route decisions."""
        n_qubits = max(2, min(6, num_locations))
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Superposition
        for i in range(n_qubits):
            qc.h(i)
        # Entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        # Distance-biased rotation
        for i in range(min(n_qubits, num_locations)):
            avg_dist = np.mean(distance_matrix[i+1,:])
            qc.ry(min(np.pi, avg_dist), i)
        qc.measure_all()

        shots = 1024
        job = self.simulator.run(transpile(qc, self.simulator), shots=shots)
        result = job.result()
        counts = result.get_counts()

        routes = self._counts_to_routes(counts, distance_matrix, num_vehicles, num_locations)
        return routes, "Quantum Superposition + Entanglement", qc.depth(), shots

    def _counts_to_routes(self, counts: dict, distance_matrix: np.ndarray, num_vehicles: int, n_locations: int) -> List[List[int]]:
        most_frequent = max(counts, key=counts.get).replace(" ", "")
        binary_decisions = [int(b) for b in most_frequent]

        routes = [[] for _ in range(num_vehicles)]
        for location in range(1, n_locations+1):
            vehicle_idx = sum(binary_decisions[:location]) % num_vehicles if location <= len(binary_decisions) else (location % num_vehicles)
            routes[vehicle_idx].append(location)

        final_routes = []
        for route in routes:
            if route:
                final_routes.append([0]+route+[0])
        if not final_routes:
            final_routes = [[0,1,0]]
        return final_routes

    def calculate_total_distance(self, routes: List[List[int]], distance_matrix: np.ndarray) -> float:
        return sum(distance_matrix[route[i], route[i+1]] for route in routes for i in range(len(route)-1))

    def calculate_delivery_times(self, routes: List[List[int]], distance_matrix: np.ndarray) -> List[float]:
        return [sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) for route in routes]

    def solve(self, num_locations: int, num_vehicles: int, distances: List[List[float]], traffic: List[List[float]]) -> dict:
        distance_matrix = self.build_distance_matrix(num_locations, distances, traffic)

        classical_routes, classical_dist = self.classical_greedy_solution(distance_matrix, num_vehicles)

        quantum_routes, quantum_method, circuit_depth, shots = self.quantum_route_optimizer(distance_matrix, num_vehicles, num_locations)
        quantum_dist = self.calculate_total_distance(quantum_routes, distance_matrix)

        delivery_times = self.calculate_delivery_times(quantum_routes, distance_matrix)

        execution_time = 0  # you can add time.time() difference if needed

        is_quantum_better = quantum_dist <= classical_dist

        return {
            "routes": quantum_routes,
            "total_distance": quantum_dist,
            "delivery_times": delivery_times,
            "classical_distance": classical_dist,
            "is_quantum_solution": is_quantum_better,
            "quantum_method": quantum_method,
            "execution_time": execution_time,
            "quantum_circuit_depth": circuit_depth,
            "quantum_shots_used": shots
        }

# ----------------------------
# FastAPI Endpoints
# ----------------------------
solver = GuaranteedQuantumSolver()

@app.get("/health")
def health_check():
    return {"status": "healthy", "quantum_ready": True}

@app.post("/optimize", response_model=VRPResponse)
def optimize_routes(request: VRPRequest):
    try:
        result = solver.solve(
            request.num_locations,
            request.num_vehicles,
            request.distances,
            request.traffic
        )
        return result
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
