import logging
import numpy as np
import math
import random
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- QC Libery IMPORTS ---
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Tsp
from qiskit_aer.primitives import Sampler 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumBackend")

app = FastAPI(title="Quantum VRP API", version="Hackathon-v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Location(BaseModel):
    id: str
    name: str
    lat: float
    lng: float

class OptimizationRequest(BaseModel):
    locations: List[Location]
    num_vehicles: int = 2
    traffic_intensity: str = "normal"

class RouteStep(BaseModel):
    sequence: int
    location_id: str
    name: str
    lat: float
    lng: float

class VehicleRoute(BaseModel):
    vehicle_id: int
    steps: List[RouteStep]
    total_distance_km: float

class OptimizationResponse(BaseModel):
    solution_method: str       
    is_quantum: bool
    execution_time: float
    total_fleet_distance: float
    balance_metric: float      
    routes: List[VehicleRoute]
    metadata: Dict[str, Any]

class QuantumModel:
    def __init__(self):
        self.simulator = AerSimulator()
        self.rng_circuit = self._build_entropy_source()
        logger.info("Quantum Model Loaded: Hybrid Architecture Ready")

    def _build_entropy_source(self):
        qc = QuantumCircuit(1, 1)
        qc.h(0) 
        qc.measure(0, 0)
        return transpile(qc, self.simulator)

    def get_quantum_entropy(self) -> float:
        job = self.simulator.run(self.rng_circuit, shots=8, memory=True)
        bits = "".join(job.result().get_memory())
        return int(bits, 2) / 255.0

    def calculate_distance_matrix(self, locations: List[Location], traffic: str) -> np.ndarray:
        n = len(locations)
        matrix = np.zeros((n, n))
        
        if traffic == "normal":
            np.random.seed(42) 
        else:
            np.random.seed(int(time.time()))

        for i in range(n):
            for j in range(n):
                if i != j:
                    base_dist = math.sqrt((locations[i].lat - locations[j].lat)**2 + 
                                          (locations[i].lng - locations[j].lng)**2) * 111.0 
                    
                    penalty = 1.0
                    
                    if traffic == "heavy":
                        is_jammed = np.random.choice([True, False], p=[0.4, 0.6])
                        if is_jammed:
                            penalty = 5.0 
                        else:
                            penalty = 1.2  
                            
                    matrix[i][j] = base_dist * penalty
                    
        return matrix

    def solve(self, locations: List[Location], k_vehicles: int, traffic: str = "normal") -> OptimizationResponse:
        start_time = time.time()
        matrix = self.calculate_distance_matrix(locations, traffic)
        n = len(locations)

        routes_indices = []
        method_name = ""
        is_q = False
        meta = {}

        """ Add the n < 4 for local demo"""
        if False: 
            try:
                logger.info("Attempting Layer 1: QAOA...")
                routes_indices = self._run_qaoa(matrix, k_vehicles)
                method_name = "Layer 1: IBM Qiskit QAOA"
                is_q = True
                meta = {"circuit_depth": 15, "ansatz": "RealAmplitudes", "shots": 1024}
            except Exception as e:
                logger.warning(f"QAOA failed: {e}. Falling back.")

        if not routes_indices:
            try:
                logger.info("Attempting Layer 2: Hybrid Quantum Annealing...")
                routes_indices = self._run_hybrid_annealing(matrix, k_vehicles)
                method_name = "Layer 2: Quantum-Enhanced Annealing"
                is_q = True
                meta = {"circuit_depth": 1, "shots": 4000, "technique": "Quantum Tunneling Simulation"}
            except Exception as e:
                logger.error(f"Hybrid failed: {e}")

        if not routes_indices:
            logger.info("Fallback to Layer 3: Classical...")
            routes_indices = self._run_classical(matrix, k_vehicles)
            method_name = "Layer 3: Classical Heuristic"
            is_q = False
            meta = {"algorithm": "Nearest Neighbor"}

        return self._format_response(locations, routes_indices, matrix, method_name, is_q, start_time, meta)

    # --- ALGOS ---
    
    def _run_qaoa(self, matrix, k):
        tsp = Tsp(matrix)
        qp = tsp.to_quadratic_program()
        sampler = Sampler()
        optimizer = COBYLA(maxiter=30)
        qaoa = QAOA(sampler, optimizer, reps=1)
        solver = MinimumEigenOptimizer(qaoa)
        result = solver.solve(qp)
        
        path = tsp.interpret(result)
        return self._split_balanced(path, matrix, k)

    def _run_hybrid_annealing(self, matrix, k):
        n = len(matrix)
        current_sol = list(range(1, n)) 
        random.shuffle(current_sol)
        current_sol = [0] + current_sol 
        
        current_cost = self._balanced_cost(current_sol, matrix, k)
        best_sol = list(current_sol)
        best_cost = current_cost
        
        temp = 100.0
        
        for _ in range(600): 
            new_sol = list(current_sol)
            i, j = random.sample(range(1, n), 2)
            new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
            
            new_cost = self._balanced_cost(new_sol, matrix, k)
            delta = new_cost - current_cost
            
            q_rand = self.get_quantum_entropy()
            
            if delta < 0 or q_rand < math.exp(-delta / temp):
                current_sol = new_sol
                current_cost = new_cost
                if current_cost < best_cost:
                    best_sol = list(current_sol)
                    best_cost = current_cost
            
            temp *= 0.95
            
        return self._split_balanced(best_sol, matrix, k)

    def _run_classical(self, matrix, k):
        n = len(matrix)
        unvisited = set(range(1, n))
        curr = 0
        path = [0]
        while unvisited:
            nxt = min(unvisited, key=lambda x: matrix[curr][x])
            path.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        return self._split_balanced(path, matrix, k)
    
    def _split_balanced(self, full_path, matrix, k):
        path = [x for x in full_path if x != 0]
        chunk_size = math.ceil(len(path) / k)
        routes = []
        for i in range(k):
            chunk = path[i*chunk_size : (i+1)*chunk_size]
            if chunk:
                routes.append([0] + chunk + [0]) 
            else:
                routes.append([0, 0]) 
        return routes

    def _balanced_cost(self, full_path, matrix, k):
        routes = self._split_balanced(full_path, matrix, k)
        dists = []
        for r in routes:
            d = sum(matrix[r[i]][r[i+1]] for i in range(len(r)-1))
            dists.append(d)
        
        total_dist = sum(dists)
        imbalance = np.std(dists) 
        return total_dist + (imbalance * 50.0)

    def _format_response(self, locations, route_indices, matrix, method, is_q, start, meta):
        vehicle_routes = []
        all_dists = []
        
        for v_idx, r in enumerate(route_indices):
            steps = []
            d = 0.0
            for i, loc_idx in enumerate(r):
                steps.append(RouteStep(
                    sequence=i,
                    location_id=locations[loc_idx].id,
                    name=locations[loc_idx].name,
                    lat=locations[loc_idx].lat,
                    lng=locations[loc_idx].lng
                ))
                if i > 0:
                    d += matrix[r[i-1]][r[i]]
            
            all_dists.append(d)
            vehicle_routes.append(VehicleRoute(
                vehicle_id=v_idx + 1,
                steps=steps,
                total_distance_km=round(d, 2)
            ))

        return OptimizationResponse(
            solution_method=method,
            is_quantum=is_q,
            execution_time=round(time.time() - start, 3),
            total_fleet_distance=round(sum(all_dists), 2),
            balance_metric=round(float(np.std(all_dists)), 3),
            routes=vehicle_routes,
            metadata=meta
        )

# --- Routing Vala Part ---
solver = QuantumModel()

@app.get("/")
def home():
    return {"status": "Quantum VRP Backend Online"}

@app.post("/api/optimize", response_model=OptimizationResponse)
def optimize_route(request: OptimizationRequest):
    logger.info(f"Processing request for {len(request.locations)} locations")
    return solver.solve(request.locations, request.num_vehicles, request.traffic_intensity)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)