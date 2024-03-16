import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi


def analytical_haar_frame_potential(nqubits: int) -> float:
    return 1 / (2 ** (nqubits - 1) * (2**nqubits + 1))


class GenerateCircuit:
    def __init__(self, circuit_type: str, nqubits: int, nlayers: int):
        self.circuit_type = circuit_type
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.count = 0

    def TPA(self, circuit: QuantumCircuit, params: np.ndarray) -> None:
        for i in range(self.nqubits):
            circuit.rx(params[i], i)
            circuit.ry(params[i], i)
        circuit.barrier()

    def HEA(self, circuit: QuantumCircuit, params: np.ndarray) -> None:
        for i in range(self.nqubits):
            circuit.rx(params[i], i)
            circuit.ry(params[i + self.nqubits], i)
        for i in range(self.nqubits - 1):
            circuit.cx(i, i + 1)
        circuit.barrier()

    def HEA2(self, circuit: QuantumCircuit, params: np.ndarray) -> None:
        for i in range(self.nqubits):
            circuit.rx(params[i], i)
            circuit.ry(params[i], i)
        for i in range(self.nqubits // 2):
            circuit.cx(2 * i, 2 * i + 1)
        if self.nqubits % 2 == 0:
            for i in range(self.nqubits // 2 - 1):
                circuit.cx(2 * i + 1, 2 * (i + 1))
        else:
            for i in range(self.nqubits // 2):
                circuit.cx(2 * i + 1, 2 * (i + 1))
        circuit.barrier()

    def ALT(self, circuit: QuantumCircuit, params: np.ndarray) -> None:
        if self.count % 2 == 0:
            for i in range(self.nqubits // 2):
                circuit.rx(params[i], 2 * i)
                circuit.ry(params[i], 2 * i)
                circuit.rx(params[i], 2 * i + 1)
                circuit.ry(params[i], 2 * i + 1)
                circuit.cz(2 * i, 2 * i + 1)
            circuit.barrier()
        else:
            if self.nqubits % 2 == 0:
                for i in range(self.nqubits // 2 - 1):
                    circuit.rx(params[i], 2 * i + 1)
                    circuit.ry(params[i], 2 * i + 1)
                    circuit.rx(params[i], 2 * (i + 1))
                    circuit.ry(params[i], 2 * (i + 1))
                    circuit.cz(2 * i + 1, 2 * (i + 1))
                circuit.barrier()
            else:
                for i in range(self.nqubits // 2):
                    circuit.rx(params[i], 2 * i + 1)
                    circuit.ry(params[i], 2 * i + 1)
                    circuit.rx(params[i], 2 * (i + 1))
                    circuit.ry(params[i], 2 * (i + 1))
                    circuit.cz(2 * i + 1, 2 * (i + 1))
                circuit.barrier()

    def generate_circuit(self, params: np.ndarray) -> QuantumCircuit:
        self.circuit = QuantumCircuit(self.nqubits)

        if self.circuit_type == "TPA":
            for i in range(self.nlayers):
                self.TPA(self.circuit, params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
        elif self.circuit_type == "HEA":
            for i in range(self.nlayers):
                self.HEA(self.circuit, params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
        elif self.circuit_type == "HEA2":
            for i in range(self.nlayers):
                self.HEA2(self.circuit, params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
        elif self.circuit_type == "ALT":
            for i in range(self.nlayers):
                self.ALT(self.circuit, params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
                self.count += 1
        else:
            raise ValueError("Invalid circuit type.")

        return self.circuit


class Expressibility1norm(GenerateCircuit):
    def __init__(self, circuit_type: str, nqubits: int, nlayers: int, nsamples: int):
        super().__init__(circuit_type, nqubits, nlayers)
        self.nsamples = nsamples

    def generate_circuit_state(self, params: np.ndarray) -> qi.DensityMatrix:
        circuit = self.generate_circuit(params)
        rho = qi.DensityMatrix.from_instruction(circuit)
        return rho

    def make_random_params(self) -> np.ndarray:
        params = np.random.uniform(0, 2 * np.pi, 2 * self.nqubits * self.nlayers)
        return params

    def generate_circuit_integrand(self) -> np.ndarray:
        rho = self.generate_circuit_state(self.make_random_params()).to_operator().data
        integrand = rho.tensor(rho)
        return integrand

    def generate_haar_integral(self) -> np.ndarray:
        d = 2**self.nqubits
        N = 2 ** (2 * self.nqubits)
        identity = np.eye(N)
        SWAP = np.zeros((N, N))
        for i in range(d):
            for j in range(d):
                SWAP[i * d + j, j * d + i] = 1
        integral = (identity + SWAP) / (d * (d + 1))
        return integral

    def expressibility(self) -> float:
        rho_integral = 0
        for _ in range(self.nsamples):
            rho_integral += self.generate_circuit_integrand()
        rho_integral /= self.nsamples

        haar_integral = self.generate_haar_integral()

        expressibility_ = np.linalg.norm(haar_integral - rho_integral, "nuc")
        # 'nuc' for 1-norm, 'fro' for 2-norm

        return expressibility_


class Expressibility2norm(GenerateCircuit):
    def __init__(self, circuit_type: str, nqubits: int, nlayers: int, nsamples: int):
        super().__init__(circuit_type, nqubits, nlayers)
        self.nsamples = nsamples

    def generate_circuit_state(self, params: np.ndarray) -> qi.Statevector:
        circuit = self.generate_circuit(params)
        state = qi.Statevector.from_instruction(circuit)
        return state

    def make_random_params(self) -> np.ndarray:
        params = np.random.uniform(0, 2 * np.pi, 2 * self.nqubits * self.nlayers)
        return params

    def random_inner_product(self) -> float:
        params1 = self.make_random_params()
        state1 = self.generate_circuit_state(params1)
        params2 = self.make_random_params()
        state2 = self.generate_circuit_state(params2)

        return np.abs(np.vdot(state1, state2))

    def circuit_frame_potential(self) -> float:
        samples = []
        for _ in range(self.nsamples):
            samples.append(self.random_inner_product() ** 4)
        return np.mean(samples)

    def expressibility(self) -> float:
        circuit_frame_potential_ = self.circuit_frame_potential()
        analytical_haar_frame_potential_ = analytical_haar_frame_potential(self.nqubits)

        expressibility_ = (circuit_frame_potential_ - analytical_haar_frame_potential_) ** 0.5
        return expressibility_
