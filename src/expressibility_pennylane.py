import pennylane as qml
from pennylane import numpy as np


def analytical_haar_frame_potential(nqubits: int) -> float:
    return 1 / (2 ** (nqubits - 1) * (2**nqubits + 1))


class GenerateCircuit_pennylane:
    def __init__(self, circuit_type: str, nqubits: int, nlayers: int):
        self.circuit_type = circuit_type
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.count = 0

    def TPA(self, params: np.ndarray) -> None:
        for i in range(self.nqubits):
            qml.RX(params[i], wires=i)
            qml.RY(params[i], wires=i)

    def HEA(self, params: np.ndarray) -> None:
        for i in range(self.nqubits):
            qml.RX(params[i], wires=i)
            qml.RY(params[i + self.nqubits], wires=i)
        for i in range(self.nqubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def HEA2(self, params: np.ndarray) -> None:
        for i in range(self.nqubits):
            qml.RX(params[i], wires=i)
            qml.RY(params[i], wires=i)
        for i in range(self.nqubits // 2):
            qml.CNOT(wires=[2 * i, 2 * i + 1])
        if self.nqubits % 2 == 0:
            for i in range(self.nqubits // 2 - 1):
                qml.CNOT(wires=[2 * i + 1, 2 * (i + 1)])
        else:
            for i in range(self.nqubits // 2):
                qml.CNOT(wires=[2 * i + 1, 2 * (i + 1)])

    def ALT(self, params: np.ndarray) -> None:
        if self.count % 2 == 0:
            for i in range(self.nqubits // 2):
                qml.RX(params[i], wires=2 * i)
                qml.RY(params[i], wires=2 * i)
                qml.RX(params[i], wires=2 * i + 1)
                qml.RY(params[i], wires=2 * i + 1)
                qml.CZ(wires=[2 * i, 2 * i + 1])
        else:
            if self.nqubits % 2 == 0:
                for i in range(self.nqubits // 2 - 1):
                    qml.RX(params[i], wires=2 * i + 1)
                    qml.RY(params[i], wires=2 * i + 1)
                    qml.RX(params[i], wires=2 * (i + 1))
                    qml.RY(params[i], wires=2 * (i + 1))
                    qml.CZ(wires=[2 * i + 1, 2 * (i + 1)])
            else:
                for i in range(self.nqubits // 2):
                    qml.RX(params[i], wires=2 * i + 1)
                    qml.RY(params[i], wires=2 * i + 1)
                    qml.RX(params[i], wires=2 * (i + 1))
                    qml.RY(params[i], wires=2 * (i + 1))
                    qml.CZ(wires=[2 * i + 1, 2 * (i + 1)])
    
    def generate_circuit(self, params: np.ndarray) -> None:
        if self.circuit_type == "TPA":
            for i in range(self.nlayers):
                self.TPA(params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
                qml.Barrier(only_visual=True, wires=range(self.nqubits))
        elif self.circuit_type == "HEA":
            for i in range(self.nlayers):
                self.HEA(params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
                qml.Barrier(only_visual=True, wires=range(self.nqubits))
        elif self.circuit_type == "HEA2":
            for i in range(self.nlayers):
                self.HEA2(params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
                qml.Barrier(only_visual=True, wires=range(self.nqubits))
        elif self.circuit_type == "ALT":
            for i in range(self.nlayers):
                self.ALT(params[2 * self.nqubits * i : 2 * self.nqubits * (i + 1)])
                qml.Barrier(only_visual=True, wires=range(self.nqubits))
                self.count += 1
        else:
            raise ValueError("Invalid circuit type.")
    
    def draw_circuit(self, nqubits: int, nlayers: int) -> None:
        self.nqubits = nqubits
        self.nlayers = nlayers
        params = np.random.uniform(0, 2 * np.pi, 2 * self.nqubits * self.nlayers)
        
        dev = qml.device("lightning.qubit", wires=self.nqubits)
        @qml.qnode(dev)
        def circuit(params: np.ndarray):
            self.generate_circuit(params)
            return qml.state()
        
        qml.draw_mpl(circuit)(params)


class Expressibility1norm_pennylane(GenerateCircuit_pennylane):
    def __init__(self, circuit_type: str, nqubits: int, nlayers: int, nsamples: int):
        super().__init__(circuit_type, nqubits, nlayers)
        self.nsamples = nsamples

    def generate_circuit_density_matrix(self, params: np.ndarray) -> qml.density_matrix:
        dev = qml.device("lightning.qubit", wires=self.nqubits)

        @qml.qnode(dev)
        def circuit_density_matrix(params: np.ndarray) -> qml.density_matrix:
            self.generate_circuit(params)
            return qml.density_matrix(wires=range(self.nqubits))
        return circuit_density_matrix(params)

    def make_random_params(self) -> np.ndarray:
        params = np.random.uniform(0, 2 * np.pi, 2 * self.nqubits * self.nlayers)
        return params

    def generate_circuit_integrand(self) -> np.ndarray:
        density_matrix = self.generate_circuit_density_matrix(self.make_random_params())
        integrand = np.kron(density_matrix, density_matrix)
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


class Expressibility2norm_pennylane(GenerateCircuit_pennylane):
    def __init__(self, circuit_type: str, nqubits: int, nlayers: int, nsamples: int):
        super().__init__(circuit_type, nqubits, nlayers)
        self.nsamples = nsamples

    def generate_circuit_state(self, params: np.ndarray) -> qml.state:
        dev = qml.device("lightning.qubit", wires=self.nqubits)

        @qml.qnode(dev)
        def circuit_state(params_: np.ndarray) -> qml.state:
            self.generate_circuit(params_)
            return qml.state()
        return circuit_state(params)

    def make_random_params(self) -> np.ndarray:
        params = np.random.uniform(0, 2 * np.pi, 2 * self.nqubits * self.nlayers)
        return params

    def random_inner_product(self) -> float:
        params1 = self.make_random_params()
        state1 = self.generate_circuit_state(params1)
        params2 = self.make_random_params()
        state2 = self.generate_circuit_state(params2)

        return np.abs(np.vdot(state1, state2))

    def circuit_frame_potential(self) -> tuple[float, float]:
        samples = []
        for _ in range(self.nsamples):
            samples.append(self.random_inner_product() ** 4)
        return np.mean(samples), np.std(samples)

    def expressibility(self) -> float:
        circuit_frame_potential_ = self.circuit_frame_potential()
        analytical_haar_frame_potential_ = analytical_haar_frame_potential(self.nqubits)

        expressibility_ = (circuit_frame_potential_ - analytical_haar_frame_potential_) ** 0.5
        return expressibility_
