from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Options
from qiskit.quantum_info.operators import Operator
from qiskit_experiments.library import ProcessTomography


class PureSWAPCNOTExperimentsController:

    def __init__(self, backend, shots=1024, path=None, simulator=None, log=None):
        if log is None:
            self.debug = lambda *args: print(*args)
        else:
            self.debug = log.debug
        self.backend = backend
        self.sim = simulator
        self.shots = shots
        if path is None:
            self.debug('No path provided, building path')
            self.build_path()

    def run(self, **kwargs):
        experiments = self.build_circuits(**kwargs)
        self.jobs = []

        if self.sim:
            with self.sim.open_session() as session:
                self.session = session
                for idx, exp in enumerate(experiments):
                    self.debug('Running experiment: {}'.format(idx))
                    exp.set_transpile_options(
                        optimization_level=0,
                        basis_gates=self.backend.operation_names
                    )
                    self.jobs.append(exp.run(
                        backend=self.sim,
                        shots=self.shots,
                    ))

        else:
            with self.backend.open_session() as session:
                self.session = session
                for idx, exp in enumerate(experiments):
                    self.debug('Running experiment: {}'.format(idx))
                    exp.set_transpile_options(
                        optimization_level=0,
                    )
                    self.jobs.append(exp.run(
                        backend=self.backend,
                        shots=self.shots,
                    ).block_for_results())

    def fidelities(self):
        return [x.analysis_results()[1].value for x in self.jobs]

    def build_path(self):
        self.debug('-build_path')
        self.path = self.backend.coupling_map.shortest_undirected_path(
            0, self.backend.n_qubits - 1)
        self.debug('Path Nodes:\t%s', self.path)
        self.debug('Path length:\t%s', len(self.path))

    def build_circuits(self, **kwargs):
        self.debug('-build_circuits')
        circuits = []
        for i in range(len(self.path) - 1):
            circuits.append(self.build_circuit(i + 2, **kwargs))

        # Warning: QISKIT IS USING THE OTHER ENDIAN SO USE OTHER CNOT
        self.target_operation = Operator([[1, 0, 0, 0],
                                          [0, 0, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 1, 0, 0]])
        [self.debug('Experiment {}:\n{}'.format(i, c.draw('text')))
         for i, c in enumerate(circuits)]
        return [self.build_tomo_experiment(c) for c in circuits]

    def build_tomo_experiment(self, circuit):
        self.debug('-build_tomo_experiment for circuit length %s',
                   circuit.num_qubits)
        self.debug('-build_tomo_experiment physical_qubits {}\nPreparation Indices:\t{}\nMeasurement Indices:\t{}.'.format(
            self.path[:circuit.num_qubits], [0,
                                             circuit.num_qubits - 1], [circuit.num_qubits - 2,
                                                                       circuit.num_qubits - 1]))

        return ProcessTomography(
            circuit=circuit,
            backend=self.backend,
            target=self.target_operation,
            preparation_indices=[0,
                                 circuit.num_qubits - 1],
            measurement_indices=[circuit.num_qubits - 2,
                                 circuit.num_qubits - 1],
            physical_qubits=self.path[:circuit.num_qubits],

        )

    def build_circuit(self, topological_length):
        c = QuantumCircuit(topological_length)
        for i in range(topological_length - 2):
            c.swap(i, i + 1)
        c.cnot(topological_length - 2, topological_length - 1)
        return c
