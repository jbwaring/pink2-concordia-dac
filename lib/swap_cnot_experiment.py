from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Options
from qiskit.quantum_info.operators import Operator
from qiskit_experiments.library import ProcessTomography


class PureSWAPCNOTExperimentsController:
    """
    Experiment Controller for long range CNOT using only SWAPs.
    """

    def __init__(self, backend, shots=1024, path=None, simulator=None, log=None):
        """
        Initializes the experiment controller with specified parameters.

        Args:
            backend: The backend to run experiments on or to use for coupling_map.
            shots (int): Number of shots for each experiment.
            path: List of adjacent qubit indices to be traversed. (Defaults to using shortest_undirected_path between indices 0 and backend.num_qubits - 1)
            simulator: The simulator backend, if not none, experiments will be run on the simulator.
            log: Logger for logging messages (defaults to python print).
        """
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
        """
        Runs the experiments on either a simulator or real backend.

        Args:
            **kwargs: Additional keyword arguments (not used at this time)
        """
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
        """
        Calculates the fidelity of the experiments.

        Returns:
            A list of fidelity values for the experiments.
        """
        return [x.analysis_results()[1].value for x in self.jobs]

    def build_path(self):
        """
        Builds the path for the quantum circuit. Defaults to using shortest_undirected_path between indices 0 and backend.num_qubits - 1
        """
        self.debug('-build_path')
        self.path = self.backend.coupling_map.shortest_undirected_path(
            0, self.backend.n_qubits - 1)
        self.debug('Path Nodes:\t%s', self.path)
        self.debug('Path length:\t%s', len(self.path))

    def build_circuits(self, **kwargs):
        """
        Builds the quantum circuits for the experiments.

        Args:
            **kwargs: Additional keyword arguments (not used yet)

        Returns:
            A list of quantum circuits.
        """
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
        """
        Builds a process tomography experiment for a given circuit.

        Args:
            circuit: The quantum circuit for which to build the experiment.

        Returns:
           ProcessTomography(): The process tomography experiment.
        """
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
        """
        Builds a single quantum circuit with a specific topological length and performs a CNOT between the first and last qubits.

        Args:
            topological_length (int): The length of the topology for the circuit. (> 2 swaps are introduced to satisfy coupling_map)

        Returns:
            QuantumCircuit()
        """
        c = QuantumCircuit(topological_length)
        for i in range(topological_length - 2):
            c.swap(i, i + 1)
        c.cnot(topological_length - 2, topological_length - 1)
        return c
