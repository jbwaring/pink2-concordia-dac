import os
import logging
from lib.helpers import get_qiskit_runtime_service
from lib.swap_cnot_experiment import PureSWAPCNOTExperimentsController
from dotenv import load_dotenv
import datetime
from lib.helpers import *
from qiskit import *
from qiskit import QuantumCircuit, transpile
from qiskit_experiments.library import ProcessTomography
from qiskit_experiments.library.tomography import ProcessTomographyAnalysis
from qiskit.quantum_info.operators import Operator
from qiskit_ibm_runtime import Session
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Sampler, Options
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData, QobjExperimentHeader
from qiskit.providers.jobstatus import JobStatus
from qiskit_experiments.library.tomography import ProcessTomographyAnalysis
import random
import numpy as np
import requests
import json
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers import BackendV2Converter

load_dotenv()

SHOULD_RUN_ON_SIM = os.environ.get("SHOULD_RUN_ON_SIM")
NUMBER_OF_SHOTS = os.environ.get("NUMBER_OF_SHOTS")
BACKEND_NAME = os.environ.get("BACKEND_NAME")
NUMBER_OF_EXPERIMENTS = os.environ.get("NUMBER_OF_EXPERIMENTS")


class Logger:
    """
    Logger class for handling logs.

    Attributes:
        prefix (str): Prefix to be added to each log message.
        file (str): Path to the log file where the logs will be saved.
    """

    def __init__(self, prefix, should_save_to_file=None):
        """
        Initializes the Logger class with a prefix and an optional file saving feature.

        Args:
            prefix (str): Prefix for log messages.
            should_save_to_file (str, optional): Filename to save logs to. Defaults to None.
        """
        self.prefix = prefix
        if should_save_to_file is not None:
            self.file = os.path.join(
                os.getcwd(), "logs", "{}.log".format(should_save_to_file))
            with open(self.file, "w") as f:
                f.write("")

    def debug(self, *kwargs):
        """
        Logs a debug message.

        Args:
            *kwargs: Variable length argument list for log messages.
        """
        if self.prefix:
            print("\u001b[34m[{}]\u001b[0m".format(self.prefix), *kwargs)
        else:
            print(*kwargs)

        self.save_to_file(*kwargs)

    def clear(self):
        """Clears the console."""
        os.system('clear')

    def save_to_file(self, *kwargs):
        """
        Saves a log message to a file.

        Args:
            *kwargs: Variable length argument list for log messages.
        """
        if self.file:
            with open(self.file, "a") as f:
                if self.prefix:
                    f.write("[{}] ".format(self.prefix))
                    f.write(" ".join([str(x) for x in kwargs]))
                    f.write("\n")
                else:
                    f.write(" ".join([str(x) for x in kwargs]))
                    f.write("\n")


def run_experiment(experiment_no=0, backend=None, completion_handler=None, service=None):
    """
    Runs a quantum experiment.

    Args:
        experiment_no (int, optional): The experiment number. Defaults to 0.
        backend: The backend on which the experiment is run.
        completion_handler: Function to be called upon completion of the experiment.
        service: Service provider for running experiments (will be used to get simulator).

    Raises:
        Exception: If essential parameters are not provided.
    """
    if backend is None:
        raise Exception("Backend cannot be None.")
    if completion_handler is None:
        raise Exception("Completion Handler cannot be None.")

    if SHOULD_RUN_ON_SIM and service is None:
        raise Exception("Service cannot be None if running on simulator.")
    log = Logger("exp-{}".format(experiment_no),
                 should_save_to_file='run_experiment')
    log.debug("Starting Experiment on backend {}".format(backend.name))

    if (SHOULD_RUN_ON_SIM):
        log.debug("Running on simulator")
        controller = PureSWAPCNOTExperimentsController(
            backend=backend,
            shots=NUMBER_OF_SHOTS,
            path=None,
            simulator=service.get_backend('ibmq_qasm_simulator'),
            log=log
        )
    else:
        controller = PureSWAPCNOTExperimentsController(
            backend=backend,
            shots=NUMBER_OF_SHOTS,
            path=None,
            log=log
        )
    controller.run()
    log.debug("Analysing {}.".format(experiment_no))
    completion_handler(experiment_no, backend,
                       controller.fidelities(), SHOULD_RUN_ON_SIM)
    log.debug("Experiment {} completed.".format(experiment_no))


def completion_handler_to_file(experiment_no, backend_name, fidelities, is_sim):
    """
    Handles the completion of an experiment by saving results to a file.

    Args:
        experiment_no (int): The experiment number.
        backend_name (str): Name of the backend used.
        fidelities: The fidelities obtained from the experiment.
        is_sim (bool): Indicates if the experiment was run on a simulator.
    """
    with open(os.path.join(
            os.getcwd(), "results", "{}-exp={}-is_sim={}.dat".format(
                backend_name, experiment_no, is_sim
            )),  "w") as f:

        f.write(str(fidelities))


def ensure_dirs_exist():
    """
    Ensures that the necessary directories for logs and results exist.
    Creates them if they do not exist.
    """
    if not os.path.isdir(os.path.join(os.getcwd(), "logs")):
        print(
            "Creating logs directory... @{}".format(os.path.join(os.getcwd(), "logs")))
        os.makedirs(os.path.join(os.getcwd(), "logs"))
    else:
        print("logs directory already exists, skipping...")

    if not os.path.isdir(os.path.join(os.getcwd(), "results")):
        print(
            "Creating results directory... @{}".format(os.path.join(os.getcwd(), "results")))
        os.makedirs(os.path.join(os.getcwd(), "results"))
    else:
        print("results directory already exists, skipping...")


def get_cnot_errors_along_path(backend, path):
    cmap = backend.configuration().coupling_map
    two_q_error_map = {}
    cx_errors = []

    for gate, prop_dict in BackendV2Converter(backend).target.items():
        if prop_dict is None or None in prop_dict:
            continue
        for qargs, inst_props in prop_dict.items():
            if inst_props is None:
                continue
            if len(qargs) == 2:
                if inst_props.error is not None:
                    two_q_error_map[qargs] = max(
                        two_q_error_map.get(qargs, 0), inst_props.error
                    )

    for i in range(len(path) - 1):
        qubit1 = path[i]
        qubit2 = path[i + 1]
        err = two_q_error_map.get((qubit1, qubit2), 0)
        cx_errors.append(err)

    cx_errors = [100 * err for err in cx_errors]  # Convert to percentages

    return cx_errors


def serializable_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if isinstance(v, (int, float, str, bool, list, dict, tuple))}


def get_job_result(job, transpiled_circuits):
    results = []
    for i, meta in enumerate(job.result().metadata):

        dist = job.result().quasi_dists[i]
        result = {f"{hex(key)}": max(0, int(value * int(NUMBER_OF_SHOTS)))
                  for key, value in dist.items()}
        results.append(ExperimentResult(
            shots=NUMBER_OF_SHOTS,
            success=True,
            meas_level=2,
            data=ExperimentResultData(
                counts=result
            ),
            header=QobjExperimentHeader(
                creg_sizes=get_creg_sizes(transpiled_circuits[i].cregs),
                global_phase=transpiled_circuits[i].global_phase,
                memory_slots=get_memory_slots(transpiled_circuits[i].cregs),
                n_qubits=transpiled_circuits[i].num_qubits,
                name=transpiled_circuits[i].name,
                qreg_sizes=get_qreg_sizes(transpiled_circuits[i].qregs),
                metadata=transpiled_circuits[i].metadata),
            status=JobStatus.DONE,
            # seed_simulator=1,
            metadata={},
            # 'parallel_state_update': 16, 'sample_measure_time': 0.000973975, 'noise': 'ideal', 'batched_shots_optimization': False, 'measure_sampling': True, 'device': 'CPU', 'num_qubits': 2,
            #           'parallel_shots': 1, 'remapped_qubits': False, 'method': 'stabilizer', 'active_input_qubits': [1, 0], 'num_clbits': 3, 'input_qubit_map': [[1, 1], [0, 0]], 'fusion': {'enabled': False}
            time_taken=0.004612647
        ))
    return results


def status():
    return JobStatus.DONE


def get_creg_sizes(cregs):
    return [[creg.name, creg.size] for creg in cregs]


def get_memory_slots(cregs):
    return sum([creg.size for creg in cregs])


def get_qreg_sizes(qregs):
    return [[qreg.name, qreg.size] for qreg in qregs]


def run_jobs(circuits, noise_model=None, basis_gates=None, coupling_map=None, is_simulator=False, run_options=None):
    options = Options(
        shots=NUMBER_OF_SHOTS,
    )
    if (is_simulator):
        options.simulator = {
            "noise_model": noise_model,
            "basis_gates": basis_gates,
            "coupling_map": coupling_map,
            "seed_simulator": random.randint(1, 10000000)
        }
    options.skip_transpilation = True
    sampler = Sampler(options=options)
    return sampler.run(circuits=circuits, skip_transpilation=True, shots=NUMBER_OF_SHOTS,  **run_options)


def setRuns(number_of_runs):
    return [int(x) for x in np.linspace(0, number_of_runs, number_of_runs+1)]


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    ensure_dirs_exist()
    log = Logger(prefix="main", should_save_to_file='main')
    log.clear()
    # log = createLogger(name=f'Worker {worker_id}')

    log.debug("Loading Qiskit Account...")

    service = QiskitRuntimeService()

    # Backend Loading
    backend = service.backend(BACKEND_NAME)
    # backend = service.least_busy(simulator=False)
    log.debug("Backend:\t{}\nNumber of Qubits:\t{}".format(
        backend.name, backend.configuration().num_qubits))
    is_simulator = backend.configuration().simulator
    real_backend_name = 'ibm_algiers'
    should_use_noise_model = True
    log.debug(
        f"Running on backend: {backend.name}, which is a simulator: {is_simulator}")

    if (is_simulator):
        # real_backend = real_backend = service.backend(real_backend_name)
        real_backend = service.backend('ibm_algiers')
        coupling_map = real_backend.coupling_map
        basis_gates = real_backend.configuration().basis_gates
        number_qubits = real_backend.configuration().num_qubits
        noise_model = NoiseModel.from_backend(real_backend)
        error_model_backend = real_backend
    else:
        coupling_map = backend.configuration().coupling_map
        basis_gates = backend.configuration().basis_gates
        number_qubits = backend.configuration().num_qubits
        noise_model = None
        error_model_backend = backend

    path = coupling_map.shortest_undirected_path(0, number_qubits - 1)

    controller = PureSWAPCNOTExperimentsController(
        backend=backend,
        shots=NUMBER_OF_SHOTS,
        path=path,
        log=log,
        no_analysis=True
    )
    experiments = controller.build_circuits()

    iterations = np.linspace(1, len(path) - 1, num=len(path)-1, dtype=int)
    log.debug("Iterations:\t{}".format(iterations))

    log.debug("Experiments:\t{}".format(experiments))
    log.debug("Number of Experiments:\t{}".format(len(experiments)))
    experiments_transpiled = [transpile(exp.circuits(
    ), coupling_map=coupling_map, basis_gates=basis_gates) for exp in experiments]
    for experiment_transpiled in experiments_transpiled:
        log.debug(experiment_transpiled[0])
    USE_SAMPLER = True

    experiments_jobs = []
    with Session(service, backend) as session:
        log.debug("Running experiments on backend '{}'".format(backend.name))
        for idx, experiment_transpiled in enumerate(experiments_transpiled):

            experiments_jobs.append(
                run_jobs(experiment_transpiled, noise_model=noise_model, basis_gates=basis_gates, coupling_map=coupling_map, is_simulator=is_simulator, run_options=experiments[idx].run_options))
            log.debug("{}/{} --> Running experiment....".format(idx +
                                                                1, len(experiments_transpiled)))

    for idx, exp in enumerate(experiments_jobs):
        log.debug("{}/{} experiments successfully sampled, status: {}".format(idx +
                                                                              1, len(experiments_jobs), exp.status()))
    job_results = []
    for idx, job in enumerate(experiments_jobs):
        job_id = job.job_id()
        status = job.status()
        log.debug('{}/{} --> Job Results Convertion for Job ID: {} with Status: {}'.format(idx +
                                                                                           1, len(experiments_jobs), job_id, job.status()))
        log.debug('{}/{} --> Result Array: {}'.format(idx +
                                                      1, len(experiments_jobs), get_job_result(job, experiments_transpiled[idx])), )
        res = Result(
            backend=backend,
            backend_name=backend.name,
            backend_version=backend.version,
            qobj_id=lambda: job_id,
            job_id=lambda: job_id,
            status=status,
            success=True,
            results=get_job_result(job, experiments_transpiled[idx])
        )
        job_results.append(res)
    fidelities = []
    for idx, job in enumerate(experiments_jobs):

        analysis = ProcessTomographyAnalysis()
        analysis.set_options(measurement_qubits=[
            path[idx], path[idx+1]], preparation_qubits=[0, path[idx+1]], target=Operator([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]]))
        job._results = job_results[idx]
        exp_data = experiments[idx]._initialize_experiment_data()
        exp_data.add_jobs([job])
        exp_data = analysis.run(exp_data).block_for_results()
        log.debug('{}/{} --> Experiment Results for Job ID: {} with Status: {}'.format(idx +
                                                                                       1, len(experiments_jobs), job.job_id(), job.status()))
        log.debug('Fidelity: {}'.format(exp_data.analysis_results()[1].value))
        fidelities.append(exp_data.analysis_results()[1].value)
        log.debug("Fidelities:\t{}".format(fidelities))

    payload = {
        "job_id": job.job_id(),
        "status": {"_value_": "job has successfully run", "_name_": "DONE", "_sort_order_": 5},
        "results": {
            "value": fidelities
        },

        "program_id": 'sampler-swap',
        "metrics": {

            'cnot_errors': [],

            'experiment_id': 'swap-test',
            'shots': NUMBER_OF_SHOTS,
            'path': [n for n in path],
        },
        "backend": {
            "name": BACKEND_NAME,
            "is_simulator": backend.configuration().simulator,
            'error_model_backend': 'ibm_algiers',

        }
    }

    with open(os.path.join(
            os.getcwd(), "results", "{}.json".format(job.job_id())), 'w') as f:

        f.write(json.dumps(payload))

    # Post job to API
    # log("postJob", requests.post(sql_backend_url, json=payload))
