import os
from lib.helpers import get_qiskit_runtime_service
from lib.swap_cnot_experiment import PureSWAPCNOTExperimentsController
from dotenv import load_dotenv
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


if __name__ == "__main__":
    ensure_dirs_exist()
    log = Logger(prefix="main", should_save_to_file='main')
    log.clear()
    log.debug("main.py ==> PureSWAPCNOTExperimentsController")
    log.debug("SHOULD_RUN_ON_SIM: {}".format(SHOULD_RUN_ON_SIM))
    log.debug("NUMBER_OF_SHOTS: {}".format(NUMBER_OF_SHOTS))
    log.debug("BACKEND_NAME: {}".format(BACKEND_NAME))
    log.debug("NUMBER_OF_EXPERIMENTS: {}".format(NUMBER_OF_EXPERIMENTS))
    log.debug("Loading Qiskit Runtime Service...")
    service = get_qiskit_runtime_service("provider")
    log.debug("Qiskit Runtime Service loaded successfully.")
    log.debug("Loading {} backend...".format(BACKEND_NAME))
    backend = service.get_backend(BACKEND_NAME)
    log.debug("Backend loaded successfully.")

    log.debug(backend)

    for i in range(NUMBER_OF_EXPERIMENTS):
        run_experiment(i, backend, completion_handler_to_file, service)
