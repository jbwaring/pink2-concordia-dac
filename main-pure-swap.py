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

    def __init__(self, prefix, should_save_to_file=None):
        self.prefix = prefix
        if should_save_to_file is not None:
            self.file = os.path.join(
                os.getcwd(), "logs", "{}.log".format(should_save_to_file))
            with open(self.file, "w") as f:
                f.write("")

    def debug(self, *kwargs):
        if self.prefix:
            print("\u001b[34m[{}]\u001b[0m".format(self.prefix), *kwargs)
        else:
            print(*kwargs)

        self.save_to_file(*kwargs)

    def clear(self):
        os.system('clear')

    def save_to_file(self, *kwargs):
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

    with open(os.path.join(
            os.getcwd(), "results", "{}-exp={}-is_sim={}.dat".format(
                backend_name, experiment_no, is_sim
            )),  "w") as f:

        f.write(str(fidelities))


def ensure_dirs_exist():
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
