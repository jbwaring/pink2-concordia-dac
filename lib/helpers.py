from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
import os


def get_qiskit_runtime_service(type='provider'):
    """
    Returns a Qiskit Runtime service object.

    Returns:
        QiskitRuntimeService: The Qiskit Runtime service object.
    """

    if type == 'runtime':
        service = QiskitRuntimeService(channel=os.environ.get(
            "CHANNEL"), instance=os.environ.get("INSTANCE"))
        acc = service.active_account()
        print(
            "Qiskit Account loaded successfully. \nChannel:\t{}\nInstance:\t{}".format(
                acc['channel'], acc['instance'])
        )
        return service
    elif type == 'provider':
        service = IBMProvider(
            instance=os.environ.get("INSTANCE")
        )
        acc = service.active_account()
        print(
            "Qiskit Account loaded successfully. \nChannel:\t{}\nInstance:\t{}".format(
                acc['channel'], acc['instance'])
        )
        return service
