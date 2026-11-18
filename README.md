# cifar10-mlops

End-to-end CIFAR-10 image classification project: training, evaluation,
explainability, export, API, frontend, deployment, and monitoring.

Status: Phase 1 (setup). Results will be added as they are measured.

## Quickstart (Windows, laptop)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install tensorflow
    pip install -r requirements-dev.txt
    pip install -e .
    .\tasks.ps1 test