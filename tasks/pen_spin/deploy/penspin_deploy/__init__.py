"""Torch-free runtime for the OrcaHand v1 pen-spin policy.

The same control loop drives the real hand through orca_core or a MuJoCo twin
of the training scene; see ``python -m penspin_deploy --help``.
"""
