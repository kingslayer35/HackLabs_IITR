"""Restrict a requested inference run without rescoring preserved controls."""

ALL_FLOWS = ("F00", "F10", "F01", "F11")


def inference_flows(cfg):
    requested = cfg.get("inference_flows", ALL_FLOWS)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(ALL_FLOWS):
        raise ValueError("inference_flows must contain unique known flow names")
    return tuple(flow for flow in ALL_FLOWS if flow in requested)
