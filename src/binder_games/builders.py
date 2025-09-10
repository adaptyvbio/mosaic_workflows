from typing import Callable


from .optimizers import (
    minmax_logits,
    stackelberg_logits,
)


def build_minmax_phase(*, name: str, build_loss: Callable, steps: int, schedule=None, transforms=None, analyzers=None, analyze_every: int | None = None):
    return {
        "name": name,
        "build_loss": build_loss,
        "optimizer": minmax_logits,
        "steps": int(steps),
        "schedule": schedule,
        "transforms": transforms or {},
        "analyzers": analyzers or [],
        "analyze_every": int(analyze_every) if analyze_every is not None else 0,
    }


def build_stackelberg_phase(*, name: str, build_loss: Callable, steps: int, schedule=None, transforms=None, analyzers=None, analyze_every: int | None = None):
    return {
        "name": name,
        "build_loss": build_loss,
        "optimizer": stackelberg_logits,
        "steps": int(steps),
        "schedule": schedule,
        "transforms": transforms or {},
        "analyzers": analyzers or [],
        "analyze_every": int(analyze_every) if analyze_every is not None else 0,
    }


def build_multi_adversary_phase(*, name: str, build_loss: Callable, steps: int, schedule=None, transforms=None, analyzers=None, analyze_every: int | None = None):
    """Alias to minmax; multi-adversary handled by loss_fn or schedule.
    """
    return build_minmax_phase(
        name=name,
        build_loss=build_loss,
        steps=steps,
        schedule=schedule,
        transforms=transforms,
        analyzers=analyzers,
        analyze_every=analyze_every,
    )


