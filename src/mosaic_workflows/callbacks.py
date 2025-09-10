import os
import json


def checkpoint(output_dir: str, save_interval: int = 50, save_logits: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    def _cb(ctx: dict):
        if ctx.get("event") != "end_phase":
            return
        traj = ctx.get("trajectory") or []
        if not traj:
            return
        step = len(traj)
        if step % save_interval != 0:
            return
        path = os.path.join(output_dir, f"phase_{ctx.get('phase')}_step_{step}.json")
        with open(path, "w") as f:
            json.dump({"phase": ctx.get("phase"), "step": step}, f)
    return _cb


def memory_housekeeping():
    def _cb(ctx: dict):
        return
    return _cb


