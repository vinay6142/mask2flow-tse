import torch

paths = [
    "checkpoints_v2/flow/flow_best_step228000.pt",
    "checkpoints_v2/flow/flow_best_step268000.pt",
    "checkpoints_v2/flow/flow_best_step270000.pt",
    "checkpoints_v2/flow/flow_best_step298000.pt",
    "checkpoints_v2/flow/flow_final_step300000.pt",
]

def average_key(key):
    avg = {}
    count = 0
    for p in paths:
        sd = torch.load(p, map_location="cpu")[key]
        if not avg:
            for k, v in sd.items():
                avg[k] = v.clone().float() if torch.is_tensor(v) else v
        else:
            for k, v in sd.items():
                if torch.is_tensor(v):
                    avg[k] += v.float()
        count += 1
    for k, v in avg.items():
        if torch.is_tensor(v):
            avg[k] = v / count
    return avg

out = {"model": average_key("model"), "step": "avg_228_268_270_298_300k"}
first = torch.load(paths[0], map_location="cpu")
if "ema" in first:
    out["ema"] = average_key("ema")

torch.save(out, "checkpoints_v2/flow/flow_avg_tail5.pt")
print("Saved: checkpoints_v2/flow/flow_avg_tail5.pt")