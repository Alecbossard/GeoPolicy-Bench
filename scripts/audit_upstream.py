import json, urllib.request
from pathlib import Path

out = {}
for name in ["mujoco", "robosuite", "torch", "lerobot", "stable-baselines3", "diffusers"]:
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json"))
    out[name] = {
        "latest": d["info"]["version"],
        "python": d["info"]["requires_python"],
        "license": d["info"].get("license_expression") or d["info"]["license"],
    }
    print(name, out[name])
for repo in [
    "google-deepmind/mujoco",
    "ARISE-Initiative/robosuite",
    "huggingface/lerobot",
    "real-stanford/diffusion_policy",
    "YanjieZe/3D-Diffusion-Policy",
]:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}", headers={"User-Agent": "GeoPolicy-Bench"}
    )
    d = json.load(urllib.request.urlopen(req))
    out[repo] = {"license": d["license"], "url": d["html_url"]}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/commits?per_page=1",
        headers={"User-Agent": "GeoPolicy-Bench"},
    )
    out[repo]["head"] = json.load(urllib.request.urlopen(req))[0]["sha"]
    print(repo, out[repo]["license"], out[repo]["head"])
Path("docs/upstream_audit.json").write_text(json.dumps(out, indent=2))
