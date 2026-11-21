#!/usr/bin/env python3
"""Workspace hygiene, warm build caches and idle shutdown for RunMat's GCP runners.

The same file runs on the Linux and Windows runner VMs. The runner's job hooks,
a daily prune timer and an idle timer call it, and workflow steps call
`acquire` and `commit` to use a warm Cargo target directory.

Disk layout (paths come from /etc/ci-runner.json or C:\\ci\\ci-runner.json):
  work disk   <work>/<slot>/{w,home,tmp}  wiped at every job start and end
  cache disk  <cache>/lanes/<lane>/{target,sources.json,dirty,last-used}
              <cache>/{cargo,rustup,vcpkg-runmat,toolcache,npm,pip}  shared tool caches
              <cache>/locks/<lane>.lock

Nothing that a job produces survives on the work disk. The cache disk holds only
the directories above; `evict` removes whole lanes, least recently used first,
until the cache is under its cap.
"""

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

WINDOWS = platform.system() == "Windows"
CONFIG = Path(os.environ.get("CI_RUNNER_CONFIG") or (r"C:\ci\ci-runner.json" if WINDOWS else "/etc/ci-runner.json"))
GiB = 1024 ** 3


def log(message):
    print(f"[ci-runner] {message}", flush=True)


def config():
    data = json.loads(CONFIG.read_text())
    data["work"] = Path(data["work"])
    data["cache"] = Path(data["cache"])
    data["state"] = Path(data["state"])
    return data


def free_bytes(path):
    return shutil.disk_usage(path).free


def tree_size(path):
    total = 0
    for root, dirs, files in os.walk(path, onerror=lambda error: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _make_writable(function, path, _info):
    """rmtree error handler: clear read-only flags and retry removals; ignore the rest."""
    if function not in (os.unlink, os.remove, os.rmdir):
        return
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    except OSError:
        pass


def remove_tree(path):
    """Remove a path without following links or junctions out of it."""
    path = Path(path)
    if not os.path.lexists(path):
        return
    if path.is_symlink() or (WINDOWS and _is_junction(path)):
        os.unlink(path) if not WINDOWS else os.rmdir(path)
        return
    mode = os.lstat(path).st_mode
    if stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode):
        # The runner's own diagnostic sockets live in the slot's TMPDIR.
        return
    if not stat.S_ISDIR(mode):
        os.unlink(path)
        return
    if WINDOWS:
        # robocopy /MIR from an empty directory removes long paths and read-only
        # files that shutil cannot; it does not follow junctions.
        empty = path.parent / f".empty-{os.getpid()}"
        empty.mkdir(exist_ok=True)
        subprocess.run(["robocopy", str(empty), str(path), "/MIR", "/XJ", "/R:1", "/W:1",
                        "/NFL", "/NDL", "/NJH", "/NJS", "/NP"], stdout=subprocess.DEVNULL)
        empty.rmdir()
    shutil.rmtree(path, onerror=_make_writable)


def _is_junction(path):
    try:
        return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (OSError, AttributeError):
        return False


def empty_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        remove_tree(child)


def slot_dirs(cfg, slot):
    base = cfg["work"] / slot
    return {"w": base / "w", "home": base / "home", "tmp": base / "tmp", "cold": base / "cold-target"}


def touch_activity(cfg):
    cfg["state"].mkdir(parents=True, exist_ok=True)
    (cfg["state"] / "last-activity").write_text(str(time.time()))


# ---------------------------------------------------------------- job hooks

TRUSTED_EVENTS = {"push", "workflow_dispatch", "schedule", "release", "merge_group", "repository_dispatch"}


def check_trusted_event():
    """Refuse code from forks, whatever the workflow file says.

    A fork pull request runs the workflow file from the fork, so a fork can name
    this runner's label. The hook is installed on the VM and the fork cannot
    change it. Same-repository pull requests need write access to push the branch.
    """
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if event in TRUSTED_EVENTS:
        return
    if event == "pull_request":
        payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        head = (payload.get("pull_request") or {}).get("head", {}).get("repo") or {}
        if head.get("full_name") == repository and not head.get("fork", False):
            return
        raise SystemExit(f"Refusing a pull request from {head.get('full_name')!r}: "
                         "fork pull requests run on GitHub-hosted runners only.")
    raise SystemExit(f"Refusing event {event!r} on a self-hosted runner.")


def job_started(cfg, slot):
    touch_activity(cfg)
    check_trusted_event()
    dirs = slot_dirs(cfg, slot)
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace and Path(workspace).exists():
        empty_dir(workspace)
    for name in ("home", "tmp", "cold"):
        empty_dir(dirs[name])
    ensure_floor(cfg)
    log(f"slot {slot}: work disk {free_bytes(cfg['work']) / GiB:.0f} GiB free, "
        f"cache disk {free_bytes(cfg['cache']) / GiB:.0f} GiB free")


def ensure_floor(cfg):
    if free_bytes(cfg["cache"]) < cfg["cache_floor_gib"] * GiB or cache_used(cfg) > cfg["cache_cap_gib"] * GiB:
        evict(cfg)
    problems = []
    for name, floor in (("work", cfg["work_floor_gib"]), ("cache", cfg["cache_floor_gib"]),
                        ("system", cfg["system_floor_gib"])):
        path = cfg["work"] if name == "work" else cfg["cache"] if name == "cache" else Path(cfg["system_root"])
        free = free_bytes(path)
        if free < floor * GiB:
            problems.append(f"{name} disk {path} has {free / GiB:.0f} GiB free, below the {floor} GiB floor")
    if problems:
        raise SystemExit("Refusing the job: " + "; ".join(problems))


def job_completed(cfg, slot):
    release_locks(cfg, slot)
    dirs = slot_dirs(cfg, slot)
    for name in ("w", "home", "tmp", "cold"):
        empty_dir(dirs[name])
    # Credentials a job may have written into the shared Cargo home.
    for name in ("credentials", "credentials.toml"):
        try:
            (cfg["cache"] / "cargo" / name).unlink()
        except FileNotFoundError:
            pass
    enforce_lane_caps(cfg)
    if cache_used(cfg) > cfg["cache_cap_gib"] * GiB:
        evict(cfg)
    touch_activity(cfg)
    log(f"slot {slot}: workspace wiped; work disk {free_bytes(cfg['work']) / GiB:.0f} GiB free, "
        f"cache {cache_used(cfg) / GiB:.0f} GiB used of a {cfg['cache_cap_gib']} GiB cap")


def slot_start(cfg, slot):
    """Called before a runner service starts (and so after every boot).

    A lane this slot still holds was in use when the runner or the VM stopped
    without the job-completed hook. A hard stop can leave files the job was
    writing truncated or zero-filled while Cargo's fingerprints call them fresh,
    so delete those lanes instead of reusing them.
    """
    locks = cfg["cache"] / "locks"
    for lock in locks.glob("*.lock") if locks.exists() else []:
        try:
            owner = json.loads(lock.read_text()).get("slot")
        except (OSError, ValueError):
            owner = slot
        if owner == slot:
            log(f"lane {lock.stem} was in use when slot {slot} stopped; deleting it")
            remove_tree(lanes_root(cfg) / lock.stem)
            lock.unlink(missing_ok=True)
    dirs = slot_dirs(cfg, slot)
    for name in ("w", "home", "tmp", "cold"):
        empty_dir(dirs[name])


# ---------------------------------------------------------------- warm lane targets

def lanes_root(cfg):
    return cfg["cache"] / "lanes"


def lock_path(cfg, lane):
    return cfg["cache"] / "locks" / f"{lane}.lock"


def valid_lane(lane):
    if not lane or len(lane) > 120 or not all(c.isalnum() or c in "-_." for c in lane) or lane.startswith("."):
        raise SystemExit(f"Invalid lane name {lane!r}")
    return lane


def release_locks(cfg, slot):
    locks = cfg["cache"] / "locks"
    if not locks.exists():
        return
    for lock in locks.glob("*.lock"):
        try:
            if json.loads(lock.read_text()).get("slot") == slot:
                lock.unlink()
        except (OSError, ValueError):
            pass


def git_blobs(checkout):
    output = subprocess.run(["git", "-C", str(checkout), "ls-files", "-s", "-z"], check=True,
                            capture_output=True).stdout.decode("utf-8", "surrogateescape")
    blobs = {}
    for entry in output.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, blob, _stage = meta.split(" ")
        blobs[path] = f"{mode}:{blob}"
    return blobs


def acquire(cfg, slot, lane, checkouts):
    """Give this job the lane's warm target, or a cold one when another job holds it.

    Cargo decides freshness from file times. A checkout gives every file the
    current time, so the lane records each tracked file's blob id and time after
    a successful job, and this gives unchanged files their recorded time back.
    Changed files keep the checkout time and rebuild what they feed. After a
    failed or cancelled job the lane is marked dirty and no time is restored.
    """
    valid_lane(lane)
    lane_dir = lanes_root(cfg) / lane
    lock = lock_path(cfg, lane)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        target = slot_dirs(cfg, slot)["cold"]
        target.mkdir(parents=True, exist_ok=True)
        log(f"lane {lane} is in use by another job; building cold in {target}")
        return target, "cold"
    with os.fdopen(fd, "w") as handle:
        json.dump({"slot": slot, "time": time.time(), "run": os.environ.get("GITHUB_RUN_ID")}, handle)
    target = lane_dir / "target"
    target.mkdir(parents=True, exist_ok=True)
    (lane_dir / "last-used").write_text(str(time.time()))
    manifest = lane_dir / "sources.json"
    dirty = lane_dir / "dirty"
    restored = changed = 0
    state = "cold"
    if manifest.exists() and not dirty.exists():
        recorded = json.loads(manifest.read_text())
        for name, checkout in checkouts.items():
            files = recorded.get(name, {})
            for path, blob in git_blobs(checkout).items():
                entry = files.get(path)
                if entry and entry[0] == blob:
                    try:
                        os.utime(Path(checkout) / path, (entry[1], entry[1]))
                        restored += 1
                    except OSError:
                        changed += 1
                else:
                    changed += 1
        state = "warm"
    elif dirty.exists():
        state = "dirty"
    dirty.write_text(str(time.time()))
    size = tree_size(target) / GiB
    log(f"lane {lane}: {state} target {target} ({size:.1f} GiB); "
        f"{restored} unchanged files keep their recorded time, {changed} are new or changed")
    return target, state


def commit(cfg, slot, lane, checkouts):
    valid_lane(lane)
    lane_dir = lanes_root(cfg) / lane
    lock = lock_path(cfg, lane)
    try:
        owner = json.loads(lock.read_text()).get("slot")
    except (OSError, ValueError):
        owner = None
    if owner != slot:
        log(f"lane {lane}: this job built cold; nothing to record")
        return
    recorded = {}
    for name, checkout in checkouts.items():
        files = {}
        # A tracked file the job rewrote no longer matches its index blob; leave
        # it out so its time is never restored onto different content.
        modified = set(subprocess.run(["git", "-C", str(checkout), "ls-files", "-m", "-z"], check=True,
                                      capture_output=True).stdout.decode("utf-8", "surrogateescape").split("\0"))
        for path, blob in git_blobs(checkout).items():
            if path in modified:
                continue
            try:
                files[path] = [blob, os.lstat(Path(checkout) / path).st_mtime]
            except OSError:
                pass
        recorded[name] = files
    temporary = lane_dir / "sources.json.tmp"
    temporary.write_text(json.dumps(recorded))
    os.replace(temporary, lane_dir / "sources.json")
    (lane_dir / "dirty").unlink(missing_ok=True)
    log(f"lane {lane}: recorded {sum(len(f) for f in recorded.values())} source times")


def lanes(cfg):
    root = lanes_root(cfg)
    if not root.exists():
        return []
    result = []
    for lane_dir in root.iterdir():
        try:
            used = float((lane_dir / "last-used").read_text())
        except (OSError, ValueError):
            used = 0.0
        result.append((used, lane_dir))
    return sorted(result)


def locked(cfg, lane_name):
    return lock_path(cfg, lane_name).exists()


def enforce_lane_caps(cfg):
    for _used, lane_dir in lanes(cfg):
        if locked(cfg, lane_dir.name):
            continue
        size = tree_size(lane_dir)
        if size > cfg["lane_cap_gib"] * GiB:
            log(f"lane {lane_dir.name} is {size / GiB:.0f} GiB, over the {cfg['lane_cap_gib']} GiB lane cap; removing it")
            remove_tree(lane_dir)


def cache_used(cfg):
    usage = shutil.disk_usage(cfg["cache"])
    return usage.total - usage.free


def evict(cfg):
    """Remove whole lanes, least recently used first, until under the cap and above the floor."""
    def satisfied():
        return (cache_used(cfg) <= cfg["cache_cap_gib"] * GiB
                and free_bytes(cfg["cache"]) >= cfg["cache_floor_gib"] * GiB)
    for _used, lane_dir in lanes(cfg):
        if satisfied():
            return
        if locked(cfg, lane_dir.name):
            continue
        log(f"evicting lane {lane_dir.name}")
        remove_tree(lane_dir)
    if not satisfied():
        log("cache is still over its cap after evicting every idle lane")


# ---------------------------------------------------------------- daily prune and idle shutdown

def prune(cfg):
    now = time.time()
    for used, lane_dir in lanes(cfg):
        if now - used > cfg["lane_max_idle_days"] * 86400 and not locked(cfg, lane_dir.name):
            log(f"pruning lane {lane_dir.name}: unused for {(now - used) / 86400:.0f} days")
            remove_tree(lane_dir)
    # Stale locks: a lock older than the longest job timeout belongs to no job.
    for lock in (cfg["cache"] / "locks").glob("*.lock") if (cfg["cache"] / "locks").exists() else []:
        if now - lock.stat().st_mtime > 12 * 3600:
            log(f"removing stale lock {lock.name}")
            lock.unlink(missing_ok=True)
    # Downloaded crate archives and npm/pip caches unused for 30 days.
    for sub in ("cargo/registry/cache", "npm/_cacache", "pip", "vcpkg-binary-cache"):
        _prune_old_files(cfg["cache"] / sub, now - 30 * 86400)
    # vcpkg keeps each port's build tree after installing it.
    for sub in ("vcpkg-runmat/buildtrees", "vcpkg-runmat/packages"):
        _prune_old_files(cfg["cache"] / sub, now - 3 * 86400)
    for diag in Path(cfg["runner_root"]).glob("*/_diag"):
        _prune_old_files(diag, now - 7 * 86400)
    enforce_lane_caps(cfg)
    evict(cfg)
    if not WINDOWS and shutil.which("docker"):
        subprocess.run(["docker", "system", "prune", "--all", "--force", "--filter", "until=72h"], check=False)
    (cfg["state"] / "last-prune").write_text(str(now))
    report(cfg)


def _prune_old_files(root, cutoff):
    if not Path(root).exists():
        return
    for directory, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            try:
                st = os.lstat(path)
                if max(st.st_atime, st.st_mtime) < cutoff:
                    os.unlink(path)
            except OSError:
                pass


def prune_if_due(cfg):
    try:
        last = float((cfg["state"] / "last-prune").read_text())
    except (OSError, ValueError):
        last = 0.0
    if time.time() - last > 20 * 3600:
        prune(cfg)


def jobs_running():
    if WINDOWS:
        output = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Runner.Worker.exe", "/NH"],
                                capture_output=True, text=True).stdout
        return "Runner.Worker.exe" in output
    return subprocess.run(["pgrep", "-f", "Runner.Worker"], capture_output=True).returncode == 0


def boot_time():
    if WINDOWS:
        output = subprocess.run(["powershell", "-NoProfile", "-Command",
                                 "[int][double]::Parse((Get-Date (Get-CimInstance Win32_OperatingSystem).LastBootUpTime -UFormat %s))"],
                                capture_output=True, text=True).stdout.strip()
        return float(output or 0)
    return time.time() - float(Path("/proc/uptime").read_text().split()[0])


def idle_check(cfg):
    """Stop the VM after `idle_minutes` with no job. The autostart service starts it again."""
    if jobs_running():
        touch_activity(cfg)
        return
    try:
        last = float((cfg["state"] / "last-activity").read_text())
    except (OSError, ValueError):
        last = 0.0
    last = max(last, boot_time())
    idle = (time.time() - last) / 60
    if idle < cfg["idle_minutes"]:
        return
    log(f"no job for {idle:.0f} minutes; stopping the runners and shutting down")
    if WINDOWS:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-Service actions.runner.* | Stop-Service -Force"], check=False)
        if jobs_running():
            log("a job started while stopping; starting the runners again")
            subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Service actions.runner.* | Start-Service"], check=False)
            return
        subprocess.run(["shutdown", "/s", "/t", "0", "/f"], check=False)
    else:
        subprocess.run(["systemctl", "stop", "ci-runner@*"], check=False)
        if jobs_running():
            log("a job started while stopping; starting the runners again")
            subprocess.run(["systemctl", "start", "ci-runners.target"], check=False)
            return
        subprocess.run(["systemctl", "poweroff"], check=False)


def report(cfg):
    lines = [f"work disk {free_bytes(cfg['work']) / GiB:.0f} GiB free",
             f"cache disk {cache_used(cfg) / GiB:.0f} GiB used, {free_bytes(cfg['cache']) / GiB:.0f} GiB free, cap {cfg['cache_cap_gib']} GiB"]
    for used, lane_dir in lanes(cfg):
        lines.append(f"  lane {lane_dir.name}: {tree_size(lane_dir) / GiB:.1f} GiB, last used "
                     f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(used))} UTC")
    log("\n".join(lines))


def main(argv):
    cfg = config()
    slot = os.environ.get("CI_SLOT", "")
    command = argv[0] if argv else ""
    if command == "job-started":
        job_started(cfg, slot)
    elif command == "job-completed":
        job_completed(cfg, slot)
    elif command == "slot-start":
        slot_start(cfg, argv[1])
    elif command in ("acquire", "commit"):
        lane = argv[1]
        checkouts = {}
        links = []  # --link=PATH links PATH to the lane target; --link=PATH@SUB to a subdirectory of it
        for item in argv[2:]:
            if item.startswith("--link="):
                path, _, sub = item[len("--link="):].partition("@")
                links.append((Path(path), sub))
                continue
            name, _, path = item.partition("=")
            checkouts[name] = Path(path or name).resolve()
        if command == "acquire":
            target, state = acquire(cfg, slot, lane, checkouts)
            for link, sub in links:
                # Steps that use relative target/ paths see the warm target through a link.
                # Workspace cleanup removes the link, never what it points to.
                destination = target / sub if sub else target
                destination.mkdir(parents=True, exist_ok=True)
                if os.path.lexists(link):
                    remove_tree(link)
                link.parent.mkdir(parents=True, exist_ok=True)
                if WINDOWS:
                    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(destination)], check=True,
                                   stdout=subprocess.DEVNULL)
                else:
                    os.symlink(destination, link, target_is_directory=True)
            env_file = os.environ.get("GITHUB_ENV")
            if env_file:
                # With --link, Cargo's default target/ already resolves to the warm target.
                with open(env_file, "a", encoding="utf-8") as handle:
                    if not links:
                        handle.write(f"CARGO_TARGET_DIR={target}\n")
                    handle.write(f"RUNMAT_CI_TARGET={target}\nRUNMAT_CI_TARGET_STATE={state}\n")
            output = os.environ.get("GITHUB_OUTPUT")
            if output:
                with open(output, "a", encoding="utf-8") as handle:
                    handle.write(f"target={target}\nstate={state}\n")
            print(target)
        else:
            commit(cfg, slot, lane, checkouts)
    elif command == "prune":
        prune(cfg)
    elif command == "prune-if-due":
        prune_if_due(cfg)
    elif command == "idle-check":
        idle_check(cfg)
    elif command == "report":
        report(cfg)
    else:
        raise SystemExit("usage: ci_cache.py job-started|job-completed|slot-start SLOT|"
                         "acquire LANE NAME=PATH...|commit LANE NAME=PATH...|prune|prune-if-due|idle-check|report")


if __name__ == "__main__":
    main(sys.argv[1:])
