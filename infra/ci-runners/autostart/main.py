"""Start a stopped RunMat CI runner VM when GitHub queues a job for its label.

GitHub sends `workflow_job` webhooks from runmat-org/runmat and
runmat-org/runmat-private. For a `queued` job whose labels name one of the
runner labels below, this service starts the matching VM. It holds no GitHub
credential: it only verifies the webhook signature and calls the Compute API
with its own service account, which may only get, start and resume the two
runner VMs.

A start can fail: the VM may still be stopping after its idle period, or the
region's CPU quota may be in use by other machines. The service therefore keeps
trying until the VM is running, for up to WATCH_MAX_SECONDS after the last
queued job, and then watches it for WATCH_SECONDS more and starts it again if
it stops.
"""

import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT = os.environ.get("CI_PROJECT", "runmat")
ZONE = os.environ.get("CI_ZONE", "us-central1-b")
SECRET = os.environ["WEBHOOK_SECRET"].strip().encode()
REPOSITORIES = {"runmat-org/runmat", "runmat-org/runmat-private"}
INSTANCES = json.loads(os.environ.get("CI_LABEL_INSTANCES",
                                      '{"runmat-gcp-linux-x64": "ci-linux-x64", "runmat-gcp-windows-x64": "ci-windows-x64"}'))
WATCH_SECONDS = 8 * 60
# Queued jobs wait up to 24 h on GitHub; quota held by other machines can take
# hours to free.
WATCH_MAX_SECONDS = int(os.environ.get("CI_WATCH_MAX_SECONDS", str(6 * 60 * 60)))
watching = {}
lock = threading.Lock()


def log(**fields):
    print(json.dumps(fields), flush=True)


def token():
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)["access_token"]


def compute(method, instance, verb=""):
    url = f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/instances/{instance}{verb}"
    request = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None,
                                     headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def ensure_running(instance):
    status = compute("GET", instance)["status"]
    verb = {"TERMINATED": "/start", "STOPPED": "/start", "SUSPENDED": "/resume"}.get(status)
    if verb:
        # The start is asynchronous. A start that fails (for example QUOTA_EXCEEDED
        # while other machines hold the region's CPUs) leaves the VM stopped, and
        # the watcher's next pass tries again.
        compute("POST", instance, verb)
        log(message="starting runner VM", instance=instance, was=status, verb=verb)
    return status


def watch(instance):
    """Start the VM, retrying until it runs, then restart it if it stops during WATCH_SECONDS.

    watching[instance] holds the time of the last queued job; each new queued job
    extends the retry window.
    """
    running_since = None
    while True:
        with lock:
            last_queued = watching[instance]
        now = time.time()
        if running_since is not None and now - running_since >= WATCH_SECONDS and now - last_queued >= WATCH_SECONDS:
            break
        if running_since is None and now - last_queued >= WATCH_MAX_SECONDS:
            log(message="gave up starting runner VM", instance=instance, hours=WATCH_MAX_SECONDS / 3600)
            break
        try:
            if ensure_running(instance) == "RUNNING":
                running_since = running_since or now
            else:
                running_since = None
        except urllib.error.HTTPError as error:
            # 400 while the VM is still stopping.
            log(message="start not possible yet", instance=instance, code=error.code,
                detail=error.read().decode(errors="replace")[:500])
        except Exception as error:  # noqa: BLE001 - keep the watcher alive
            log(message="start failed", instance=instance, error=repr(error))
        time.sleep(20)
    with lock:
        watching.pop(instance, None)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, "ok")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 5 * 1024 * 1024:
            return self.reply(413, "too large")
        body = self.rfile.read(length)
        expected = "sha256=" + hmac.new(SECRET, body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, self.headers.get("X-Hub-Signature-256", "")):
            log(message="rejected unsigned or mis-signed delivery")
            return self.reply(401, "bad signature")
        event = self.headers.get("X-GitHub-Event", "")
        if event == "ping":
            return self.reply(200, "pong")
        if event != "workflow_job":
            return self.reply(202, "ignored")
        payload = json.loads(body)
        repository = payload.get("repository", {}).get("full_name")
        job = payload.get("workflow_job", {})
        if payload.get("action") != "queued" or repository not in REPOSITORIES:
            return self.reply(202, "ignored")
        targets = {INSTANCES[label] for label in job.get("labels", []) if label in INSTANCES}
        for instance in targets:
            log(message="queued job", repository=repository, job=job.get("name"), run=job.get("run_id"), instance=instance)
            with lock:
                if instance in watching:
                    watching[instance] = time.time()
                    continue
                watching[instance] = time.time()
            threading.Thread(target=watch, args=(instance,), daemon=True).start()
        return self.reply(202, "starting " + ",".join(sorted(targets)) if targets else "no runner label")

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("", int(os.environ.get("PORT", "8080"))), Handler).serve_forever()
