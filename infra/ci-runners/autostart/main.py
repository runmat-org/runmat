"""Start a stopped RunMat CI runner VM when GitHub queues a job for its label.

GitHub sends `workflow_job` webhooks from runmat-org/runmat and
runmat-org/runmat-private. For a `queued` job whose labels name one of the
runner labels below, this service starts the matching VM. It holds no GitHub
credential: it only verifies the webhook signature and calls the Compute API
with its own service account, which may only get, start and resume the two
runner VMs.

A job can be queued while the VM is shutting down after its idle period. The
service therefore watches the VM for a few minutes after each queued job and
starts it again if it stops.
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
    if status in ("TERMINATED", "STOPPED"):
        compute("POST", instance, "/start")
        log(message="starting runner VM", instance=instance, was=status)
    elif status == "SUSPENDED":
        compute("POST", instance, "/resume")
        log(message="resuming runner VM", instance=instance, was=status)
    return status


def watch(instance):
    """Start the VM now, and again if it stops during the watch window."""
    deadline = time.time() + WATCH_SECONDS
    while time.time() < deadline:
        try:
            ensure_running(instance)
        except urllib.error.HTTPError as error:
            # 400 while the VM is still stopping; a quota error is logged and retried.
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
