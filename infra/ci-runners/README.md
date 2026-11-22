# GCP runners for GitHub Actions

RunMat's Linux and Windows CI jobs run on two self-hosted VMs in the `runmat`
GCP project. macOS jobs stay on WarpBuild. Each VM starts when GitHub queues a
job for its label, keeps a warm build cache on its own disk, and stops itself
after 30 idle minutes.

| Label | VM | Machine | Runner slots | Disks |
|---|---|---|---|---|
| `runmat-gcp-linux-x64` | `ci-linux-x64` (Ubuntu 24.04) | n2-highmem-32: 32 vCPU, 256 GB | 5 for runmat, 2 for runmat-private | boot 100 GB, work 500 GB pd-ssd (`/ci/work`), cache 1500 GB pd-ssd (`/ci/cache`) |
| `runmat-gcp-windows-x64` | `ci-windows-x64` (Windows Server 2022, licence included) | n2-highmem-64: 64 vCPU, 512 GB | 8 for runmat, 1 for runmat-private | boot 200 GB, work 600 GB pd-ssd (`D:`), cache 2000 GB pd-ssd (`E:`) |

A slot is one runner registration, so a VM runs up to that many jobs at once.
The earlier `-8x`, `-4x` and `-16x` Warp labels all map to these two labels.

## Who may use these runners

The runners run only code from runmat-org repositories:

- Workflows use a `runmat-gcp-*` label only for pushes, `workflow_dispatch`,
  `workflow_call`, schedules, releases and pull requests whose head branch is
  in the same repository. In `.github/workflows/ci.yml`, a pull request from a
  fork selects GitHub-hosted `ubuntu-latest` or `windows-latest` instead.
- A fork's pull request runs the workflow file from the fork, so the fork could
  name a `runmat-gcp-*` label. The job-started hook on the VM refuses any
  `pull_request` job whose head repository is not the base repository, and any
  event outside push, `workflow_dispatch`, schedule, release, merge group and
  repository dispatch (for example `pull_request_target` and `workflow_run`).
  The hook lives on the VM, so a workflow cannot change it.
- runmat-org/runmat requires approval before workflows run for any outside
  contributor's pull request. runmat-org/runmat-private cannot be forked.
- The runners are registered per repository (the organization is on GitHub
  Free, which has no restricted runner groups).

## How slots share a VM

- CPU: on Linux each slot's unit has `CPUWeight=100`, so busy slots get equal
  CPU shares however many threads their jobs start. An idle VM gives one job
  every core.
- JavaScript tests: Vitest otherwise starts one worker per core less one (31 on
  32 vCPU), and its main process, which transforms every module, cannot keep up;
  tests then time out waiting for imports. On Linux each slot's `.env` sets
  `VITEST_MAX_THREADS`, `VITEST_MIN_THREADS`, `VITEST_MAX_FORKS` and
  `VITEST_MIN_FORKS` to `2 × vCPUs / slots` (9 on `ci-linux-x64`), a share that
  assumes half the slots run CPU-bound tests at once. Vitest ignores a maximum
  below its default minimum, so both are set.
- Checkout path: every Linux slot sees its own work folder at `/ci/w` (a bind
  mount in the slot's mount namespace; the runners are registered with
  `--work /ci/w`). Jobs on different slots therefore build from the same
  absolute paths, and a lane's warm Cargo target stays fresh whichever slot ran
  last. Cargo fingerprints a package whose build script prints no
  `rerun-if-changed` by its newest file's absolute path, so different paths
  rebuilt those crates and everything above them. On the host, `/ci/w` is an
  empty read-only mount point; the hooks and timers use `/ci/work/<slot>/w`.
  A step that hands a workspace path to the Docker daemon (`docker run -v`)
  would name the host's `/ci/w`; no workflow does this today.

## What a job may leave behind

Every slot has its own work folder, `HOME` and temporary folder on the work
disk. The job-started hook empties them and the job-completed hook wipes them,
so nothing a job builds stays on the work disk. Artifacts go to GitHub
(`actions/upload-artifact`) or to GCS through the workflows' own identities.

The cache disk holds only these directories, each with a bound:

| Directory | Contents | Bound |
|---|---|---|
| `lanes/<lane>/target` | Warm Cargo target for one workflow lane, such as `runmat-linux-tests-runtime` | A lane over `lane_cap_gib` (160 GB Linux, 200 GB Windows) is deleted after its job. Lanes unused for 14 days are deleted. |
| `cargo`, `rustup/<slot>` | Cargo registry and rustup toolchains (one rustup home per slot, because `rustup default` is global) | Crate archives unused for 30 days are deleted. |
| `vcpkg-runmat`, `vcpkg-binary-cache` | vcpkg with OpenCASCADE and other native libraries | Shared; one job provisions it at a time. |
| `toolcache`, `npm`, `pip` | setup-python/setup-node downloads and package caches | Files unused for 30 days are deleted. |
| `docker` (Linux) | Docker images and build cache | `docker system prune` for anything older than 72 hours. |

When the cache disk passes `cache_cap_gib` (80% of the disk) or falls below
its free-space floor (12%), whole lanes are evicted, least recently used first.
The job-started hook refuses the job when the work disk has less than 100 GB
(Linux) or 120 GB (Windows) free, the cache disk is below its floor after
eviction, or the system disk has less than 15 GB (Linux) or 25 GB (Windows)
free. The prune runs every 24 hours while the VM is up and soon after each
boot when the last prune is older than 20 hours.

A Cloud Monitoring alert (`CI runner disk above 85%`) emails the owner when any
disk on either VM passes 85% for 10 minutes.

### Warm Cargo targets

A workflow step calls `acquire` before building and `commit` at the end:

```bash
python3 "$RUNMAT_CI_TOOL" acquire runmat-linux-tests-runtime "runmat=$GITHUB_WORKSPACE"
# ... build and test ...
python3 "$RUNMAT_CI_TOOL" commit runmat-linux-tests-runtime "runmat=$GITHUB_WORKSPACE"
```

`acquire` locks the lane, points `CARGO_TARGET_DIR` at it (or links
`target/` there with `--link=PATH[@SUBDIR]`), and gives every tracked file
whose Git blob is unchanged since the lane's last `commit` its recorded
modification time. Cargo decides freshness from file times, so only crates
whose sources changed rebuild. `commit` records the blob and time of every
tracked file the job did not modify. A job that ends without `commit` leaves
the lane marked dirty; the next job restores no times and Cargo rebuilds every
workspace crate. When a second job wants a lane that is in use, it builds in a
cold target on the work disk, which the job-completed hook deletes.

Runner names start with `gcp-` (`gcp-ci-linux-x64-pub1`), and workflows test
`startsWith(runner.name, 'gcp-')`; WarpBuild's macOS runners also report
`runner.environment == 'self-hosted'`, so that test cannot tell them apart.
Workflows skip `Swatinem/rust-cache` and the target and vcpkg `actions/cache`
steps on these runners (`if: ${{ !startsWith(runner.name, 'gcp-') }}`): the warm
target replaces them, and rust-cache's cleanup would delete registry sources
that concurrent jobs share. Small caches that live in the job's temporary
folder (npm downloads) still use `actions/cache`.

## Autostart and idle shutdown

```
GitHub workflow_job "queued" webhook (runmat, runmat-private)
        |
        v
Cloud Run ci-runner-autostart (verifies the HMAC signature)
        |
        v
Compute API: start ci-linux-x64 or ci-windows-x64 for the job's label
```

The service holds no GitHub credential. Its service account can only get,
start and resume the two VMs (`roles/ciRunnerStarter` on each instance) and
read the webhook secret. After a queued job it watches the VM for eight
minutes and starts it again if the VM was shutting down when the job arrived.

On each VM, `idle-check` runs every 2 minutes. When no `Runner.Worker` process
has run for 30 minutes (counted from the last job or the boot), it stops the
runner services and powers the VM off. If a job starts while the services stop,
it starts them again and stays up.

## Files

| Path | Purpose |
|---|---|
| `common/ci_cache.py` | Job hooks, warm lanes, eviction, daily prune and idle shutdown (Linux and Windows) |
| `linux/setup.sh`, `linux/startup.sh`, `linux/hooks/`, `linux/systemd/` | Linux provisioning, runner slots as `ci-runner@<slot>` units, timers |
| `windows/setup.ps1`, `windows/startup.ps1`, `windows/hooks/` | Windows provisioning, runner services, scheduled tasks |
| `autostart/` | Cloud Run service that starts the VMs |
| `gcp/push-bundle.sh` | Publishes these files to a VM and reruns its setup |
| `gcp/create.sh` | The commands that created the GCP resources |

## Operating the runners

- Change a script: edit it here, then run `gcp/push-bundle.sh linux` or
  `gcp/push-bundle.sh windows`. Linux reruns setup over IAP SSH; Windows resets
  the VM and reruns setup at boot.
- Add a slot or re-register: add it to the VM's `ci-slots` metadata, then run
  `gcp/push-bundle.sh <os> --register`. The script puts one-hour registration
  tokens in the VM's metadata for that setup run and removes them afterwards;
  tokens never enter the repository or a log.
- Look at a VM: `gcloud compute ssh ci-linux-x64 --tunnel-through-iap` or RDP
  through IAP to `ci-windows-x64`. Firewall rules allow only IAP; the VMs take
  no other inbound traffic.
- Cache report: `sudo python3 /opt/ci/common/ci_cache.py report` (Linux) or
  `C:\Python313\python.exe C:\ci\common\ci_cache.py report` (Windows).
- Do not borrow these VMs for other work: every job wipes the work disk.
