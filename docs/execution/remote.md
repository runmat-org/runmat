# Remote Execution

Run a script on another machine by adding `--cluster` to the command you use locally. Create the cluster once, join the machines that will do the work, then submit from your computer. RunMat Server coordinates the runs; the machines in the cluster execute your code.

## 1. Create a Cluster

Install [the RunMat CLI](/docs/runtime/getting-started/cli) on your computer and each execution machine. On your computer, sign in with an organization administrator account and create a cluster named `engineering`:

```bash
runmat login
runmat cluster create --name engineering
```

For an existing cluster, use its name in the following commands.

## 2. Join an Execution Machine

On the machine that will run your code, sign in with an account authorized to register machines, then join:

```bash
runmat login
runmat cluster join engineering
```

Leave this terminal running while the machine accepts work. Press Ctrl-C to stop it. Repeat these commands on each machine you want to add. A machine with an authorized `RUNMAT_API_KEY` can join directly.

Back on your computer, check that a node is active:

```bash
runmat cluster nodes engineering
```

To start the node as a background service at boot, see [Keep a Node Running as a Service](#keep-a-node-running-as-a-service).

## 3. Run Your Script

On your computer, open the local project containing `analysis.m` and `runmat.toml`. Sign in if needed. If you have not selected the Server project that will own the run, select it once:

```bash
runmat project select PROJECT_ID
```

Run your script:

```bash
runmat analysis.m --cluster engineering
```

RunMat waits for completion. `runmat run analysis.m` and the filename shorthand use the same execution path. Omit `--cluster` to run locally.

When the run finishes, the CLI displays captured `disp` and console `fprintf` output. `job attach` also displays this output. Capture has finite per-source and aggregate limits; interrupted or limit-reached captures include a notice with the retained output. Output from separate worker attempts is labeled by task and attempt, with ordering preserved within each source. Console output is returned at completion, rather than streamed live. Script workspace variables remain on the execution machine.

RunMat verifies the signed execution identity selected by the authenticated Server and encrypts your workload automatically. For independent identity verification, see [Encryption and Independent Identity Verification](#encryption-and-independent-identity-verification).

## Follow or Cancel a Run

Add `--detach` to return after submission:

```bash
runmat analysis.m --cluster engineering --detach
runmat job list
runmat job show RUN_ID
runmat job attach RUN_ID
runmat job cancel RUN_ID
```

Use the run ID returned by submission. Cancellation is durable. External effects, such as file writes, may persist after cancellation. If a lost driver or node leaves the run `indeterminate`, inspect its logs and external effects before retrying.

## Run Tests on the Cluster

Run the project test suite remotely with the same cluster name:

```bash
runmat test --cluster engineering --max-workers 8
```

Tests retain their fixture grouping, retry, reporting and coverage behavior while their attempts run on cluster workers.

## Advanced Usage

### Organizations and Cluster Names

Names are unique within an organization and canonicalized to lowercase. They contain 1–63 ASCII letters, digits, hyphens or underscores, starting with a letter or digit. Commands resolve the exact name within the selected organization to a stable ID.

`--org` is optional when the CLI has a selected organization or the credentials authorize exactly one organization. With several available organizations, interactive commands ask which to use; unattended and JSON-output commands require an explicit selection. Use an account or API key with permission to register nodes.

### Project Dependencies and Execution Options

For production work, resolve and lock dependencies before submission:

```bash
runmat package resolve
runmat package fetch --locked
```

Submission freezes the selected package graph and dependency identities. Ordinary jobs send compiled artifacts; workers do not resolve Git branches, fetch packages or receive registry credentials. Remote tests also carry the verified source closure needed for fixtures and discovery.

Use `--project PROJECT_ID` to override the selected Server project, `--queue` to select a queue other than `default`, and `--workers` to request a worker count.

Script entrypoints accept no positional inputs. To invoke a source file's declared function with string inputs, select it explicitly:

```bash
runmat functions/evaluate_case.m \
  --function evaluate_case \
  --cluster engineering \
  -- case-a input/data.mat
```

Use `[sources].roots` for stable project source roots. Runtime `addpath` supports dynamic session paths, but a computed path is not generally a statically proven compilation target.

### Keep a Node Running as a Service

For a machine that should accept work after startup without an open terminal, join with `--service`:

```bash
runmat cluster join engineering --service
```

This installs and starts systemd on Linux, launchd on macOS or a Windows service. It requires administrative privileges. Registration and installation must use the same OS identity and private state directory, with that identity's account credentials and selected organization. The foreground default uses your user's state directory. Register the system service under its service identity and state directory. Linux system services use `/var/lib/runmat/node-agent`; macOS uses `/Library/Application Support/RunMat/node-agent`.

The account authorizes registration; the running node uses its own machine credential. Retrying an interrupted registration reuses its original signing identity. A registered directory belongs to one Server, organization and cluster. Removing a service preserves its credential for reinstallation. Revoke the node before deleting its state.

For an explicitly configured Linux installation, inspect the service plan before applying it:

```bash
sudo /usr/local/bin/runmat node --server https://api.runmat.com --state-directory /var/lib/runmat/node-agent --runmat /usr/local/bin/runmat service install --dry-run
sudo /usr/local/bin/runmat node --server https://api.runmat.com --state-directory /var/lib/runmat/node-agent --runmat /usr/local/bin/runmat service install
```

Use `runmat node run` for diagnosis with the original server, executable and state-directory options. Stop the service before starting a foreground node. `runmat node status` prints the saved local identity; `runmat cluster nodes engineering` shows reported cluster state. `runmat node service print` prints the installation plan. Inspect an installed service with `sudo systemctl status runmat-node-agent` on Linux or `sudo launchctl print system/com.runmat.node-agent` on macOS.

### Token-Based Provisioning

For automated provisioning, an administrator can issue a short-lived token with `runmat cluster enroll engineering`; the machine reads it through `runmat node enroll --token-stdin`. Transfer the secret through your organization's secret channel. Keep it out of service files, images, command arguments and logs. RunMat operates registration and service lifecycle for hosted nodes.

### Native and Accelerator Requirements

Compiled programs and extensions carry their host requirements. Native target, runtime/catalog identity, adapter ABI, isolation, trust and accelerator requirements are checked before execution. A CUDA MEX module, for example, requires a compatible CUDA provider and a device with the required memory and features.

The scheduler leases specific devices to an attempt. Provider selection is restricted to those leases; the provider selects kernels. A different provider version or ABI does not satisfy a requirement. CUDA and WGPU views of one physical GPU share an allocation identity so it cannot receive two exclusive leases at once.

Remote GPU inventory requires a positive declared total device-memory capacity, separate from the maximum buffer size. It excludes providers with unknown capacity; those remain available for local work. To pass device-resident values to another worker, use a supported materialization or transfer operation. Browser workers reject native-code and unsupported accelerator artifacts before execution.

### Encryption and Independent Identity Verification

Encryption is automatic for remote execution. The submitting client verifies signed evidence binding the execution identity to the run, organization, cluster, allocation and fencing token before encrypting the workload. The execution node receives the decryption key; Server stores and relays ciphertext. Signed evidence, expiry and allocation checks apply whether or not you specify an independent pin.

By default, you trust the authenticated Server to identify the registered execution node. This avoids a manual key exchange, but a compromised control plane could substitute a node it controls. Independent pinning adds a separate identity check against a fingerprint you obtained through a trusted channel.

To use it, get the expected execution identity fingerprint from your cluster administrator, verify it independently, then supply it:

```bash
runmat analysis.m --cluster engineering --trust-identity SHA256_FINGERPRINT
```

Replace the placeholder with the verified fingerprint. To reuse the pin, set `RUNMAT_TRUST_IDENTITY` in your shell. The explicit flag takes precedence over the environment. A fingerprint mismatch stops the run before workload encryption. Review an intended identity rotation before changing the pin.

RunMat encrypts every remote workload. Choose execution machines and administrators you trust with the code and data they run.

### Browser and Web Execution

The web runtime uses the Rust scheduler, artifact formats, encryption, endpoint verification and recovery-recipient validation compiled to WebAssembly. Dedicated Web Workers isolate browser execution. The JavaScript host creates workers and handles network requests.

Browser submissions fetch the project execution policy before encryption. When organization recovery is enabled, the Rust/WASM boundary seals the same per-run content key to both the admitted execution endpoint and the organization recovery public key. Private recovery material never enters the browser application or RunMat Server.

### Organization Recovery

Recovery is optional and organization-administered. Generate the private key on a trusted custodian host:

```bash
runmat recovery-key create \
  --output runmat-recovery-2026.json \
  --valid-days 365 \
  --custodian-uri kms://example/runmat-recovery
```

`create` refuses to overwrite an existing file and creates a private `0600` file on Unix. Windows uses a protected owner-and-SYSTEM DACL supplied atomically at file creation, never inherited directory permissions. Imported Windows keys granting access beyond owner and SYSTEM are rejected. Back up that file through the organization's key-custody process before configuring its public recipient:

```bash
runmat job recovery configure --org ORG_ID --key runmat-recovery-2026.json
runmat job recovery show --org ORG_ID
```

Once enabled, admission requires new submissions to carry an envelope for the exact active recovery fingerprint. Policy rotation and run creation are serialized so a run cannot be admitted against a stale recipient. Existing runs keep the fingerprint and envelope with which they were created.

An organization administrator can recover a terminal result or encrypted diagnostic locally:

```bash
runmat job recovery recover RUN_ID \
  --project PROJECT_ID \
  --key runmat-recovery-2026.json
```

The Server authorizes access and returns ciphertext plus the matching envelope. Decryption happens only in the CLI. The Server never receives the private key, unwrapped run key, result, or detailed diagnostic. Disable the requirement with `runmat job recovery disable --org ORG_ID`; keep old private keys for the retention period of runs encrypted to them.

Key creation requires no network or sign-in. `--json` emits only the path and public recipient, suitable for transferring to an online administrator. The server recipient is camelCase JSON: suite `x25519-hkdf-sha256-aes128gcm-v1`, unpadded base64url `publicKey` (32 raw X25519 bytes), lowercase hex `fingerprint` computed by `recipient_fingerprint` (SHA-256 of ASCII `runmat-execution-recipient-key-v1`, one zero byte, then the raw public key), millisecond validity bounds, and optional `custodianUri`. The private file also contains `schemaVersion: 1` and `privateKey` (unpadded base64url of the raw 32-byte X25519 secret). Never transfer the private file to configure a server; transfer only its `recipient` object. Existing `job recovery keygen` remains accepted.

For an offline recovery, export the authorized server recovery response and encrypted artifact on an online machine. Decode its `envelope` field from unpadded base64url to canonical binary CBOR, transfer that binary envelope and ciphertext to the custodian, and run:

```bash
runmat recovery-key decrypt --key runmat-recovery-2026.json \
  --run-id RUN_ID --envelope envelope.cbor --artifact result.cbor \
  --output recovered-result.json
```

Use `--diagnostic` for a diagnostic artifact. This authenticates the recipient, run identity, epoch (currently 1), purpose and ciphertext before writing any plaintext. Output is a new private file; existing files are not overwritten. Keep the output under the same custody policy as the original result. Desktop/native/browser hosts can call `runmat_execution_artifact::encryption::open_recovery_artifact` with a `PortableExecutionPrivateKey`, the canonical binary inputs and an explicit `RecoveryArtifactScope`; this uses the existing HPKE and run-object wire formats, not a second crypto implementation.

### Drain and Retire a Node

Drain before maintenance so active leases can finish while new work is placed elsewhere:

```bash
runmat cluster node-state engineering NODE_ID draining
runmat cluster nodes engineering
```

Wait until the node has no active allocations, stop or uninstall the service, and perform maintenance. Reactivate it only after the agent reports a healthy inventory. To permanently retire a node, revoke it through the organization administration surface before deleting its private state. Removing local state first can leave a valid orphaned credential.

### Automation and Output

Cluster administration and durable job observation or mutation support `--json`.
JSON output is a single API object or page without ANSI escapes. Join and the
node service run until stopped unless `--service` is selected. `NO_COLOR` and
`--color=never` disable styling for human output. Avoid logging enrollment
secrets, attach secrets, recovery key files, signed artifact URLs or relay tickets.

Credentials belong to the complete Server endpoint, including its port and base path. Signing in to one endpoint does not authorize another endpoint on the same host. Endpoints require HTTPS; local development may use HTTP on loopback.

## Failure Triage

- `endpoint identity changed` with a pin selected: inspect the admitted node and verify the intended rotation before replacing the pin.
- `recovery recipient mismatch`: refresh policy and resubmit so the envelope uses the currently active public key.
- `indeterminate`: inspect coarse run/node events, then use organization recovery for the encrypted diagnostic if configured.
- relay unavailable: keep the client attached so authenticated sequence resumption can recover after the relay backend returns; frames are not persisted by the relay.
- node loss: drain or revoke the node and verify its process tree is gone before reactivation.

Operators should use the private execution operability runbook for relay, retention, billing, and coarse control-plane diagnosis. Workload plaintext must never be requested for routine operations.
