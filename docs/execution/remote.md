---
title: "Remote Execution"
category: "Execution"
section: "12.3"
last_updated: "August 4, 2026"
---

# Remote Execution

RunMat uses the same frozen package graph, portable execution artifacts, value codec, and driver scheduler for local processes, browser workers, customer-managed nodes, and hosted nodes. The Server admits work, assigns coarse resources, relays opaque frames when a direct connection is unavailable, records coarse lifecycle and usage, and stores encrypted artifacts. It does not execute MATLAB code or receive workload plaintext.

## Before You Start

Sign in with `runmat auth login` and select a Server project in `runmat.toml` or with the command's `--project` option. Resolve and lock the project before submitting production work:

```bash
runmat package resolve
runmat package fetch --locked
```

Remote execution freezes the selected source closure and dependency identities. Workers do not resolve Git branches, fetch packages, or receive registry credentials.

## Create a Cluster and Enroll a Customer Node

Organization administrators create a cluster and mint a short-lived, single-use enrollment:

```bash
runmat cluster create --name workstation-pool --queue default
runmat cluster enroll CLUSTER_ID --ttl-seconds 900
```

Transfer the enrollment secret through the organization's secret-delivery channel. On the intended host, run the node agent's enrollment flow shown by `runmat-node-agent --help`. Do not place the secret in a service file, shell profile, image, or log.

Inspect the native service plan before installing it:

```bash
runmat-node-agent --server https://api.runmat.com --runmat /usr/local/bin/runmat service install --dry-run
sudo runmat-node-agent --server https://api.runmat.com --runmat /usr/local/bin/runmat service install
runmat-node-agent service inspect
```

The agent uses systemd on Linux, launchd on macOS, and the Windows Service Control Manager on Windows. Installation persists non-secret startup configuration only. The enrolled node credential remains in the agent's private state directory and survives service removal so a service can be repaired without creating a second node identity.

Use `runmat cluster nodes CLUSTER_ID` to wait for the node to become active. Hosted-node clusters use the same submission and scheduling contract; RunMat operates enrollment and service lifecycle for those nodes.

## Submit Jobs and Tests

Pin the endpoint identity shown for the admitted node. RunMat confirms the signed endpoint evidence before encrypting the package, program, inputs, results, detailed events, or diagnostics:

```bash
runmat job submit analysis.m \
  --cluster CLUSTER_ID \
  --trust-identity SHA256_FINGERPRINT \
  --detach

runmat job list
runmat job show RUN_ID
runmat job attach RUN_ID
```

Request cancellation with `runmat job cancel RUN_ID`. Cancellation is fenced and durable, but arbitrary MATLAB effects outside the managed result boundary cannot be rolled back. A lost driver or node can therefore produce an `indeterminate` run rather than a misleading retry or success.

Distributed tests preserve test-owned discovery, fixture grouping, retry, event, result, report, and coverage semantics while scheduling attempts through the same execution driver:

```bash
runmat test \
  --cluster CLUSTER_ID \
  --project PROJECT_ID \
  --trust-identity SHA256_FINGERPRINT \
  --max-workers 8
```

## Browser and Web Execution

The web runtime uses the same Rust-owned scheduler, artifact formats, encryption contexts, endpoint verification, and recovery-recipient validation compiled to WebAssembly. Dedicated Web Workers provide the browser's strongest available isolation. The browser host supplies worker and network mechanisms; it does not implement a second key schedule or scheduler in JavaScript.

Browser submissions fetch the project execution policy before encryption. When organization recovery is enabled, the Rust/WASM boundary seals the same per-run content key to both the admitted execution endpoint and the organization recovery public key. Private recovery material never enters the browser application or RunMat Server.

## Organization Recovery

Recovery is optional and organization-administered. Generate the private key on a trusted custodian host:

```bash
runmat job recovery keygen \
  --output runmat-recovery-2026.json \
  --valid-days 365 \
  --custodian-uri kms://example/runmat-recovery
```

`keygen` refuses to overwrite an existing file and creates a private `0600` file on Unix. Back up that file through the organization's key-custody process before configuring its public recipient:

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

## Drain and Retire a Node

Drain before maintenance so active leases can finish while new work is placed elsewhere:

```bash
runmat cluster node-state CLUSTER_ID NODE_ID draining
runmat cluster nodes CLUSTER_ID
```

Wait until the node has no active allocations, stop or uninstall the service, and perform maintenance. Reactivate it only after the agent reports a healthy inventory. To permanently retire a node, revoke it through the organization administration surface before deleting its private state. Removing local state first can leave a valid orphaned credential.

## Automation and Output

Every cluster command and durable job observation or mutation supports `--json`. JSON output is a single stable API object or page and never includes ANSI escapes. `NO_COLOR` and `--color=never` disable styling for human output. Avoid logging enrollment secrets, attach secrets, recovery key files, signed artifact URLs, or relay tickets.

Use `[sources].roots` for stable project source roots so static analysis, navigation, output arity, and type/shape analysis see the same intended project closure. Runtime `addpath` remains supported for session-dynamic resolution, but a dynamically computed path cannot generally become a statically proven compilation target.

## Failure Triage

- `endpoint identity changed`: inspect the admitted node and deliberately trust the new signed fingerprint; never silently accept it.
- `recovery recipient mismatch`: refresh policy and resubmit so the envelope uses the currently active public key.
- `indeterminate`: inspect coarse run/node events, then use organization recovery for the encrypted diagnostic if configured.
- relay unavailable: keep the client attached so authenticated sequence resumption can recover after the relay backend returns; frames are not persisted by the relay.
- node loss: drain or revoke the node and verify its process tree is gone before reactivation.

Operators should use the private execution operability runbook for relay, retention, billing, and coarse control-plane diagnosis. Workload plaintext must never be requested for routine operations.
