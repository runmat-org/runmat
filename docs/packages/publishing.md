---
title: "Publishing Packages"
category: "Packages"
section: "8.3"
last_updated: "August 1, 2026"
---

# Publishing Packages

Publishing is a deterministic build-and-verify workflow. The same manifest and included bytes produce the same canonical inventory, artifact digest, tree digest, release metadata digest, and signing input across supported hosts.

## Manifest

A publishable package declares an organization, semantic version, source roots, and publication policy:

```toml
[package]
name = "optimization"
organization = "acme"
version = "2.1.0"
runmat-version = ">=0.6.1"

[sources]
roots = ["src"]

[publish]
license = "MIT"
readme = "README.md"
include = ["src/**", "README.md", "LICENSE"]
exclude = ["**/*.tmp", "private-notes/**"]
```

The package canonical ID is derived from registry, organization, and name. Include/exclude rules are normalized and applied deterministically. The release builder rejects parent/absolute traversal, escaping links, device files, collisions after portable normalization, unsupported names, credential-bearing metadata, unapproved native content, file-count/size expansion violations, and content that changes during assembly.

## Inspect Before Upload

Use `inspect` locally and in CI:

```bash
runmat package inspect
runmat package inspect --json
runmat package inspect --allow-native
```

`--allow-native` permits entries selected by native-content policy; it does not suppress target/capability declarations or Server review. The JSON output is suitable for comparing inventory and digests between builders.

## Publish

Create the package in Desktop or the organization administration API, then publish with the returned IDs:

```bash
runmat package publish \
  --org-id org_0123456789abcdef0123456789abcdef \
  --package-id pkg_0123456789abcdef0123456789abcdef
```

The CLI builds the artifact, creates a publication, uploads through the Server-issued transfer contract, completes upload, requests verification, requests approval, and finalizes. It retries only operations whose idempotency contract makes retry safe. Transfer URLs and authorization grants are ephemeral and never enter the lock or artifact.

For a private package, first register recipient keys for principals that need to decrypt it, then add `--private`:

```bash
runmat package keys register acme optimization
runmat package publish \
  --org-id org_0123456789abcdef0123456789abcdef \
  --package-id pkg_0123456789abcdef0123456789abcdef \
  --private \
  --key-version 1
```

The client encrypts the deterministic artifact with a fresh content key and wraps that key to every active recipient. The Server validates encrypted metadata and envelopes but cannot decrypt package content.

## Provenance, License, SBOM, And Advisories

Publication metadata binds the artifact/tree digests, canonical package/version, dependency declarations, capabilities, license/readme digests, encryption metadata, and optional supply-chain statement into one release digest. Signature verification uses that digest rather than an ambient file path or transfer URL.

Administrators may require an approved builder identity, source repository/commit, workflow, reproducible-build statement, signature algorithm/key, license declaration, and SBOM. Missing or mismatched required evidence blocks verification or finalization. Advisories are managed independently after release and can target exact or ranged versions.
