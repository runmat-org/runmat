---
title: "Hosted Package Registry"
category: "Packages"
section: "8.2"
last_updated: "August 1, 2026"
---

# Hosted Package Registry

The RunMat registry stores immutable, versioned package releases. The Server is authoritative for package ownership, visibility, release lifecycle, signing/provenance policy, recipient grants, yanks, revocations, advisories, and transfer authorization. The website, Desktop, CLI, and browser clients consume the same public API and do not reproduce those decisions.

## Using Registry Packages

The default registry uses the active RunMat Server:

```toml
[dependencies]
optimization = { package = "acme/optimization", version = "^2.1" }
```

Named registries use credential-free HTTPS indexes:

```toml
[registries.research]
index = "https://packages.research.example"

[dependencies]
models = { package = "research:lab/models", version = "=3.4.1" }
```

Credentials never belong in `runmat.toml`, `runmat.lock`, source identities, signed URLs, or diagnostics. Authentication is supplied by the matching host credential provider and is sent only to the explicitly configured origin. Source replacement is explicit:

```toml
[source-replacements.default]
replace-with = "mirror"

[registries.mirror]
index = "https://packages-mirror.example"
```

A mirror changes acquisition authority, not package identity or verification policy. Candidate metadata and downloaded artifacts must still match the locked registry origin/package/release/version/digests and configured signature/provenance policy.

## Search And Release State

Public package search and package/release pages expose only public packages and current public release state. Search results and details include ownership, official/wrapper labels, versions, yanks/revocations, advisories, license, SBOM/provenance summary, and release timestamps. Public discovery is anonymous and read-only; private packages and administrative state require authorization.

Yanking prevents new selection without rewriting existing locks. Revocation is stronger: clients reject a revoked release under the applicable security policy. Advisories remain visible independently of whether a release is yanked. A lock preserves exact historical identity but does not turn mutable Server policy into local authority.

## Administration

Organization administrators create package identities, choose public or private visibility, manage maintainers/owners, configure publication policy, review staged publications, inspect provenance/license/SBOM data, approve or reject releases, yank or revoke releases, publish advisories, rotate signing material, and manage recipient grants for private packages.

Publication state is monotonic:

```text
created → uploaded → verified → approved → finalized
                         └──────→ rejected
```

An immutable finalized version cannot be overwritten. Repeating an operation with the same idempotency key and same input returns the existing result; conflicting reuse is rejected. Approval cannot bypass verification, and finalization cannot bypass approval. Audit records identify the principal, organization, package, publication, transition, policy decision, and request correlation without recording secrets or plaintext private content.

## Official And Wrapper Labels

An official label is registry-controlled and indicates a package maintained or endorsed under RunMat’s official-package policy. A wrapper label identifies a package whose primary purpose is to integrate an external native/service/toolbox dependency. Package owners cannot self-assert either label through artifact metadata. Labels are presentation and policy metadata, not a substitute for signatures, provenance, capability declarations, or advisories.
