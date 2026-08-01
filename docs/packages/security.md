---
title: "Package Security"
category: "Packages"
section: "8.4"
last_updated: "August 1, 2026"
---

# Package Security

RunMat treats dependency selection, downloaded bytes, cached objects, private decryption, and publication transitions as separate trust boundaries.

## Integrity And Immutability

Every immutable dependency is accepted only after its source identity, artifact inventory, content digests, canonical tree digest, manifest identity/version, and locked expectation agree. Verification happens before publication into the shared cache or exposure through a mount. A failed or interrupted transfer cannot become a complete object, and a corrupt cached payload becomes an explicit corruption record and recoverable miss.

Locks exclude native paths, cache directories, IndexedDB keys, browser mount roots, tab/worker IDs, credentials, signed URLs, and authorization artifacts. This makes identities portable while preventing host-local or secret material from becoming reproducibility inputs.

Materialized native trees and browser mounts are read-only views of verified content. Leases and pins protect active graphs from garbage collection. GC never treats an incomplete transaction, stale writer, or unverified staging directory as live package content.

## Private Packages

Private release bodies are encrypted client-side. The registry stores ciphertext, public recipient keys, wrapped content-key envelopes, and signed metadata. It does not receive plaintext content keys or recipient private keys.

Native recipient private keys are generated locally and stored in the OS credential store. Browser keys are non-extractable WebCrypto P-256 keys stored in IndexedDB. Registration sends only public material, and the client verifies the Server-returned fingerprint before retaining a key. If local persistence fails after registration, the client performs a compensating revoke. Revocation succeeds on the Server before local private material is removed.

Browser decrypted artifacts are verified and mounted in volatile memory by default. Plaintext package bodies do not enter the persistent package cache. Logout, organization/access-token change, key revocation, filesystem/project switch, worker disposal, and explicit invalidation clear private mounts and best-effort zero owned bytes. Closing a browser cannot revoke plaintext a hostile process already copied; the policy prevents RunMat from making an additional durable plaintext copy.

## Authorization And Revocation

Package visibility, ownership, publication transitions, release administration, recipient-key registration/revocation, and private transfers are authorized independently. Anonymous access is exact-shape GET/HEAD access to public discovery and public release acquisition only.

Revoking remote access cannot erase plaintext already lawfully delivered to a native client cache or process. Public and unencrypted private snapshots already cached by exact identity can remain available offline according to local policy. Encrypted private artifacts remain ciphertext in durable cache and require a currently available local private key to mount. Administrators should rotate content-key versions and recipient grants when compromise requires cryptographic separation of future releases.

## Reporting Security Issues

Do not publish exploit details, credentials, private artifacts, or recipient material in package metadata or advisories. Use the project’s private security-reporting channel. Preserve the exact package ID, release ID, version, registry origin, lock digest, artifact/tree/release digests, client version, and non-secret diagnostic correlation IDs needed to reproduce the issue.
