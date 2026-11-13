//! Registration verification without constructing the documentation/governance inventory.
use runmat_builtins::{registration_manifest_entries, RegistrationKind};
use sha2::{Digest, Sha256};

use super::{
    environment::hex, RegistrationKindRecord, RegistrationManifest, RegistrationManifestCounts,
    RegistrationManifestRecord,
};

pub fn registration_manifest() -> RegistrationManifest {
    let mut entries = registration_manifest_entries()
        .into_iter()
        .map(|entry| RegistrationManifestRecord {
            kind: match entry.kind {
                RegistrationKind::Builtin => RegistrationKindRecord::Builtin,
                RegistrationKind::Constant => RegistrationKindRecord::Constant,
                RegistrationKind::GpuSpec => RegistrationKindRecord::GpuSpec,
                RegistrationKind::FusionSpec => RegistrationKindRecord::FusionSpec,
            },
            declaration: entry.declaration,
            variant: entry.variant,
            builtin_path: entry.builtin_path,
        })
        .collect::<Vec<_>>();
    entries.sort_unstable();
    let digest = hex(&Sha256::digest(
        serde_json::to_vec(&entries).expect("registration manifest must serialize"),
    ));
    let counts = RegistrationManifestCounts {
        builtin: entries
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::Builtin)
            .count(),
        constant: entries
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::Constant)
            .count(),
        gpu_spec: entries
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::GpuSpec)
            .count(),
        fusion_spec: entries
            .iter()
            .filter(|entry| entry.kind == RegistrationKindRecord::FusionSpec)
            .count(),
    };
    RegistrationManifest {
        schema_version: 1,
        digest,
        counts,
        entries,
    }
}
