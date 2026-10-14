//! Canonical foreign type requirements shared by portable inference and
//! configured artifact issuance. This module is independent of native hosts.

pub(crate) fn python_environment_requirement() -> runmat_types::ForeignRequirement {
    runmat_types::ForeignRequirement {
        type_identity: runmat_types::ForeignTypeIdentity {
            family: "python".into(),
            name: "environment".into(),
            version: 1,
        },
        ownership: runmat_types::ForeignOwnership::Shared,
        affinity: runmat_types::ForeignAffinity::OriginProcess,
        lifetime: runmat_types::ForeignLifetime::Session,
        capabilities: vec![runmat_types::ForeignCapability::Invoke],
        wasm: runmat_types::WasmInteropPolicy::Reject,
    }
}
