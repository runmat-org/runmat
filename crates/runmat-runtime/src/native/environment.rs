//! Fingerprints of the actual linked Runtime and builtin catalog.
//!
//! Core and the standalone host use the same owner so a selected product is
//! checked against linked code rather than against another serialized field.

use runmat_execution::Digest;

pub const PROGRAM_RUNTIME_ABI_SCHEMA: u32 = 2;

pub fn linked_runtime_fingerprint() -> Digest {
    Digest::sha256(format!(
        "runmat-runtime-compatibility-v2\0{}\0{}\0{}",
        env!("CARGO_PKG_VERSION"),
        PROGRAM_RUNTIME_ABI_SCHEMA,
        super::native_abi_contract_fingerprint()
    ))
}

pub fn linked_catalog_fingerprint() -> Digest {
    Digest::from_bytes(runmat_builtins::builtin_catalog_fingerprint())
}
