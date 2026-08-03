use runmat_execution::{Digest, ProgramEnvironment};

use crate::CompatMode;

pub const PROGRAM_SEMANTIC_SCHEMA: u32 = 1;
pub const PROGRAM_COMPILER_SCHEMA: u32 = 1;
pub const PROGRAM_RUNTIME_ABI_SCHEMA: u32 = 1;

/// Describe the portable execution compatibility of this RunMat runtime.
///
/// This is constructed by Core—the layer that composes the parser/compiler,
/// runtime, and builtin catalog—so lower-level test and execution contracts do
/// not invent placeholder fingerprints.
pub fn program_environment(compatibility_mode: CompatMode) -> ProgramEnvironment {
    runmat_runtime::builtins::wasm_registry::register_all();
    let runtime_fingerprint = Digest::sha256(format!(
        "runmat-runtime-compatibility-v1\0{}\0{}",
        env!("CARGO_PKG_VERSION"),
        PROGRAM_RUNTIME_ABI_SCHEMA
    ));
    let catalog_fingerprint = Digest::from_bytes(runmat_builtins::builtin_catalog_fingerprint());
    ProgramEnvironment::new(
        PROGRAM_SEMANTIC_SCHEMA,
        PROGRAM_COMPILER_SCHEMA,
        runtime_fingerprint,
        catalog_fingerprint,
        compatibility_mode_name(compatibility_mode),
    )
    .expect("Core program compatibility constants are valid")
}

fn compatibility_mode_name(mode: CompatMode) -> &'static str {
    match mode {
        CompatMode::RunMat => "runmat",
        CompatMode::Matlab => "matlab",
        CompatMode::Strict => "strict",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_is_stable_and_preserves_compatibility_mode() {
        let first = program_environment(CompatMode::Matlab);
        let second = program_environment(CompatMode::Matlab);
        assert_eq!(first, second);
        assert_eq!(first.compatibility_mode, "matlab");
        assert_ne!(first.runtime_fingerprint, first.catalog_fingerprint);
    }
}
