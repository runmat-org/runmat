use runmat_execution::ProgramEnvironment;

use crate::CompatMode;

pub const PROGRAM_SEMANTIC_SCHEMA: u32 = runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2;
pub const PROGRAM_COMPILER_SCHEMA: u32 = runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2;
pub const PROGRAM_RUNTIME_ABI_SCHEMA: u32 = runmat_runtime::native::PROGRAM_RUNTIME_ABI_SCHEMA;

/// Describe the portable execution compatibility of this RunMat runtime.
///
/// This is constructed by Core—the layer that composes the parser/compiler,
/// runtime, and builtin catalog—so lower-level test and execution contracts do
/// not invent placeholder fingerprints.
#[tracing::instrument(name = "runtime.environment", skip_all)]
pub fn program_environment(compatibility_mode: CompatMode) -> ProgramEnvironment {
    runmat_runtime::builtins::wasm_registry::register_all();
    let runtime_fingerprint = tracing::info_span!("runtime.environment.runtime_fingerprint")
        .in_scope(runmat_runtime::native::linked_runtime_fingerprint);
    let catalog_fingerprint = tracing::info_span!("runtime.environment.catalog_fingerprint")
        .in_scope(runmat_runtime::native::linked_catalog_fingerprint);
    ProgramEnvironment::new(
        PROGRAM_SEMANTIC_SCHEMA,
        PROGRAM_COMPILER_SCHEMA,
        runtime_fingerprint,
        catalog_fingerprint,
        execution_compatibility_mode(compatibility_mode),
    )
    .expect("Core program compatibility constants are valid")
}

fn execution_compatibility_mode(mode: CompatMode) -> runmat_execution::LanguageCompatibilityMode {
    match mode {
        CompatMode::RunMat => runmat_execution::LanguageCompatibilityMode::RunMat,
        CompatMode::Matlab => runmat_execution::LanguageCompatibilityMode::Matlab,
        CompatMode::Strict => runmat_execution::LanguageCompatibilityMode::Strict,
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
        assert_eq!(
            first.compatibility_mode,
            runmat_execution::LanguageCompatibilityMode::Matlab
        );
        assert_ne!(first.runtime_fingerprint, first.catalog_fingerprint);
    }
}
