//! Restore the compiler-owned product; never substitute a bare VM registry for a session unit.
use runmat_execution::{
    ExecutableComponentKind as Kind, ExecutableEntrypointKind, ExecutableUnitEnvelope,
};
use serde::de::DeserializeOwned;

use super::{ExecutableRevision, ExecutableUnit};
mod sources;
#[cfg(test)]
mod tests;

impl ExecutableUnit {
    /// Decode a current compiler product with its original source, MIR, analysis,
    /// layout, function inventory, coverage and extension metadata. This checks
    /// schema and component consistency; it does not recompile or attest source semantics.
    pub fn from_portable_envelope(envelope: &ExecutableUnitEnvelope) -> Result<Self, String> {
        envelope.validate().map_err(|error| error.to_string())?;
        envelope
            .manifest
            .validate_for(&runmat_execution::ExecutableSectionSupport::default())
            .map_err(|error| error.to_string())?;
        require_environment(envelope)?;
        let mir: runmat_mir::MirAssembly = decode(envelope, Kind::Mir)?;
        let analysis: runmat_mir::analysis::AnalysisStore = decode(envelope, Kind::Analysis)?;
        analysis
            .revision
            .validate_current()
            .map_err(|error| error.to_string())?;
        let mut bytecode: runmat_vm::Bytecode = decode(envelope, Kind::Bytecode)?;
        if !bytecode.bound_functions.is_empty()
            || !bytecode.function_registry.functions.is_empty()
            || bytecode.layout.is_some()
        {
            return Err("portable bytecode contains duplicate component authorities".into());
        }
        let registry: runmat_vm::FunctionRegistry = decode(envelope, Kind::FunctionRegistry)?;
        let layout: runmat_vm::VmAssemblyLayout = decode(envelope, Kind::VmLayout)?;
        let (source, source_map) = sources::decode(envelope, &bytecode)?;
        let revision = ExecutableRevision::derive(
            &source,
            Some(envelope.manifest.identity.program.clone()),
            envelope.manifest.identity.program.environment(),
        );
        if revision
            .source_digest
            .parse::<runmat_execution::Digest>()
            .map_err(|error| error.to_string())?
            != envelope.manifest.identity.source_digest
        {
            return Err("portable primary source differs from its original identity".into());
        }
        bytecode.bound_functions = registry.functions.clone();
        bytecode.function_registry = registry;
        bytecode.layout = Some(layout);
        let mut unit = Self::new(source, revision, source_map, mir, analysis, bytecode)?;
        unit.interop = envelope.manifest.interop.clone();
        unit.parallel = envelope.manifest.parallel.clone();
        unit.optional_sections = envelope.manifest.optional_sections.clone();
        let restored = match envelope.manifest.identity.entrypoint_kind {
            ExecutableEntrypointKind::Function => {
                unit.portable_envelope_for_function(envelope.manifest.identity.entrypoint_function)?
            }
            ExecutableEntrypointKind::Script => unit.portable_envelope_for_selection(None)?,
        };
        if restored != *envelope {
            return Err("portable product differs from its canonical source, coverage, region or component inventory".into());
        }
        Ok(unit)
    }
}

fn decode<T: DeserializeOwned>(envelope: &ExecutableUnitEnvelope, kind: Kind) -> Result<T, String> {
    let bytes = &envelope
        .component(kind)
        .ok_or_else(|| format!("portable product lacks {kind:?}"))?
        .bytes;
    serde_json::from_slice(bytes).map_err(|error| format!("invalid portable {kind:?}: {error}"))
}

fn require_environment(envelope: &ExecutableUnitEnvelope) -> Result<(), String> {
    let original = envelope.manifest.identity.program.environment();
    let mode = match original.compatibility_mode.as_str() {
        "matlab" => crate::CompatMode::Matlab,
        "runmat" => crate::CompatMode::RunMat,
        "strict" => crate::CompatMode::Strict,
        _ => return Err("portable language mode is unsupported".into()),
    };
    if original != crate::program_environment(mode) {
        return Err("portable compiler/runtime environment differs from this consumer".into());
    }
    Ok(())
}
