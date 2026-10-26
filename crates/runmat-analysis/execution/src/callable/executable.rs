//! Recover callable facts from their authenticated portable component, not a second authority.

use std::collections::BTreeSet;

use runmat_analysis_core::PhysicalSignal;
use runmat_execution::{
    ExecutableComponentKind, ExecutableEntrypointKind, ExecutableUnitEnvelope, ProgramRevision,
};
use runmat_execution_artifact::BundleCallable;
use runmat_mir::analysis::AnalysisStore;
use runmat_types::CapabilityRequirement;

use crate::{AdmittedSignalGenerator, AnalysisBridgeError, AnalysisBridgeResult};

#[cfg(test)]
pub(crate) mod tests;

/// The caller must anchor the envelope to the enclosing frozen input contract. This check
/// authenticates its internal component identities; it does not authorize a new executable.
pub(crate) fn admit_named_signal_generator_from_executable(
    signal: &PhysicalSignal,
    revision: ProgramRevision,
    callable: BundleCallable,
    executable: &ExecutableUnitEnvelope,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    let function = verified_named_function(executable, &revision, &callable)?;
    let identity = &executable.manifest.identity;
    let compiled_binding = crate::named_callable::CompiledCallableBinding::from_selected_identity(
        identity,
        &callable.qualified_name,
    )?;
    super::admit_named_signal_generator(
        signal,
        revision,
        callable,
        compiled_binding,
        &function,
        available_capabilities,
    )
}

/// Recover the function contract from the authenticated executable component.
pub(crate) fn verified_named_function(
    executable: &ExecutableUnitEnvelope,
    revision: &ProgramRevision,
    callable: &BundleCallable,
) -> AnalysisBridgeResult<runmat_mir::analysis::FunctionAnalysis> {
    executable.validate()?;
    let identity = &executable.manifest.identity;
    for (matches, diagnostic) in [
        (
            &identity.program == revision,
            "portable executable program differs from the frozen named callable",
        ),
        (
            identity.root_package == callable.owner_identity,
            "portable executable root package differs from the frozen named callable",
        ),
        (
            identity.entrypoint_kind == ExecutableEntrypointKind::Function,
            "portable executable entrypoint kind differs from the frozen named callable",
        ),
    ] {
        if !matches {
            return Err(AnalysisBridgeError::Identity(diagnostic));
        }
    }
    let payload = executable
        .component(ExecutableComponentKind::Analysis)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("portable executable has no analysis component".into())
        })?;
    let analysis: AnalysisStore = serde_json::from_slice(&payload.bytes).map_err(|error| {
        AnalysisBridgeError::Invalid(format!("invalid executable analysis: {error}"))
    })?;
    let canonical = serde_json::to_vec(
        &serde_json::to_value(&analysis)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if canonical != payload.bytes {
        return Err(AnalysisBridgeError::Invalid(
            "executable analysis is not canonical JSON".into(),
        ));
    }
    analysis
        .revision
        .validate_current()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let revisions = &executable.manifest.revisions;
    if analysis.revision.schema_version != revisions.analysis_schema
        || analysis.revision.fact_schema_major != revisions.contract_schema
        || analysis.revision.catalog_schema != u32::from(revisions.catalog_schema)
        || analysis.revision.catalog_fingerprint != *revisions.catalog_fingerprint.bytes()
        || analysis
            .functions
            .windows(2)
            .any(|pair| pair[0].function >= pair[1].function)
    {
        return Err(AnalysisBridgeError::Identity(
            "executable analysis revision or function inventory differs from its manifest",
        ));
    }
    let function = analysis
        .function(identity.entrypoint_function.ok_or_else(|| {
            AnalysisBridgeError::Invalid("executable has no function entrypoint".into())
        })?)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "executable entrypoint has no static function facts".into(),
            )
        })?;
    if !function
        .capabilities
        .0
        .is_subset(&executable.manifest.capabilities.0)
    {
        return Err(AnalysisBridgeError::Identity(
            "executable manifest omits callable capabilities",
        ));
    }
    Ok(function.clone())
}
