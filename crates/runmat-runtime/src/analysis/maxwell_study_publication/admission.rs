//! Runtime projection admission. Whole-study closure replay belongs to the supplied source.
use super::*;
use runmat_analysis_fea::results::field::RecoveredFieldArtifact;

pub(super) fn validate(
    completed: &MaxwellStudyCompletion,
    source: &MaxwellStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisMaxwellRunOptions,
    definition: Digest,
) -> Result<RecoveredFieldArtifact, String> {
    // The placement adapter reopens the complete immutable root and compares the
    // completion, including observations, accuracy, states and accounting. Runtime
    // must not grow a second coordinator or independently recompute any fields.
    source.authenticate(completed)?;
    completed.catalog.validate().map_err(|e| e.to_string())?;
    completed
        .compiled
        .validate_against(mesh)
        .map_err(|e| e.to_string())?;
    evidence.validate(mesh).map_err(|e| e.to_string())?;
    if !completed.catalog.roots.contains(&completed.root)
        || completed
            .compiled
            .definition_digest()
            .map_err(|e| e.to_string())?
            != definition
        || completed.workload.model != completed.compiled.model
    {
        return Err("Maxwell publication differs from its authenticated definition/root".into());
    }
    validate_options(options, completed.workload.relative_residual_target)?;
    let expected = runmat_analysis_fea::study::plan_maxwell_study(
        &completed.compiled,
        runmat_analysis_fea::study::MaxwellStudyPlanningPolicy {},
    )
    .map_err(|e| e.to_string())?;
    if expected != completed.workload {
        return Err("Maxwell publication differs from original authored planning".into());
    }
    completed
        .result
        .validate_for(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    completed
        .numerical_receipt
        .validate_against(&completed.workload, &completed.result, mesh)
        .map_err(|e| e.to_string())?;
    let request = completed
        .result
        .native_field_request_from_result()
        .map_err(|e| e.to_string())?;
    let fields = completed
        .result
        .recover_native_fields_from_result(&request)
        .map_err(|e| e.to_string())?;
    if fields.manifest().map_err(|e| e.to_string())? != completed.fields {
        return Err("Maxwell projection differs from original complex native values".into());
    }
    Ok(fields)
}

fn validate_options(options: &AnalysisMaxwellRunOptions, residual: f64) -> Result<(), String> {
    if options.residual_target != residual
        || options.precision_mode != super::super::PrecisionMode::Fp64
    {
        return Err("Maxwell publication controls differ from original numerical authority".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_maxwell_precision_and_tolerance_cannot_be_replaced() {
        let mut options = AnalysisMaxwellRunOptions::default();
        assert!(validate_options(&options, options.residual_target).is_ok());
        assert!(validate_options(&options, 1e-9).is_err());
        options.precision_mode = super::super::super::PrecisionMode::Fp32;
        assert!(validate_options(&options, options.residual_target).is_err());
    }
}
