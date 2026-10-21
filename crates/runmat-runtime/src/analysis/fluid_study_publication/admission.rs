//! Authenticate the original cold source before recovering any public field storage.
use super::*;
use runmat_analysis_fea::results::field::RecoveredFieldArtifact;

pub(super) fn validate(
    completed: &FluidStudyCompletion,
    source: &FluidStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisCfdRunOptions,
    definition: Digest,
) -> Result<RecoveredFieldArtifact, String> {
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
        return Err("Fluid publication differs from its authenticated definition/root".into());
    }
    validate_options(
        options,
        completed.workload.policy,
        completed.workload.geometry_policy,
    )?;
    let expected = runmat_analysis_fea::study::plan_fluid_study(
        &completed.compiled,
        runmat_analysis_fea::study::FluidStudyPlanningPolicy {
            numerical: options.numerical,
            geometry: options.geometry,
        },
    )
    .map_err(|e| e.to_string())?;
    if expected != completed.workload {
        return Err("Fluid publication differs from original authored planning".into());
    }
    completed
        .result
        .validate_for(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    completed
        .numerical_receipt
        .validate_against(&completed.workload, &completed.result, mesh)
        .map_err(|e| e.to_string())?;
    // Publication consumes original governed recovery partitions, not a second
    // numerical-to-field copy outside the retained recovery execution authority.
    let fields = source.materialize_fields(completed)?;
    completed
        .result
        .validate_native_fields(&completed.workload, mesh, &fields)
        .map_err(|e| e.to_string())?;
    if fields.manifest().map_err(|e| e.to_string())? != completed.fields {
        return Err(
            "Fluid projection differs from original signed Cell U/p and all owner-face q".into(),
        );
    }
    Ok(fields)
}

fn validate_options(
    options: &AnalysisCfdRunOptions,
    numerical: runmat_analysis_fea::physics::cfd::canonical::CanonicalFluidPolicy,
    geometry: runmat_analysis_fea::topology::FiniteVolumeProjectionPolicy,
) -> Result<(), String> {
    if options.precision_mode != super::super::PrecisionMode::Fp64
        || options.numerical != numerical
        || options.geometry != geometry
    {
        return Err("Fluid publication controls differ from original numerical authority".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
