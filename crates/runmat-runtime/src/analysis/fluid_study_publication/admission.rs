//! Authenticate the original cold source before recovering any public field storage.
use super::*;
use runmat_analysis_fea::{results::field::RecoveredFieldArtifact, study::AnalysisStudyDefinition};

pub(super) fn validate(
    completed: &FluidStudyCompletion,
    source: &FluidStudyObjectSource,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisCfdRunOptions,
    definition: AnalysisStudyDefinition<'_>,
) -> Result<RecoveredFieldArtifact, String> {
    source.authenticate(completed)?;
    completed.catalog.validate().map_err(|e| e.to_string())?;
    // Authenticates the mesh content against its canonical digest.
    evidence.validate(mesh).map_err(|e| e.to_string())?;
    // The coordinator planned the product on this mesh (sealed); `validate_options` below
    // requires these controls to be its workload's effective ones. Compare the bindings
    // instead of re-compiling properties and re-planning the study.
    completed
        .product
        .validate_on_mesh(mesh)
        .map_err(|e| e.to_string())?;
    if !completed.catalog.roots.contains(&completed.root) || !definition.is_of(completed.compiled())
    {
        return Err("Fluid publication differs from its authenticated definition/root".into());
    }
    validate_options(
        options,
        completed.workload().policy,
        completed.workload().geometry_policy,
    )?;
    // Validate the result against its source once; the receipt and field checks reuse it.
    let result = completed
        .result
        .on_source(completed.workload(), mesh)
        .map_err(|e| e.to_string())?;
    completed
        .numerical_receipt
        .validate_against_on(&result, mesh)
        .map_err(|e| e.to_string())?;
    // Publication consumes original governed recovery partitions, not a second
    // numerical-to-field copy outside the retained recovery execution authority.
    let fields = source.materialize_fields(completed)?;
    result
        .validate_native_fields(&fields)
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
