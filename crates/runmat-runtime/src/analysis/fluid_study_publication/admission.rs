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
    let layout = completed
        .result
        .native_field_topology(&completed.workload, mesh)
        .map_err(|e| e.to_string())?;
    if layout != completed.layout {
        return Err("Fluid publication differs from original cell/owner-face topology".into());
    }
    // Publication consumes original governed recovery partitions, not a second
    // numerical-to-field copy outside the retained recovery execution authority.
    let fields = source.materialize_fields(completed)?;
    validate_original_values(&fields, completed.result.original())?;
    if fields.manifest().map_err(|e| e.to_string())? != completed.fields {
        return Err(
            "Fluid projection differs from original signed Cell U/p and all owner-face q".into(),
        );
    }
    Ok(fields)
}

/// Borrow original solver arrays: no second field recovery or numeric clone.
fn validate_original_values(
    fields: &RecoveredFieldArtifact,
    original: &runmat_analysis_fea::physics::cfd::canonical::CanonicalFluidResult,
) -> Result<(), String> {
    let definitions = runmat_analysis_fea::execution::fluid_native_field_definition_inventory()
        .map_err(|error| error.to_string())?;
    if fields.fields().len() != definitions.len() {
        return Err("fluid publication omitted or added an original native field".into());
    }
    for (field, definition) in fields.fields().iter().zip(definitions.iter()) {
        if field.definition() != definition
            || field.sample() != runmat_analysis_fea::results::field::RecoveredFieldSample::Final
        {
            return Err(
                "fluid publication reordered or substituted native field definitions".into(),
            );
        }
    }
    let values = |index: usize| {
        fields.fields()[index]
            .storage()
            .as_host_f64()
            .ok_or_else(|| "fluid publication requires original CPU FP64 values".to_string())
    };
    let velocity = values(0)?;
    let count = original
        .velocity_m_per_s
        .len()
        .checked_mul(3)
        .ok_or("fluid publication velocity count overflow")?;
    let exact = |left: &[f64], right: &[f64]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    if velocity.len() != count
        || !velocity
            .iter()
            .zip(original.velocity_m_per_s.iter().flatten())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        || !exact(values(1)?, &original.pressure_pa)
        || !exact(values(2)?, &original.face_volume_flux_m3_per_s)
    {
        return Err("fluid recovered artifact substituted original signed U/p/q values".into());
    }
    Ok(())
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
