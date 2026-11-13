//! Exact source, topology, authored inventory and numerical-control admission.
use super::*;
use runmat_analysis_core::{AnalysisProcedure, PhysicsFamily};
use runmat_analysis_fea::ThermalProcedureControl;

pub(super) fn validate(
    completed: &ThermalStudyCompletion,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisThermalRunOptions,
    definition: Digest,
) -> Result<(), String> {
    // The caller has authenticated the original source under its captured resource lease.
    completed
        .compiled
        .validate_against(mesh)
        .map_err(|e| e.to_string())?;
    evidence.validate(mesh).map_err(|e| e.to_string())?;
    if completed
        .compiled
        .definition_digest()
        .map_err(|e| e.to_string())?
        != definition
    {
        return Err(
            "thermal publication definition differs from authenticated compiled study".into(),
        );
    }
    completed.workload.validate().map_err(|e| e.to_string())?;
    let [step] = completed.compiled.model.sequence.steps.as_slice() else {
        return Err("standalone thermal publication requires exactly one authored step".into());
    };
    if completed.workload.model != completed.compiled.model
        || step.step_id != completed.workload.step_id
        || step.family != PhysicsFamily::Thermal
        || !matches!(
            step.procedure,
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
        )
    {
        return Err("thermal publication workload differs from standalone authored study".into());
    }
    validate_options(
        options,
        completed.workload.procedure,
        completed.workload.residual_target,
    )?;
    completed
        .numerical_receipt
        .validate_against(&completed.workload, &completed.manifest)
        .map_err(|e| e.to_string())?;
    let manifest = &completed.manifest;
    if manifest.topology_digest != Digest::from_bytes(*mesh.canonical_digest.bytes())
        || !manifest.ordered_node_ids.iter().copied().eq(mesh
            .topology
            .nodes
            .iter()
            .map(|n| n.node_id))
        || !manifest.ordered_element_ids.iter().copied().eq(mesh
            .topology
            .volume_elements
            .iter()
            .map(|e| e.element_id))
    {
        return Err("thermal publication result differs from authoritative mesh ordering".into());
    }
    super::super::study_result::thermal_observation_admission::validate(completed, mesh)?;
    let requests = completed
        .workload
        .state_publication_requests(manifest.topology_digest, manifest.identity())
        .map_err(|e| e.to_string())?;
    if requests.len() != completed.states.len() {
        return Err("thermal publication omits or adds an authored persisted state".into());
    }
    for request in requests {
        let matches = completed
            .states
            .iter()
            .filter(|state| state.state_id == request.declaration.state_id)
            .collect::<Vec<_>>();
        let [state] = matches.as_slice() else {
            return Err("thermal publication requires exactly one state per declaration".into());
        };
        state.validate().map_err(|e| e.to_string())?;
        if state.producer != request.producer
            || state.source_result != request.source_result
            || state.topology_digest != request.topology_digest
            || state.payload.thermal().is_none()
            || state.kind != request.declaration.kind
            || state.completeness != request.declaration.completeness
            || state.persistence != request.declaration.persistence
        {
            return Err("thermal publication state differs from its exact producer result".into());
        }
    }
    Ok(())
}

fn validate_options(
    options: &AnalysisThermalRunOptions,
    procedure: ThermalProcedureControl,
    residual_target: f64,
) -> Result<(), String> {
    let samples = match procedure {
        ThermalProcedureControl::Steady => 1,
        ThermalProcedureControl::BackwardEuler { snapshot_count } => snapshot_count,
    };
    if options.snapshot_count != samples
        || options.residual_warn_threshold != residual_target
        || options.precision_mode != super::super::PrecisionMode::Fp64
    {
        return Err("thermal publication options differ from completed numerical controls".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_controls_cannot_substitute_schedule_precision_or_residual() {
        let mut options = AnalysisThermalRunOptions {
            snapshot_count: 1,
            residual_warn_threshold: 1e-8,
            ..Default::default()
        };
        assert!(validate_options(&options, ThermalProcedureControl::Steady, 1e-8).is_ok());
        options.snapshot_count = 2;
        assert!(validate_options(&options, ThermalProcedureControl::Steady, 1e-8).is_err());
        assert!(validate_options(
            &options,
            ThermalProcedureControl::BackwardEuler { snapshot_count: 2 },
            1e-8
        )
        .is_ok());
        options.precision_mode = super::super::super::PrecisionMode::Fp32;
        assert!(validate_options(
            &options,
            ThermalProcedureControl::BackwardEuler { snapshot_count: 2 },
            1e-8
        )
        .is_err());
        options.precision_mode = super::super::super::PrecisionMode::Fp64;
        assert!(validate_options(
            &options,
            ThermalProcedureControl::BackwardEuler { snapshot_count: 2 },
            1e-9
        )
        .is_err());
    }
}
