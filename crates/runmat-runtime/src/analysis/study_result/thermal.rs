//! Exact typed thermal-prefix completion authority; no inferred source fields or model projection.
use super::*;
use crate::analysis::ThermalStudyCompletion;
use runmat_analysis_core::{ProcedureStepId, StateSource, StateUse};
use runmat_analysis_fea::execution::{CanonicalStateTransition, StructuralLinearStateInput};

pub(in crate::analysis) fn structural_steps(
    result: &StructuralStudyNumericalResult,
) -> Result<(ProcedureStepId, Vec<ProcedureStepId>), String> {
    Ok(match result {
        StructuralStudyNumericalResult::StructuralBuckling(result) => (result.preload_step_id.clone(), vec![result.buckling_step_id.clone()]),
        StructuralStudyNumericalResult::StructuralPrestressedModal(result) => (result.preload_step_id.clone(), vec![result.modal_step_id.clone()]),
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
            let mut consumers = Vec::new();
            if let super::super::StructuralStudyPrestressedFrequencyReductionResult::ModalReduced(reduction) = &result.reduction { consumers.push(reduction.modal_step_id.clone()); }
            consumers.push(result.frequency_step_id.clone());
            (result.preload_step_id.clone(), consumers)
        }
        _ => return Err("thermal prestress requires one supported structural prestress child, without nested wrappers".into()),
    })
}

pub(in crate::analysis) fn validate(
    compiled: &CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    workloads: &StructuralStudyWorkloads,
    thermal: &ThermalStudyCompletion,
    structural: &StructuralStudyNumericalResult,
) -> Result<(), String> {
    let prefix = workloads
        .prestress_prefix()
        .map_err(|error| error.to_string())?;
    let StructuralStudyWorkloads::HeterogeneousSequence { workloads: entries } = workloads else {
        return Err("thermal prefix needs heterogeneous inventory".into());
    };
    let terminal = prefix
        .consumer_step_ids
        .last()
        .and_then(|id| entries.get(id));
    if !matches!(
        (terminal, structural),
        (
            Some(runmat_analysis_fea::execution::StructuralStepWorkload::Buckling { .. }),
            StructuralStudyNumericalResult::StructuralBuckling(_)
        ) | (
            Some(runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedModal { .. }),
            StructuralStudyNumericalResult::StructuralPrestressedModal(_)
        ) | (
            Some(
                runmat_analysis_fea::execution::StructuralStepWorkload::PrestressedFrequency { .. }
            ),
            StructuralStudyNumericalResult::StructuralPrestressedFrequency(_)
        )
    ) {
        return Err("thermal structural result kind differs from authored consumer".into());
    }
    if &thermal.compiled != compiled || prefix.thermal != Some(&thermal.workload) {
        return Err("thermal completion differs from the exact compiled producer".into());
    }
    super::thermal_observation_admission::validate(thermal, mesh)?;
    thermal
        .compiled
        .validate_against(mesh)
        .map_err(|error| error.to_string())?;
    let [input] = prefix
        .preload
        .procedure_step()
        .map_err(|error| error.to_string())?
        .state_inputs
        .as_slice()
    else {
        return Err("thermal preload must have one source".into());
    };
    let StateSource::Upstream {
        producer_step,
        state_id,
    } = &input.source
    else {
        return Err("thermal preload source is not upstream".into());
    };
    let mut matching = thermal
        .states
        .iter()
        .filter(|state| &state.state_id == state_id);
    let state = matching
        .next()
        .ok_or("named thermal state was not retained")?;
    if matching.next().is_some()
        || producer_step != &thermal.workload.step_id
        || state.source_result != thermal.manifest.identity()
    {
        return Err("thermal state differs from its exact producer result".into());
    }
    let transition = CanonicalStateTransition::bind_upstream(
        &compiled.model,
        &prefix.preload.step_id,
        state,
        StateUse::Prestress,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .map_err(|error| error.to_string())?;
    let (preload, frozen_project_digest) = match structural {
        StructuralStudyNumericalResult::StructuralBuckling(result) => {
            (&result.preload_result, result.frozen_project_digest)
        }
        StructuralStudyNumericalResult::StructuralPrestressedModal(result) => {
            (&result.preload_result, result.frozen_project_digest)
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
            (&result.preload_result, result.frozen_project_digest)
        }
        _ => return Err("unsupported thermal prestress child".into()),
    };
    if frozen_project_digest != thermal.frozen_project_digest {
        return Err(
            "thermal and structural completions have different frozen project identities".into(),
        );
    }
    let mut bound = prefix.preload.clone();
    bound.state_input = Some(StructuralLinearStateInput::ThermalPrestress(transition));
    preload
        .validate_thermal_workload(&bound)
        .map_err(|error| error.to_string())?;
    let manifest = &thermal.manifest;
    manifest.validate().map_err(|error| error.to_string())?;
    if manifest.source_workload
        != thermal
            .workload
            .identity()
            .map_err(|error| error.to_string())?
    {
        return Err("thermal result manifest differs from its retained source".into());
    }
    thermal
        .numerical_receipt
        .validate_against(&thermal.workload, manifest)
        .map_err(|error| error.to_string())?;
    thermal
        .accuracy
        .validate()
        .map_err(|error| error.to_string())?;
    if thermal.accuracy.source_result != manifest.identity()
        || thermal.accuracy.producer != manifest.producer
        || thermal.accuracy.source_kind
            != runmat_analysis_fea::execution::AccuracySourceKind::Thermal
    {
        return Err("thermal accuracy differs from retained producer result".into());
    }
    Ok(())
}
