//! Publish an admitted thermal prefix and its structural suffix as one complete study.
use super::*;
use crate::analysis::ThermalStudyCompletion;
use runmat_analysis_fea::execution::{StructuralStepWorkload, StructuralStudyWorkloads};

use crate::analysis::study_publication::thermal::materialization as thermal_run;

pub(super) fn complete(
    intent: PreparedStructuralStudyIntent,
    study_model: runmat_analysis_core::AnalysisModel,
    persisted_mesh: crate::analysis::meshing::ResolvedStudyMesh,
    mesh: runmat_meshing_core::SolverMeshArtifact,
    mut workloads: BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralStepWorkload>,
    thermal: ThermalStudyCompletion,
    thermal_source: crate::analysis::ThermalStudyObjectSource,
    structural: StructuralStudyNumericalResult,
    identity: AnalysisStudyCompletionIdentity,
    preparation: crate::analysis::study_publication::preparation::PublicationPreparation,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let invalid = |message| {
        intent_completion_error(
            &intent,
            "RM.FEA.RUN_STUDY.THERMAL_PREFIX_INVALID",
            OperationErrorType::Contract,
            OperationErrorSeverity::Fatal,
            message,
        )
    };
    {
        let _source_phase = preparation.staging.begin_source_phase().map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &intent.context,
                invalid,
            )
        })?;
        let maximum = preparation
            .staging
            .source_replay_allowance(&thermal.catalog.objects)
            .map_err(|error| {
                crate::analysis::prepared_study_publication::publication_failure(
                    error,
                    &intent.context,
                    invalid,
                )
            })?;
        thermal_source
            .authenticate_admitted(&thermal, maximum)
            .map_err(|error| {
                crate::analysis::prepared_study_publication::publication_failure(
                    error,
                    &intent.context,
                    invalid,
                )
            })?;
    }
    thermal_source
        .persist(&thermal, intent.authored_study_digest, &preparation.staging)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &intent.context,
                invalid,
            )
        })?;
    if thermal.compiled.model != study_model {
        return Err(invalid(
            "thermal completion differs from resolved model".to_string(),
        ));
    }
    let inventory = StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: workloads.clone(),
    };
    crate::analysis::study_result::thermal::validate(
        &thermal.compiled,
        &mesh,
        &inventory,
        &thermal,
        &structural,
    )
    .map_err(invalid)?;
    let (preload, consumers) =
        crate::analysis::study_result::thermal::structural_steps(&structural).map_err(invalid)?;
    let thermal_run_id = identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Thermal,
        intent.authored_study_digest,
        &thermal.workload.step_id,
    );
    let preload_run_id = identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Structural,
        intent.authored_study_digest,
        &preload,
    );
    let consumer_run_ids = consumers
        .iter()
        .map(|step| {
            identity.local_run_id(
                runmat_analysis_core::PhysicsFamily::Structural,
                intent.authored_study_digest,
                step,
            )
        })
        .collect::<Vec<_>>();
    let mut steps = intent
        .steps
        .iter()
        .map(|step| super::super::PreparedStudyStep {
            authored_execution: step.authored_execution.clone(),
            procedure_step: step.procedure_step.clone(),
        })
        .collect::<Vec<_>>();
    if steps.first().map(|step| &step.procedure_step.step_id) != Some(&thermal.workload.step_id) {
        return Err(invalid(
            "thermal prefix is not the first authored step".to_string(),
        ));
    }
    let mut authored = steps.remove(0);
    let AnalysisSolverOptions::ThermalDiffusion(options) =
        &mut authored.authored_execution.solver_options
    else {
        return Err(invalid(
            "thermal step lacks thermal publication controls".to_string(),
        ));
    };
    options.solver_mesh_artifact_path = persisted_mesh.artifact_path.clone();
    let run = {
        let _source_phase = preparation.staging.begin_source_phase().map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error,
                &intent.context,
                invalid,
            )
        })?;
        thermal_run::admit(
            &thermal,
            &thermal_source,
            &mesh,
            options,
            &thermal_run_id,
            &intent.context,
        )
        .map_err(invalid)?
    };
    workloads.remove(&thermal.workload.step_id);
    let prefix = super::super::publication::ThermalPrefixCompletion {
        completed: CompletedStudyStep {
            step_id: thermal.workload.step_id.clone(),
            solver_result_identity: thermal.manifest.identity(),
            run: CompletedStudyStepRun::Standard(Box::new(run)),
            resolved_solver_options: authored.authored_execution.solver_options.clone(),
        },
        authored,
        states: thermal.states.clone(),
        observation_source: thermal_source,
        thermal: Box::new(thermal),
    };
    let prepared = PreparedStudyCompletion {
        thermal_prefix: Some(Box::new(prefix)),
        context: intent.context,
        study_id: intent.study_id,
        study_fingerprint: study_fingerprint_from_digest(intent.authored_study_digest),
        authored_study_digest: intent.authored_study_digest,
        model: study_model,
        steps,
        operation_sequence: intent.operation_sequence,
        solver_mesh_artifact_path: persisted_mesh.artifact_path,
        meshing_evidence_artifact_path: persisted_mesh.evidence_path,
    };
    match structural {
        StructuralStudyNumericalResult::StructuralBuckling(result) => complete_buckling_study(
            prepared,
            mesh,
            workloads,
            *result,
            preload_run_id,
            consumer_run_ids[0].clone(),
            preparation,
        ),
        StructuralStudyNumericalResult::StructuralPrestressedModal(result) => {
            complete_prestressed_modal_study(
                prepared,
                mesh,
                workloads,
                *result,
                preload_run_id,
                consumer_run_ids[0].clone(),
                preparation,
            )
        }
        StructuralStudyNumericalResult::StructuralPrestressedFrequency(result) => {
            let modal_id = (consumer_run_ids.len() == 2).then(|| consumer_run_ids[0].clone());
            let frequency_id = consumer_run_ids.last().unwrap().clone();
            complete_prestressed_frequency_study(
                prepared,
                mesh,
                workloads,
                *result,
                preload_run_id,
                modal_id,
                frequency_id,
                preparation,
            )
        }
        _ => Err(completion_error(
            &prepared,
            "RM.FEA.RUN_STUDY.RESULT_KIND_MISMATCH",
            "unsupported thermal child",
        )),
    }
}
