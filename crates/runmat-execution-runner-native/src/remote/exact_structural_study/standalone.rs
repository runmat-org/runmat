//! Execution of individual structural analysis families after exact input resolution.
use super::*;
pub(super) async fn modal(
    input: Composition<'_, '_>,
    workload: Box<runmat_analysis_fea::execution::StructuralModalWorkload>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let Composition {
        pool,
        request,
        decoded,
        completed,
        region_resolution_objects,
        completed_inputs,
        unresolved_workloads_digest,
        policy,
        recovery,
    } = input;
    if matches!(
        &recovery,
        StructuralTransientRecoveryAdmission::Available {
            restored: Some(_),
            ..
        }
    ) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "modal extraction cannot consume a transient recovery closure".into(),
        ));
    }
    let completion = match crate::execute_structural_modal_remotely(
        Arc::clone(&pool),
        runmat_analysis_execution::StructuralModalTaskPreparation {
            local_execution: decoded.local_execution,
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            program_revision: request.recipe.program_revision.clone(),
            artifact_access: decoded.artifact_access.clone(),
            workload: *workload,
            mesh: completed.result.artifact.clone(),
            prestress_state: None,
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
            solve_selection:
                runmat_analysis_fea::planning::resource::ModalSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        },
        policy,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) = crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralModal {
            inputs: completed_inputs,
            completion: &completion,
            unresolved_workloads_digest,
        },
    )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}

pub(super) async fn linear(
    input: Composition<'_, '_>,
    workloads: BTreeMap<
        runmat_analysis_core::ProcedureStepId,
        runmat_analysis_fea::execution::StructuralLinearWorkload,
    >,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let Composition {
        pool,
        request,
        decoded,
        completed,
        region_resolution_objects,
        completed_inputs,
        unresolved_workloads_digest,
        policy,
        recovery,
    } = input;
    if matches!(
        &recovery,
        StructuralTransientRecoveryAdmission::Available {
            restored: Some(_),
            ..
        }
    ) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "linear structural study cannot consume a transient recovery closure".into(),
        ));
    }
    // Planned within the task's resident envelope, min(allocation, memory_per_worker), as local
    // placement plans within min(machine, memory_per_worker).
    let worker_memory_bytes = policy.resources.memory_bytes;
    let sequence = match crate::execute_structural_sequence_remotely(
        Arc::clone(&pool),
        StructuralSequencePreparation {
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            program_revision: request.recipe.program_revision.clone(),
            artifact_access: decoded.artifact_access.clone(),
            workloads,
            mesh: Some(completed.result.artifact.clone()),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
            local_execution: decoded.local_execution,
            solve_selection:
                runmat_analysis_execution::StructuralSequenceSolveSelection::WithinCeiling {
                    machine_memory_bytes: Some(worker_memory_bytes),
                },
        },
        policy,
    )
    .await
    {
        Ok(sequence) => sequence,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) = crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::LinearSequence {
            inputs: completed_inputs,
            sequence: &sequence,
            unresolved_workloads_digest,
        },
    )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}

pub(super) async fn transient(
    input: Composition<'_, '_>,
    workload: Box<runmat_analysis_fea::execution::StructuralTransientWorkload>,
    block_plan: runmat_analysis_fea::execution::StructuralTransientBlockPlan,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let Composition {
        pool,
        request,
        decoded,
        completed,
        region_resolution_objects,
        completed_inputs,
        unresolved_workloads_digest,
        policy,
        recovery,
    } = input;
    let (recovered, recovery_publisher) = match recovery {
        StructuralTransientRecoveryAdmission::Available {
            restored,
            publisher,
        } => (restored, publisher),
        StructuralTransientRecoveryAdmission::Unavailable => {
            return Ok(RemoteExactStructuralStudyComposition::Failed(
                "structural transient execution requires a transactional recovery publisher".into(),
            ))
        }
    };
    let recovery = recovered
        .map(|restored| {
            StructuralTransientRecoveryClosure::from_objects(
                &decoded.artifact_access,
                restored.root.clone(),
                restored.objects.clone(),
            )
            .map_err(protocol)
        })
        .transpose()?;
    let preparation = runmat_analysis_execution::prepare_structural_transient_sequence(
        StructuralTransientSequencePreparationInput {
            local_execution: decoded.local_execution,
            workload: &workload,
            mesh: &completed.result.artifact,
            modal_state: None,
            block_plan,
            initial_dependencies: Vec::new(),
            artifact_access: decoded.artifact_access.clone(),
            program_revision: request.recipe.program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
        },
    )
    .map_err(protocol)?;
    let completion = match crate::execute_structural_transient_remotely(
        Arc::clone(&pool),
        preparation,
        policy,
        recovery.as_ref(),
        recovery_publisher,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) = crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralTransient {
            inputs: completed_inputs,
            completion: &completion,
            unresolved_workloads_digest,
        },
    )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}

pub(super) async fn frequency(
    input: Composition<'_, '_>,
    workload: Box<runmat_analysis_fea::execution::StructuralFrequencyWorkload>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let Composition {
        pool,
        request,
        decoded,
        completed,
        region_resolution_objects,
        completed_inputs,
        unresolved_workloads_digest,
        policy,
        recovery,
    } = input;
    if matches!(
        &recovery,
        StructuralTransientRecoveryAdmission::Available {
            restored: Some(_),
            ..
        }
    ) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "structural frequency study cannot consume a transient recovery closure".into(),
        ));
    }
    if !matches!(
        workload.solver(),
        StructuralFrequencySolverPlan::Direct { .. }
    ) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "standalone structural frequency execution requires a direct solver plan".into(),
        ));
    }
    let completion = match crate::execute_structural_frequency_remotely(
        Arc::clone(&pool),
        crate::StructuralFrequencyExecutionPreparation {
            local_execution: decoded.local_execution,
            workload: *workload,
            mesh: completed.result.artifact.clone(),
            prestress_state: None,
            modal_state: None,
            program_revision: request.recipe.program_revision.clone(),
            artifact_access: decoded.artifact_access.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            observation_limits:
                runmat_analysis_fea::results::observation::ObservationLimits::default(),
        },
        policy,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) = crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralFrequency {
            inputs: completed_inputs,
            completion: &completion,
            unresolved_workloads_digest,
        },
    )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}
