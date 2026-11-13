//! Consumer execution isolated from sequence admission to bound native poll frames.
use super::*;
pub(super) async fn modal_frequency(
    input: Composition<'_, '_>,
    authored: StructuralStudyWorkloads,
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
        recovery: _,
    } = input;
    let completion = match Box::pin(
        crate::execute_structural_prestressed_modal_frequency_study_remotely(
            Arc::clone(&pool),
            runmat_analysis_execution::StructuralPrestressedFrequencyStudyPreparation {
                authored_study_digest: decoded.authored_study_digest,
                frozen_project_digest: decoded.frozen_project_digest,
                workloads: authored,
                mesh: completed.result.artifact.clone(),
                artifact_access: decoded.artifact_access.clone(),
                program_revision: request.recipe.program_revision.clone(),
                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                inventory_limits: ObjectInventoryLimits::default(),
                required_capabilities: Default::default(),
                accelerator_requests: decoded.analysis_accelerator_requests.clone(),
                local_execution: decoded.local_execution,
            },
            policy,
        ),
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
    crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralPrestressedModalFrequency {
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
    authored: StructuralStudyWorkloads,
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
        recovery: _,
    } = input;
    let completion = match Box::pin(
        crate::execute_structural_prestressed_frequency_study_remotely(
            Arc::clone(&pool),
            runmat_analysis_execution::StructuralPrestressedFrequencyStudyPreparation {
                authored_study_digest: decoded.authored_study_digest,
                frozen_project_digest: decoded.frozen_project_digest,
                workloads: authored,
                mesh: completed.result.artifact.clone(),
                artifact_access: decoded.artifact_access.clone(),
                program_revision: request.recipe.program_revision.clone(),
                chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
                inventory_limits: ObjectInventoryLimits::default(),
                required_capabilities: Default::default(),
                accelerator_requests: decoded.analysis_accelerator_requests.clone(),
                local_execution: decoded.local_execution,
            },
            policy,
        ),
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
    crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralPrestressedFrequency {
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

pub(super) async fn modal(
    input: Composition<'_, '_>,
    authored: StructuralStudyWorkloads,
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
        recovery: _,
    } = input;
    let completion = match Box::pin(crate::execute_structural_prestressed_modal_study_remotely(
        Arc::clone(&pool),
        runmat_analysis_execution::StructuralPrestressedModalStudyPreparation {
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            workloads: authored,
            mesh: completed.result.artifact.clone(),
            artifact_access: decoded.artifact_access.clone(),
            program_revision: request.recipe.program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
            local_execution: decoded.local_execution,
        },
        policy,
    ))
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
    crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralPrestressedModal {
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

pub(super) async fn buckling(
    input: Composition<'_, '_>,
    authored: StructuralStudyWorkloads,
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
        recovery: _,
    } = input;
    let completion = match crate::execute_structural_buckling_study_remotely(
        Arc::clone(&pool),
        runmat_analysis_execution::StructuralBucklingStudyPreparation {
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            workloads: authored,
            mesh: completed.result.artifact.clone(),
            artifact_access: decoded.artifact_access.clone(),
            program_revision: request.recipe.program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
            local_execution: decoded.local_execution,
        },
        policy,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
    crate::exact_structural_study_root::package_exact_structural_study_result(
    request,
    decoded,
    &completed.result,
    region_resolution_objects,
    crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralBuckling {
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

pub(super) async fn spectrum(
    input: Composition<'_, '_>,
    authored: StructuralStudyWorkloads,
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
        recovery: _,
    } = input;
    let completion = match crate::execute_structural_modal_response_spectrum_remotely(
        Arc::clone(&pool),
        runmat_analysis_execution::StructuralModalResponseSpectrumStudyPreparation {
            local_execution: decoded.local_execution,
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            workloads: authored,
            mesh: completed.result.artifact.clone(),
            artifact_access: decoded.artifact_access.clone(),
            program_revision: request.recipe.program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
        },
        policy,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
    crate::exact_structural_study_root::package_exact_structural_study_result(
    request,
    decoded,
    &completed.result,
    region_resolution_objects,
    crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralModalResponseSpectrum {
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

pub(super) async fn random(
    input: Composition<'_, '_>,
    authored: StructuralStudyWorkloads,
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
        recovery: _,
    } = input;
    let completion = match crate::execute_structural_random_response_remotely(
        Arc::clone(&pool),
        runmat_analysis_execution::StructuralRandomResponseStudyPreparation {
            local_execution: decoded.local_execution,
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            workloads: authored,
            mesh: completed.result.artifact.clone(),
            artifact_access: decoded.artifact_access.clone(),
            program_revision: request.recipe.program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
        },
        policy,
    )
    .await
    {
        Ok(completion) => completion,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
    crate::exact_structural_study_root::package_exact_structural_study_result(
        request,
        decoded,
        &completed.result,
        region_resolution_objects,
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralRandomResponse {
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

pub(super) async fn transient(
    input: Composition<'_, '_>,
    authored: StructuralStudyWorkloads,
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
                "modal-transient execution requires a transactional recovery publisher".into(),
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
    let completion = match crate::execute_structural_modal_transient_remotely(
        Arc::clone(&pool),
        runmat_analysis_execution::StructuralModalTransientStudyPreparation {
            local_execution: decoded.local_execution,
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            workloads: authored,
            mesh: completed.result.artifact.clone(),
            artifact_access: decoded.artifact_access.clone(),
            program_revision: request.recipe.program_revision.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
            accelerator_requests: decoded.analysis_accelerator_requests.clone(),
        },
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
        crate::exact_structural_study_root::ExactStructuralAnalysisPublication::StructuralModalTransient {
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
