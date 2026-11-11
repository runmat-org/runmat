//! Remote composition for a typed study whose workloads are derived after exact meshing.

use std::sync::Arc;

use runmat_execution::Digest;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_meshing::{MeshedGeometryRegions, RemoteExactMeshingComposition};
use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::{execution_failure_outcome, packaged};
use super::RemotePoolDriver;
use crate::{NativeExecutionResult, StructuralTransientRecoveryPublisher};

pub(super) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: &mut super::recovery::RemoteRecoveryPublisher<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_for_composition(
        Arc::clone(&pool),
        &request,
        &run_id,
        &resources,
        recovered,
        recovery_publisher,
    )
    .await?
    {
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Completed(
            response,
        ) => packaged(response, pool).map(Ok),
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(message) => {
            execution_failure_outcome(pool, message)
        }
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Cancelled => Ok(Err(
            crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
        )),
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Indeterminate(
            message,
        ) => Ok(Err(
            crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(message),
        )),
    }
}

pub(super) async fn execute_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: &mut dyn StructuralTransientRecoveryPublisher,
) -> NativeExecutionResult<super::exact_structural_study::RemoteExactStructuralStudyComposition> {
    let decoded = crate::decode_structural_study_root(request, pool.as_ref())?;
    // A strict (`accelerators = required`) study whose placement this worker cannot honor
    // fails here, before meshing or any analysis task, instead of after a full remote run.
    if let Err(error) = decoded.admit_accelerator_placement() {
        return Ok(
            super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(
                error.to_string(),
            ),
        );
    }
    let input_materializations = decoded.input_materializations.clone();
    let artifact_access = decoded.artifact_access.clone();
    let geometry_contract =
        crate::decode_geometry_meshing_root(&decoded.geometry_meshing, pool.as_ref())?;
    let geometry = super::exact_meshing::execute_geometry_for_composition(
        Arc::clone(&pool),
        &decoded.geometry_meshing,
        run_id,
        resources,
    );
    let inputs = super::input_materialization::execute_ready(
        Arc::clone(&pool),
        &input_materializations,
        &artifact_access,
        resources,
    );
    let (geometry, input_outcomes) = tokio::join!(geometry, inputs);
    let completed = match geometry? {
        RemoteExactMeshingComposition::Completed(completed) => completed,
        RemoteExactMeshingComposition::Terminal(outcome) => {
            return Ok(super::exact_structural_study::composition(*outcome))
        }
    };
    let mut input_outcomes = input_outcomes?;
    let deferred_outcomes = super::input_materialization::execute_deferred(
        Arc::clone(&pool),
        &input_materializations,
        &completed.result.artifact,
        &artifact_access,
        resources,
    )
    .await?;
    for (identity, outcome) in deferred_outcomes {
        if input_outcomes.insert(identity, outcome).is_some() {
            return Err(crate::NativeExecutionError::Protocol(
                "ready and mesh-dependent input identities collide".into(),
            ));
        }
    }
    let regions = match &completed.regions {
        MeshedGeometryRegions::Materialized(regions) => regions,
        MeshedGeometryRegions::PersistentEntities => {
            return Ok(
                super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(
                    "typed studies require source-neutral materialized geometry".into(),
                ),
            )
        }
    };
    let Some(planning) = decoded.planning.structural() else {
        let regions = regions.clone();
        return Box::pin(super::family_study::execute(
            super::family_study::FamilyComposition {
                pool,
                request,
                meshing_request: geometry_contract.request.meshing_request.clone(),
                decoded,
                completed,
                regions,
                input_outcomes,
                resources,
            },
        ))
        .await;
    };
    let product = match runmat_analysis_fea::study::build_structural_study_product_with_fields(
        runmat_analysis_fea::study::StructuralStudyProductInput {
            study: &decoded.study,
            geometry: &completed.document,
            regions,
            materials: &decoded.materials,
            mesh: &completed.result.artifact,
            planning,
        },
        &decoded.initial_fields,
    ) {
        Ok(product) => product,
        Err(error) => {
            return Ok(
                super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(
                    error.to_string(),
                ),
            )
        }
    };
    let geometry_source_digest =
        Digest::from_bytes(*completed.result.artifact.geometry.source_digest.bytes());
    let compiled_geometry_output = completed.materialization.clone().ok_or_else(|| {
        crate::NativeExecutionError::Protocol(
            "typed structural study geometry has no materialization receipt".into(),
        )
    })?;
    let mut completed = completed;
    completed.regions = MeshedGeometryRegions::PersistentEntities;
    let execution_view = crate::DecodedExactStructuralStudyRoot {
        authored_study_digest: decoded.authored_study_digest,
        frozen_project_digest: decoded.frozen_project_digest,
        geometry_source_digest,
        workloads: product.workloads,
        analysis_accelerator_requests: decoded.analysis_accelerator_requests,
        local_execution: decoded.local_execution,
        memory_per_worker_bytes: decoded.memory_per_worker_bytes,
        input_materializations,
        exact_meshing: decoded.geometry_meshing,
        meshing_request: geometry_contract.request.meshing_request,
        artifact_access: decoded.artifact_access,
        compiled_study: Some(product.compiled),
        compiled_geometry_output: Some(compiled_geometry_output),
    };
    super::exact_structural_study::execute_with_completed_meshing(
        pool,
        request,
        execution_view,
        completed,
        &input_outcomes,
        resources,
        super::exact_structural_study::StructuralTransientRecoveryAdmission::Available {
            restored: recovered,
            publisher: recovery_publisher,
        },
    )
    .await
}

/// Terminal of one authored study root composed on an existing worker pool.
#[derive(Debug)]
pub enum RemoteStudyRootOutcome {
    Completed {
        response: runmat_execution_artifact::ProgramExecutionResponse,
        objects: Vec<runmat_execution_artifact::ProgramExecutionObject>,
    },
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

/// Compose any authored study root (structural or family) on a connected pool and return its
/// complete result closure. The driver process uses the same composition for cluster runs.
pub async fn execute_study_root_on_pool(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    scope: &str,
    resources: &AllocationResources,
) -> NativeExecutionResult<RemoteStudyRootOutcome> {
    let mut recovery = crate::RetainedStructuralTransientRecovery::default();
    let outcome = execute_for_composition(
        Arc::clone(&pool),
        request,
        scope,
        resources,
        None,
        &mut recovery,
    )
    .await?;
    Ok(match outcome {
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Completed(
            response,
        ) => {
            let mut objects = Vec::new();
            if let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
                result_objects,
                ..
            } = &response
            {
                for reference in result_objects {
                    let bytes = pool.execution_object(reference)?.ok_or_else(|| {
                        crate::NativeExecutionError::Protocol(
                            "remote result object is absent from the worker pool".into(),
                        )
                    })?;
                    objects.push(runmat_execution_artifact::ProgramExecutionObject {
                        reference: reference.clone(),
                        bytes: bytes.to_vec(),
                    });
                }
            }
            RemoteStudyRootOutcome::Completed { response, objects }
        }
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(message) => {
            RemoteStudyRootOutcome::Failed(message)
        }
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Cancelled => {
            RemoteStudyRootOutcome::Cancelled
        }
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Indeterminate(
            message,
        ) => RemoteStudyRootOutcome::Indeterminate(message),
    })
}
