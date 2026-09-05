//! Driver-owned composition of worker-side CAD admission and the meshing-owned exact DAG.

use std::collections::BTreeSet;
use std::sync::Arc;

use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;
use runmat_meshing_core::CacheAdmissionDecision;
use runmat_meshing_execution::{
    execute_exact_meshing_dag_async, import_exact_geometry_input, ExactMeshingDagRun,
    MeshingRunEvidenceContext,
};

use super::pool_execution::{
    execution_failure_outcome, packaged, submit_root_task, task_failure_outcome,
    RemotePoolExecutionOutcome,
};
use super::RemotePoolDriver;
use crate::{
    MeshingExecutionPolicy, NativeExecutionError, NativeExecutionResult, RemoteExactMeshingExecutor,
};

pub(crate) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let decoded = crate::decode_exact_meshing_root(&request, pool.as_ref())?;
    let execution_policy = MeshingExecutionPolicy::default();
    let (task_id, completion) = submit_root_task(
        &pool,
        &request,
        &run_id,
        b"remote-exact-admission",
        &resources,
        RetryPolicy::ExplicitlyIdempotent {
            max_attempts: execution_policy.maximum_attempts,
        },
    )?;
    let admitted = match completion.wait().await {
        Ok(success) => success,
        Err(message) => return task_failure_outcome(pool, task_id, message),
    };
    let admission_reference = match admitted.outputs.as_slice() {
        [ValuePayload::Object(reference)] if reference.kind == ValueRefKind::ResultObject => {
            reference.as_ref()
        }
        _ => {
            return Err(protocol(
                "exact geometry admission returned an invalid result root",
            ))
        }
    };
    decoded
        .artifact_access
        .validate_reference(
            admission_reference,
            ValueRefKind::ResultObject,
            crate::exact_meshing_root::ADMISSION_MEDIA_TYPE,
            crate::exact_meshing_root::ADMISSION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    if !admitted.result_objects.contains(admission_reference) {
        return Err(protocol(
            "exact geometry admission omitted its result root from the inventory",
        ));
    }
    let admission_bytes = pool
        .execution_object(admission_reference)?
        .ok_or_else(|| protocol("exact geometry admission object is unavailable"))?;
    let admission = crate::exact_meshing_root::decode_admission(&admission_bytes)?;
    if !admitted.result_objects.contains(&admission.geometry_root) {
        return Err(protocol("exact geometry admission contract is invalid"));
    }
    let inventory = ObjectInventoryLimits::default();
    let mut executor = RemoteExactMeshingExecutor::new(
        Arc::clone(&pool),
        decoded.artifact_access.clone(),
        execution_policy,
        super::object_transfer::MAX_REMOTE_OBJECT_BYTES,
    )
    .map_err(protocol)?;
    let geometry = import_exact_geometry_input(
        &executor,
        admission.document.clone(),
        &admission.geometry_root,
        decoded.artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?;
    validate_source_faces(&admission, geometry.geometry_objects())?;
    let mut meshing_request = decoded.request.meshing_request;
    if !crate::exact_geometry_admission::same_requested_tolerance(
        &meshing_request.tolerance,
        &admission.document.tolerance,
    ) {
        return Err(protocol(
            "meshing request tolerance differs from admitted exact geometry",
        ));
    }
    meshing_request.tolerance = admission.document.tolerance;
    let capability_cohort = admission.platform.capability_cohort.clone();
    let result = match execute_exact_meshing_dag_async(
        ExactMeshingDagRun {
            geometry: &geometry,
            request: meshing_request,
            artifact_access: decoded.artifact_access,
            capability_cohort: Some(capability_cohort),
            program_revision: request.recipe.program_revision.clone(),
            preferred_edges_per_partition: decoded.request.preferred_edges_per_partition,
            preferred_faces_per_partition: decoded.request.preferred_faces_per_partition,
            inventory_limits: inventory,
            evidence: MeshingRunEvidenceContext {
                platform: admission.platform,
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
        },
        &mut executor,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => return execution_failure_outcome(pool, error.to_string()),
    };
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(result.root))],
        result_objects: result.result_objects,
    };
    response.validate_against(&request).map_err(protocol)?;
    packaged(response, pool).map(RemotePoolExecutionOutcome::Completed)
}

fn validate_source_faces(
    admission: &crate::exact_meshing_root::ExactGeometryAdmission,
    geometry: &runmat_meshing_execution::PreparedExactGeometryObjects,
) -> NativeExecutionResult<()> {
    let face_ids = geometry
        .topology
        .faces
        .iter()
        .map(|face| &face.id)
        .collect::<BTreeSet<_>>();
    if admission
        .source_face_ids
        .iter()
        .enumerate()
        .any(|(index, (source_index, face_id))| {
            *source_index != index as u64 || !face_ids.contains(face_id)
        })
    {
        return Err(protocol(
            "exact geometry admission contains an invalid source-face map",
        ));
    }
    Ok(())
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
