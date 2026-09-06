//! Driver-owned composition of worker-side CAD admission and the meshing-owned exact DAG.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::CacheAdmissionDecision;
use runmat_meshing_execution::{
    execute_exact_meshing_dag_async, import_exact_geometry_input, prepare_exact_geometry_result,
    ExactMeshingDagRun, ExactMeshingDagRunResult, MeshingRunEvidenceContext,
};

use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::{
    execution_failure_outcome, packaged, submit_root_task, task_failure_outcome,
};
use super::RemotePoolDriver;
use crate::{
    MeshingExecutionPolicy, NativeExecutionError, NativeExecutionResult, RemoteExactMeshingExecutor,
};

/// Exact meshing output retained in driver-owned form for composition into a larger root.
pub(super) struct CompletedRemoteExactMeshing {
    pub(super) source_face_ids: BTreeMap<u64, PersistentEntityId>,
    pub(super) result: ExactMeshingDagRunResult,
}

/// Outcome of exact meshing before the caller chooses its terminal response contract.
pub(super) enum RemoteExactMeshingComposition {
    Completed(Box<CompletedRemoteExactMeshing>),
    Terminal(RemotePoolExecutionOutcome),
}

pub(crate) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_for_composition(Arc::clone(&pool), &request, &run_id, &resources).await? {
        RemoteExactMeshingComposition::Completed(completed) => {
            let CompletedRemoteExactMeshing {
                source_face_ids,
                result,
            } = *completed;
            drop(source_face_ids);
            package_completed(pool, &request, result)
        }
        RemoteExactMeshingComposition::Terminal(outcome) => Ok(outcome),
    }
}

pub(super) async fn execute_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
) -> NativeExecutionResult<RemoteExactMeshingComposition> {
    let decoded = crate::decode_exact_meshing_root(request, pool.as_ref())?;
    let execution_policy = MeshingExecutionPolicy {
        deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(resources)?),
        ..MeshingExecutionPolicy::default()
    };
    let (task_id, completion) = submit_root_task(
        &pool,
        request,
        run_id,
        b"remote-exact-admission",
        resources,
        BTreeSet::from([runmat_execution::resource::Capability::Custom(
            runmat_meshing_execution::EXACT_CAD_ADMISSION_CAPABILITY.into(),
        )]),
        RetryPolicy::ExplicitlyIdempotent {
            max_attempts: execution_policy.maximum_attempts,
        },
    )?;
    let admitted = match completion.wait().await {
        Ok(success) => success,
        Err(message) => {
            return task_failure_outcome(pool, task_id, message)
                .map(RemoteExactMeshingComposition::Terminal)
        }
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
    let mut expected_admission_inventory = prepare_exact_geometry_result(
        geometry.geometry_objects().clone(),
        decoded.artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?
    .input_objects()
    .to_vec();
    expected_admission_inventory.push(admission_reference.clone());
    validate_reference_closure(&admitted.result_objects, &expected_admission_inventory)?;
    validate_source_faces(&admission, geometry.geometry_objects())?;
    let source_face_ids = admission.source_face_ids.clone();
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
        Err(error) => {
            return execution_failure_outcome(pool, error.to_string())
                .map(RemoteExactMeshingComposition::Terminal)
        }
    };
    Ok(RemoteExactMeshingComposition::Completed(Box::new(
        CompletedRemoteExactMeshing {
            source_face_ids,
            result,
        },
    )))
}

fn package_completed(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    result: ExactMeshingDagRunResult,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(result.root))],
        result_objects: result.result_objects,
    };
    response.validate_against(request).map_err(protocol)?;
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
        .map(|face| face.id.clone())
        .collect::<BTreeSet<_>>();
    validate_source_face_bijection(&admission.source_face_ids, &face_ids)
}

fn validate_source_face_bijection(
    source_face_ids: &BTreeMap<u64, PersistentEntityId>,
    topology_face_ids: &BTreeSet<PersistentEntityId>,
) -> NativeExecutionResult<()> {
    let mapped_face_ids = source_face_ids.values().cloned().collect::<BTreeSet<_>>();
    if source_face_ids.len() != topology_face_ids.len() || &mapped_face_ids != topology_face_ids {
        return Err(protocol(
            "exact geometry admission source-face map does not cover exact topology bijectively",
        ));
    }
    Ok(())
}

fn validate_reference_closure(
    actual: &[runmat_execution::value::ValueRef],
    expected: &[runmat_execution::value::ValueRef],
) -> NativeExecutionResult<()> {
    let actual = reference_inventory(actual)?;
    let expected = reference_inventory(expected)?;
    if actual != expected {
        return Err(protocol(
            "exact geometry admission inventory differs from its canonical closure",
        ));
    }
    Ok(())
}

fn reference_inventory(
    references: &[runmat_execution::value::ValueRef],
) -> NativeExecutionResult<
    BTreeMap<runmat_execution::identity::ValueId, runmat_execution::value::ValueRef>,
> {
    let mut inventory = BTreeMap::new();
    let mut digests = BTreeSet::new();
    for reference in references {
        if inventory.insert(reference.id, reference.clone()).is_some()
            || !digests.insert(reference.logical_digest)
        {
            return Err(protocol(
                "exact geometry admission inventory contains duplicate references",
            ));
        }
    }
    Ok(inventory)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
    use runmat_execution::value::ValueRef;
    use runmat_geometry_core::PersistentEntityKind;

    use super::*;

    fn reference(identity: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"admission-reference", identity]),
            logical_digest: runmat_execution::Digest::sha256(identity),
            encoded_length: identity.len() as u64,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            encryption_context: runmat_execution::Digest::sha256(b"admission-context"),
            kind: ValueRefKind::ResultObject,
            authorization_scope: "admission-test".into(),
            resident_fence: None,
        }
    }

    fn face(source: &str, occurrence: &str) -> PersistentEntityId {
        PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: source.into(),
            assembly_path: vec!["root".into(), occurrence.into()],
        }
    }

    #[test]
    fn admission_reference_closure_compares_cardinality_and_full_metadata() {
        let first = reference(b"first");
        let second = reference(b"second");
        validate_reference_closure(
            &[second.clone(), first.clone()],
            &[first.clone(), second.clone()],
        )
        .unwrap();
        assert!(validate_reference_closure(
            &[first.clone(), first.clone()],
            &[first.clone(), second.clone()]
        )
        .is_err());

        let mut substituted = second.clone();
        substituted.value_schema = "runmat.substituted".into();
        assert!(
            validate_reference_closure(&[first.clone(), substituted], &[first, second]).is_err()
        );
    }

    #[test]
    fn source_face_bijection_requires_complete_occurrence_coverage() {
        let first = face("shared", "a");
        let second = face("shared", "b");
        let topology = BTreeSet::from([first.clone(), second.clone()]);
        validate_source_face_bijection(
            &BTreeMap::from([(0, first.clone()), (1, second.clone())]),
            &topology,
        )
        .unwrap();
        assert!(
            validate_source_face_bijection(&BTreeMap::from([(0, first.clone())]), &topology)
                .is_err()
        );
        assert!(validate_source_face_bijection(
            &BTreeMap::from([(0, first.clone()), (1, first)]),
            &topology
        )
        .is_err());
        assert!(validate_source_face_bijection(
            &BTreeMap::from([(0, second), (1, face("extra", "c"))]),
            &topology
        )
        .is_err());
    }
}
