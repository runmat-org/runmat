//! Driver-owned composition of worker-side CAD admission and the meshing-owned exact DAG.

mod policy;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::TaskId;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;
use runmat_geometry_core::{GeometryDocument, PersistentEntityId};
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
#[derive(Clone)]
pub(super) struct CompletedRemoteExactMeshing {
    pub(super) document: GeometryDocument,
    pub(super) regions: MeshedGeometryRegions,
    pub(super) materialization: Option<runmat_analysis_execution::GeometryMaterializationOutput>,
    pub(super) result: ExactMeshingDagRunResult,
}

#[derive(Clone)]
pub(super) enum MeshedGeometryRegions {
    /// Meshed directly from the exact source; the mesh provenance carries canonical identities.
    PersistentEntities,
    Materialized(runmat_geometry_ops::ResolvedGeometryRegions),
}

/// Outcome of exact meshing before the caller chooses its terminal response contract.
pub(super) enum RemoteExactMeshingComposition {
    Completed(Box<CompletedRemoteExactMeshing>),
    // Ordinary driver transport control, not managed Value backing.
    Terminal(Box<RemotePoolExecutionOutcome>),
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
                document,
                regions,
                materialization,
                result,
            } = *completed;
            drop(document);
            drop(regions);
            drop(materialization);
            package_completed(pool, &request, result)
        }
        RemoteExactMeshingComposition::Terminal(outcome) => Ok(*outcome),
    }
}

pub(crate) async fn execute_geometry_root(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_geometry_for_composition(Arc::clone(&pool), &request, &run_id, &resources).await?
    {
        RemoteExactMeshingComposition::Completed(completed) => {
            package_completed(pool, &request, completed.result)
        }
        RemoteExactMeshingComposition::Terminal(outcome) => Ok(*outcome),
    }
}

pub(super) async fn execute_geometry_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
) -> NativeExecutionResult<RemoteExactMeshingComposition> {
    let root_id = TaskId::derive(&[run_id.as_bytes(), b"remote-geometry-materialization"]);
    let mut wave = super::RemoteObservedWave::begin(Arc::clone(&pool), &[root_id]).await?;
    let decoded = crate::decode_geometry_meshing_root(request, pool.as_ref())?;
    let (geometry_request, _) = runmat_analysis_execution::decode_geometry_materialization_program(
        &decoded.geometry,
        pool.as_ref(),
        &decoded.artifact_access,
    )
    .map_err(protocol)?;
    let execution_policy = policy::execution_policy(resources)?;
    let (task_id, completion) = submit_root_task(
        &pool,
        &decoded.geometry,
        root_id,
        resources,
        BTreeSet::from([runmat_analysis_execution::geometry_provider_capability(
            &geometry_request.provider,
        )]),
        RetryPolicy::ExplicitlyIdempotent {
            max_attempts: execution_policy.maximum_attempts,
        },
        &mut wave,
    )?;
    wave.checkpoint().await?;
    let materialized = match completion.wait().await {
        Ok(success) => success,
        Err(message) => {
            return task_failure_outcome(pool, task_id, message)
                .map(|outcome| RemoteExactMeshingComposition::Terminal(Box::new(outcome)))
        }
    };
    wave.close().await?;
    let output_reference = match materialized
        .values()
        .map_or(&[][..], |(outputs, _)| outputs)
    {
        [ValuePayload::Object(reference)] if reference.kind == ValueRefKind::ResultObject => {
            reference.as_ref()
        }
        _ => {
            return Err(protocol(
                "geometry materialization returned an invalid result root",
            ))
        }
    };
    if !materialized.result_objects().contains(output_reference) {
        return Err(protocol(
            "geometry materialization omitted its output root from the inventory",
        ));
    }
    let output = runmat_analysis_execution::import_geometry_materialization_output(
        pool.as_ref(),
        output_reference,
        &decoded.artifact_access,
    )
    .map_err(protocol)?;
    output
        .validate_against(&geometry_request)
        .map_err(protocol)?;
    if !materialized
        .result_objects()
        .contains(&output.geometry_root)
    {
        return Err(protocol(
            "geometry materialization omitted its exact geometry root",
        ));
    }
    let inventory = ObjectInventoryLimits::default();
    let mut executor = RemoteExactMeshingExecutor::new(
        Arc::clone(&pool),
        decoded.artifact_access.clone(),
        execution_policy,
        super::object_transfer::MAX_REMOTE_OBJECT_BYTES,
        inventory,
    )
    .map_err(protocol)?;
    let geometry = import_exact_geometry_input(
        &executor,
        output.document.clone(),
        &output.geometry_root,
        decoded.artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?;
    let mut expected_inventory = prepare_exact_geometry_result(
        geometry.geometry_objects().clone(),
        decoded.artifact_access.clone(),
        inventory,
    )
    .map_err(protocol)?
    .input_objects()
    .to_vec();
    expected_inventory.push(output_reference.clone());
    validate_reference_closure(materialized.result_objects(), &expected_inventory)?;
    let mut meshing_request = decoded.request.meshing_request;
    if !crate::exact_geometry_admission::same_requested_tolerance(
        &meshing_request.tolerance,
        &output.document.tolerance,
    ) {
        return Err(protocol(
            "meshing request tolerance differs from materialized exact geometry",
        ));
    }
    meshing_request.tolerance = output.document.tolerance;
    let materialization = output.clone();
    let platform = output.platform;
    platform.validate().map_err(protocol)?;
    let result = match execute_exact_meshing_dag_async(
        ExactMeshingDagRun {
            geometry: &geometry,
            ceilings: policy::execution_ceilings(resources, &meshing_request.limits)?,
            request: meshing_request,
            artifact_access: decoded.artifact_access,
            capability_cohort: Some(platform.capability_cohort.clone()),
            program_revision: request.recipe.program_revision.clone(),
            preferred_edges_per_partition: decoded.request.preferred_edges_per_partition,
            preferred_faces_per_partition: decoded.request.preferred_faces_per_partition,
            inventory_limits: inventory,
            evidence: MeshingRunEvidenceContext {
                platform,
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            named_regions: runmat_meshing_core::NamedRegionEntities::from_resolved_regions(
                &materialization.regions,
            ),
        },
        &mut executor,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return execution_failure_outcome(pool, error.to_string())
                .map(|outcome| RemoteExactMeshingComposition::Terminal(Box::new(outcome)))
        }
    };
    Ok(RemoteExactMeshingComposition::Completed(Box::new(
        CompletedRemoteExactMeshing {
            document: output.document,
            regions: MeshedGeometryRegions::Materialized(output.regions),
            materialization: Some(materialization),
            result,
        },
    )))
}

pub(super) async fn execute_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
) -> NativeExecutionResult<RemoteExactMeshingComposition> {
    let root_id = TaskId::derive(&[run_id.as_bytes(), b"remote-exact-admission"]);
    let mut wave = super::RemoteObservedWave::begin(Arc::clone(&pool), &[root_id]).await?;
    let decoded = crate::decode_exact_meshing_root(request, pool.as_ref())?;
    let execution_policy = policy::execution_policy(resources)?;
    let (task_id, completion) = submit_root_task(
        &pool,
        request,
        root_id,
        resources,
        BTreeSet::from([runmat_execution::resource::Capability::Custom(
            runmat_meshing_execution::EXACT_CAD_ADMISSION_CAPABILITY.into(),
        )]),
        RetryPolicy::ExplicitlyIdempotent {
            max_attempts: execution_policy.maximum_attempts,
        },
        &mut wave,
    )?;
    wave.checkpoint().await?;
    let admitted = match completion.wait().await {
        Ok(success) => success,
        Err(message) => {
            return task_failure_outcome(pool, task_id, message)
                .map(|outcome| RemoteExactMeshingComposition::Terminal(Box::new(outcome)))
        }
    };
    wave.close().await?;
    let admission_reference = match admitted.values().map_or(&[][..], |(outputs, _)| outputs) {
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
    if !admitted.result_objects().contains(admission_reference) {
        return Err(protocol(
            "exact geometry admission omitted its result root from the inventory",
        ));
    }
    let admission_bytes = pool
        .execution_object(admission_reference)?
        .ok_or_else(|| protocol("exact geometry admission object is unavailable"))?;
    let admission = crate::exact_meshing_root::decode_admission(&admission_bytes)?;
    if !admitted.result_objects().contains(&admission.geometry_root) {
        return Err(protocol("exact geometry admission contract is invalid"));
    }
    let inventory = ObjectInventoryLimits::default();
    let mut executor = RemoteExactMeshingExecutor::new(
        Arc::clone(&pool),
        decoded.artifact_access.clone(),
        execution_policy,
        super::object_transfer::MAX_REMOTE_OBJECT_BYTES,
        inventory,
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
    validate_reference_closure(admitted.result_objects(), &expected_admission_inventory)?;
    validate_source_entities(&admission, geometry.geometry_objects())?;
    let source_region_ids = admission.source_region_ids.clone();
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
            ceilings: policy::execution_ceilings(resources, &meshing_request.limits)?,
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
            named_regions: Default::default(),
        },
        &mut executor,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return execution_failure_outcome(pool, error.to_string())
                .map(|outcome| RemoteExactMeshingComposition::Terminal(Box::new(outcome)))
        }
    };
    let meshed_region_ids = result
        .artifact
        .topology
        .regions
        .iter()
        .map(|region| region.region_id.clone())
        .collect::<BTreeSet<_>>();
    validate_source_entity_bijection("region", &source_region_ids, &meshed_region_ids)?;
    Ok(RemoteExactMeshingComposition::Completed(Box::new(
        CompletedRemoteExactMeshing {
            document: admission.document,
            regions: MeshedGeometryRegions::PersistentEntities,
            materialization: None,
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
    packaged(response, pool).map(Ok)
}

fn validate_source_entities(
    admission: &crate::exact_meshing_root::ExactGeometryAdmission,
    geometry: &runmat_meshing_execution::PreparedExactGeometryObjects,
) -> NativeExecutionResult<()> {
    let edge_ids = geometry
        .topology
        .edges
        .iter()
        .map(|edge| edge.id.clone())
        .collect::<BTreeSet<_>>();
    validate_source_entity_bijection("edge", &admission.source_edge_ids, &edge_ids)?;
    let face_ids = geometry
        .topology
        .faces
        .iter()
        .map(|face| face.id.clone())
        .collect::<BTreeSet<_>>();
    validate_source_entity_bijection("face", &admission.source_face_ids, &face_ids)?;
    let region_ids = geometry
        .topology
        .regions
        .iter()
        .map(|region| region.id.clone())
        .collect::<BTreeSet<_>>();
    validate_source_entity_bijection("region", &admission.source_region_ids, &region_ids)
}

fn validate_source_entity_bijection(
    label: &str,
    source_ids: &BTreeMap<u64, PersistentEntityId>,
    topology_ids: &BTreeSet<PersistentEntityId>,
) -> NativeExecutionResult<()> {
    let mapped_ids = source_ids.values().cloned().collect::<BTreeSet<_>>();
    if source_ids.len() != topology_ids.len() || &mapped_ids != topology_ids {
        return Err(protocol(format!(
            "exact geometry admission source-{label} map does not cover exact topology bijectively"
        )));
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
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
    use runmat_execution::value::ValueRef;
    use runmat_geometry_core::PersistentEntityKind;

    use super::*;

    fn reference(identity: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
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
        PersistentEntityId::new(
            PersistentEntityKind::Face,
            source.into(),
            vec!["root".into(), occurrence.into()],
        )
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
        validate_source_entity_bijection(
            "face",
            &BTreeMap::from([(0, first.clone()), (1, second.clone())]),
            &topology,
        )
        .unwrap();
        assert!(validate_source_entity_bijection(
            "face",
            &BTreeMap::from([(0, first.clone())]),
            &topology
        )
        .is_err());
        assert!(validate_source_entity_bijection(
            "face",
            &BTreeMap::from([(0, first.clone()), (1, first)]),
            &topology
        )
        .is_err());
        assert!(validate_source_entity_bijection(
            "face",
            &BTreeMap::from([(0, second), (1, face("extra", "c"))]),
            &topology
        )
        .is_err());
    }
}
