use super::*;

use runmat_execution_runner_native::supervisor::{
    execute_and_complete_program_batch, prepare_batch_driver, BatchDriverInvocation, LocalJobState,
    LocalSupervisor, LocalSupervisorConfig, ProgramBatchSubmission, SupervisorPaths,
    MIN_RETENTION_MILLIS,
};
pub(super) use runmat_meshing_execution::MeshingKernelDispatcher as CurvePipelineKernel;
use runmat_meshing_execution::{ExactCurveJoinKernel, ExactCurveStageKernel};

pub(super) struct CurveFixture {
    pub(super) partition_host: MeshingHostWorkload,
    pub(super) partition_request: runmat_execution_artifact::ProgramExecutionRequest,
    pub(super) join_host: MeshingHostWorkload,
    pub(super) join_request: runmat_execution_artifact::ProgramExecutionRequest,
    pub(super) input: PreparedExactGeometryInput,
    pub(super) expected_partition_outputs: Vec<ValuePayload>,
    pub(super) expected_join_outputs: Vec<ValuePayload>,
}

pub(super) async fn native_conformance() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = fixture(revision(), "native-curve-meshing-run");
    let session = NativeProgramSession::new(config(directory.path(), "--curve-child")).unwrap();
    let mut store = session.object_store();
    seed_exact_input(&mut store, &fixture.input);
    seed_host_invocation(
        &mut store,
        &fixture.partition_host,
        &fixture.partition_request,
    );
    seed_host_invocation(&mut store, &fixture.join_host, &fixture.join_request);

    let partition = session
        .submit(
            fixture.partition_request.clone(),
            submission(
                &session,
                &fixture.partition_host,
                &fixture.partition_request,
            ),
        )
        .unwrap();
    let partition = wait_for_native(partition).await;
    assert_eq!(
        partition.values().expect("ordinary task result").0,
        fixture.expected_partition_outputs
    );

    let join = session
        .submit(
            fixture.join_request.clone(),
            submission(&session, &fixture.join_host, &fixture.join_request),
        )
        .unwrap();
    let join = wait_for_native(join).await;
    let (join_outputs, _) = join.values().expect("ordinary task result");
    assert_eq!(join_outputs, fixture.expected_join_outputs);
    let [ValuePayload::Object(root)] = join_outputs else {
        panic!("native curve join returned a non-object root")
    };
    let publication = import_result_publication(
        &store,
        root,
        fixture.join_host.artifact_access,
        limits().inventory,
    )
    .unwrap();
    let streams = publication.stage_objects().decoded_streams().unwrap();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].media_type, MeshingChunkMediaType::CurveMesh);
}

pub(super) async fn durable_conformance() {
    let directory = tempfile::tempdir().unwrap();
    let mut execution = config(directory.path(), "--curve-child");
    execution.max_object_bytes = limits().inventory.max_object_bytes;
    let configuration = LocalSupervisorConfig::new(
        execution,
        SupervisorPaths::new(directory.path().join("durable-state")).unwrap(),
    )
    .unwrap();
    let fixture = fixture(revision(), "durable-curve-meshing-run");
    let supervisor = LocalSupervisor::open(configuration.clone()).unwrap();
    let mut store = supervisor.object_store().unwrap();
    seed_exact_input(&mut store, &fixture.input);
    seed_host_invocation(
        &mut store,
        &fixture.partition_host,
        &fixture.partition_request,
    );
    seed_host_invocation(&mut store, &fixture.join_host, &fixture.join_request);

    let context = supervisor
        .execution_context(Some("curve-partition"))
        .await
        .unwrap();
    let partition_task = submission_for(
        context.scope_id,
        context.pool_id,
        &fixture.partition_host,
        &fixture.partition_request,
    );
    let partition = supervisor
        .submit_program(ProgramBatchSubmission::from_request(
            fixture.partition_request.clone(),
            partition_task.clone(),
            Some("curve-partition".into()),
            MIN_RETENTION_MILLIS,
        ))
        .await
        .unwrap()
        .0;
    let partition = wait_for_durable(&supervisor, partition.handle.id, &partition_task).await;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = partition.response.unwrap()
    else {
        panic!("durable curve partition did not externalize its result")
    };
    assert_eq!(outputs, fixture.expected_partition_outputs);
    assert!(result_objects.iter().all(|reference| store
        .read_verified(reference.logical_digest)
        .unwrap()
        .is_some()));

    drop(supervisor);
    let supervisor = LocalSupervisor::open(configuration).unwrap();
    let recovered = supervisor
        .attach(partition.record.handle.id, 0, 0)
        .await
        .unwrap();
    assert_eq!(recovered.record.state, LocalJobState::Succeeded);
    assert!(matches!(
        recovered.response,
        Some(ProgramExecutionResponse::ExternalizedSuccess { .. })
    ));

    let context = supervisor
        .execution_context(Some("curve-join"))
        .await
        .unwrap();
    let join_task = submission_for(
        context.scope_id,
        context.pool_id,
        &fixture.join_host,
        &fixture.join_request,
    );
    let join = supervisor
        .submit_program(ProgramBatchSubmission::from_request(
            fixture.join_request.clone(),
            join_task.clone(),
            Some("curve-join".into()),
            MIN_RETENTION_MILLIS,
        ))
        .await
        .unwrap()
        .0;
    let join = wait_for_durable(&supervisor, join.handle.id, &join_task).await;
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = join.response.unwrap()
    else {
        panic!("durable curve join did not externalize its result")
    };
    assert_eq!(outputs, fixture.expected_join_outputs);
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        panic!("durable curve join returned a non-object root")
    };
    let publication = import_result_publication(
        &store,
        root,
        fixture.join_host.artifact_access,
        limits().inventory,
    )
    .unwrap();
    assert_eq!(
        publication.stage_objects().decoded_streams().unwrap()[0].media_type,
        MeshingChunkMediaType::CurveMesh
    );
}

pub(super) async fn run_durable_driver() {
    let invocation = prepare_batch_driver().unwrap();
    let BatchDriverInvocation::Program {
        attempt,
        submission,
        lease,
        ..
    } = invocation
    else {
        panic!("meshing conformance durable driver received a script")
    };
    execute_and_complete_program_batch(*attempt, *submission, *lease)
        .await
        .unwrap();
}

async fn wait_for_durable(
    supervisor: &std::sync::Arc<LocalSupervisor>,
    job_id: runmat_execution::JobId,
    original_task: &runmat_execution_runner::TaskSubmission,
) -> runmat_execution_runner_native::supervisor::JobAttachment {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let attachment = supervisor.attach(job_id, 0, 0).await.unwrap();
            if attachment.record.state.is_terminal() {
                assert_eq!(attachment.record.state, LocalJobState::Succeeded);
                let resources = attachment
                    .resources
                    .as_ref()
                    .expect("durable meshing must retain generic resource accounting");
                resources.validate().unwrap();
                assert_eq!(
                    resources.resources.allocation,
                    original_task.request.resources
                );
                let history = attachment
                    .execution
                    .as_ref()
                    .expect("durable meshing must retain original admitted history");
                history.validate().unwrap();
                assert_eq!(&history.task.submission, original_task);
                break attachment;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("durable curve stage timed out")
}

pub(super) async fn remote_conformance(
    pool: &Arc<RemotePoolDriver>,
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
    fixture: &CurveFixture,
) {
    for (object, reference) in fixture
        .input
        .geometry_objects()
        .objects
        .iter()
        .zip(fixture.input.input_objects())
    {
        pool.register_execution_object(reference.clone(), object.bytes.clone())
            .unwrap();
    }
    register_host_invocation(pool, &fixture.partition_host, &fixture.partition_request);
    register_host_invocation(pool, &fixture.join_host, &fixture.join_request);
    let partition = pool
        .submit(
            submission_for(
                scope_id,
                pool_id,
                &fixture.partition_host,
                &fixture.partition_request,
            ),
            fixture.partition_request.clone(),
        )
        .unwrap();
    let partition = tokio::time::timeout(Duration::from_secs(10), partition.wait())
        .await
        .expect("remote curve partition timeout")
        .unwrap();
    assert_eq!(
        partition.values().expect("ordinary task result").0,
        fixture.expected_partition_outputs
    );

    let join = pool
        .submit(
            submission_for(scope_id, pool_id, &fixture.join_host, &fixture.join_request),
            fixture.join_request.clone(),
        )
        .unwrap();
    let join = tokio::time::timeout(Duration::from_secs(10), join.wait())
        .await
        .expect("remote curve join timeout")
        .unwrap();
    let (join_outputs, join_result_objects) = join.values().expect("ordinary task result");
    assert_eq!(join_outputs, fixture.expected_join_outputs);
    let [ValuePayload::Object(root)] = join_outputs else {
        panic!("remote curve join returned a non-object root")
    };
    let mut store = TestStore::default();
    for reference in join_result_objects {
        store.0.insert(
            reference.logical_digest,
            pool.execution_object(reference)
                .unwrap()
                .expect("verified remote curve result object")
                .to_vec(),
        );
    }
    let publication = import_result_publication(
        &store,
        root,
        fixture.join_host.artifact_access.clone(),
        limits().inventory,
    )
    .unwrap();
    assert_eq!(
        publication.stage_objects().decoded_streams().unwrap()[0].media_type,
        MeshingChunkMediaType::CurveMesh
    );
}

pub(super) fn fixture(revision: ProgramRevision, authorization_scope: &str) -> CurveFixture {
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: Digest::sha256(b"native-meshing-encryption-context"),
    };
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_circle();
    let geometry_objects = prepare_exact_geometry_objects(
        document,
        topology,
        evaluators,
        None,
        None,
        limits().inventory,
    )
    .unwrap();
    let document = geometry_objects.document.clone();
    let input =
        prepare_exact_geometry_input(geometry_objects, access.clone(), limits().inventory).unwrap();
    let mut resolved_request = request();
    resolved_request.tolerance = document.tolerance;
    resolved_request.quality.curve.minimum_metric_edge_length = 1.0e-6;
    resolved_request.limits.maximum_nodes = 10_000;
    resolved_request.limits.maximum_elements = 10_000;
    let exact_input = MeshingInputRef {
        kind: MeshingInputKind::ExactGeometry,
        digest: StableDigest::from_bytes(*input.root_input().logical_digest.bytes()),
    };
    let partition =
        runmat_meshing_curve::curve_partition_descriptors(&input.geometry_objects().topology, 64)
            .unwrap()
            .remove(0);
    let partition_host = host(
        resolved_request.clone(),
        document.clone(),
        access.clone(),
        vec![exact_input.clone()],
        partition,
    );
    let partition_request = partition_host
        .prepare_program_request(revision.clone(), std::slice::from_ref(input.root_input()))
        .unwrap()
        .into_request();

    let mut serial_store = TestStore::default();
    seed_exact_input(&mut serial_store, &input);
    seed_host_invocation(&mut serial_store, &partition_host, &partition_request);
    let partition_response = execute_meshing_program_request(
        &partition_request,
        &mut serial_store,
        &ExactCurveStageKernel::default(),
        &NeverCancelled,
        &mut NoopMeshingProgress,
        limits(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs: expected_partition_outputs,
        ..
    } = partition_response
    else {
        panic!("serial curve partition reference failed: {partition_response:?}")
    };
    let [ValuePayload::Object(partition_root)] = expected_partition_outputs.as_slice() else {
        panic!("serial curve partition returned a non-object root")
    };
    let partition_input = MeshingInputRef {
        kind: MeshingInputKind::StageArtifact,
        digest: StableDigest::from_bytes(*partition_root.logical_digest.bytes()),
    };
    let join_host = host(
        resolved_request,
        document,
        access,
        vec![exact_input, partition_input],
        MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::DeterministicJoin,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
    );
    let join_request = join_host
        .prepare_program_request(
            revision,
            &[input.root_input().clone(), partition_root.as_ref().clone()],
        )
        .unwrap()
        .into_request();
    seed_host_invocation(&mut serial_store, &join_host, &join_request);
    let join_response = execute_meshing_program_request(
        &join_request,
        &mut serial_store,
        &ExactCurveJoinKernel::default(),
        &NeverCancelled,
        &mut NoopMeshingProgress,
        limits(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs: expected_join_outputs,
        ..
    } = join_response
    else {
        panic!("serial curve join reference failed: {join_response:?}")
    };
    CurveFixture {
        partition_host,
        partition_request,
        join_host,
        join_request,
        input,
        expected_partition_outputs,
        expected_join_outputs,
    }
}

fn host(
    resolved_request: MeshingRequest,
    document: runmat_geometry_core::GeometryDocument,
    access: ArtifactAccess,
    inputs: Vec<MeshingInputRef>,
    partition: MeshingPartitionDescriptor,
) -> MeshingHostWorkload {
    let identity = MeshingStageIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::CurveMesh,
        geometry: GeometryRevisionRef {
            source_unit_evidence:
                runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
                    runmat_geometry_core::LengthUnit::Meter,
                    "canonical solver fixture",
                ),
            source_digest: StableDigest::from_bytes(*document.source.content_digest.bytes()),
            geometry_revision: document.revision.revision,
            persistent_mapping_version: document.revision.persistent_mapping_version,
        },
        resolved_request_digest: resolved_request.canonical_digest().unwrap(),
        tolerance_policy_digest: resolved_request.tolerance.canonical_digest().unwrap(),
        metric_policy_digest: resolved_request.metric.canonical_digest().unwrap(),
        algorithm_set_digest: resolved_request.algorithms.canonical_digest().unwrap(),
        deterministic_seed: resolved_request.deterministic_seed,
        prerequisites: inputs.clone(),
        capability_cohort: Some("native-cohort-v1".into()),
    };
    let workload = MeshingWorkloadRequest {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageKind::CurveMesh,
        stage_identity_digest: identity.canonical_digest().unwrap(),
        partition,
        inputs,
        required_capabilities: vec![
            MeshingCapabilityRequirement::HostWorkload {
                abi: "host-v2".into(),
            },
            MeshingCapabilityRequirement::ExactCadKernel {
                abi: "occt-v1".into(),
            },
            MeshingCapabilityRequirement::MeshingAlgorithm {
                version: "curve/v2".into(),
            },
            MeshingCapabilityRequirement::ElementOrder {
                order: ElementOrder::Tet4,
            },
            MeshingCapabilityRequirement::DeterministicPlatformCohort {
                cohort: "native-cohort-v1".into(),
            },
        ],
    };
    MeshingHostWorkload::new(
        workload,
        identity,
        resolved_request,
        runmat_meshing_core::MeshingExecutionCeilings {
            maximum_search_work: 1_000_000,
            maximum_iterations: 1_000_000,
            ..super::fixture_ceilings()
        },
        access,
        Some(document),
    )
    .unwrap()
}

fn seed_exact_input(store: &mut impl CacheExport, input: &PreparedExactGeometryInput) {
    for object in &input.geometry_objects().objects {
        store.write_verified(object).unwrap();
    }
}

async fn wait_for_native(
    task: runmat_execution_runner_native::NativeProgramTask,
) -> runmat_execution_runner::AttemptSuccess {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(result) = task.try_result() {
                break result.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}
