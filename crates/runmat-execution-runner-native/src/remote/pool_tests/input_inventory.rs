use super::*;
use runmat_execution::value::{ValueRef, ValueRefKind};
use std::sync::Mutex;

struct InputWorker {
    spec: WorkerSpec,
    bundle: Arc<Vec<u8>>,
    received: Arc<Mutex<Vec<ValueRef>>>,
}

#[async_trait]
impl RemoteWorkerChannel for InputWorker {
    async fn authenticated_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<super::super::AuthenticatedPhysicalRetirement>> {
        Ok(Some(super::super::AuthenticatedPhysicalRetirement::issue(
            request.clone(),
        )))
    }
    async fn acknowledge_physical_retirement(
        &self,
        _: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    fn node_identity(&self) -> &str {
        "cold-input-worker"
    }
    fn worker(&self) -> &WorkerSpec {
        &self.spec
    }
    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        Ok(self.spec.clone())
    }
    async fn install_bundle(
        &self,
        digest: Digest,
        bytes: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        Ok(bundle_receipt(digest, bytes))
    }
    async fn activate_bundle(&self, digest: Digest) -> NativeExecutionResult<RemoteBundleReceipt> {
        Ok(bundle_receipt(digest, &self.bundle))
    }
    async fn transfer_value(
        &self,
        _: ValueRef,
        _: &[u8],
    ) -> NativeExecutionResult<super::super::RemoteValueReceipt> {
        panic!("all task objects belong to declared closure")
    }
    async fn transfer_object(
        &self,
        reference: ValueRef,
        bytes: &[u8],
    ) -> NativeExecutionResult<super::super::RemoteObjectReceipt> {
        assert_eq!(Digest::sha256(bytes), reference.logical_digest);
        self.received.lock().unwrap().push(reference.clone());
        Ok(super::super::RemoteObjectReceipt {
            value_id: reference.id,
            next_offset: bytes.len() as u64,
            complete: true,
        })
    }
    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        Ok(AttemptReport::Succeeded {
            result: AttemptSuccess::new(
                &attempt.scheduling,
                vec![ValuePayload::Inline(Box::new(InlineValue::String(
                    "scoped".into(),
                )))],
                Vec::new(),
                1,
                0,
                0,
            )
            .unwrap(),
        })
    }
    async fn cancel(
        &self,
        _: &runmat_execution_runner::AttemptRequest,
        _: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        Ok(())
    }
    async fn drain(&self) -> NativeExecutionResult<()> {
        Ok(())
    }
}

#[tokio::test]
async fn cold_worker_receives_only_exact_task_input_closure_and_inventory_is_immutable() {
    let (mut program, artifact, bundle) = executable_bundle_with_input().await;
    let scope = ExecutionScopeId::derive(&[b"scoped-objects"]);
    let pool_id = PoolId::derive(&[b"scoped-objects"]);
    let pool = RemotePoolDriver::new(
        scope,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1000),
        },
        1,
        bundle.clone(),
    )
    .unwrap();
    let scope_name = scope.to_string();
    let mut references = Vec::new();
    for ordinal in 0..102_u32 {
        let bytes = ordinal.to_le_bytes().to_vec();
        let reference = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: runmat_execution::identity::ValueId::derive(&[b"scoped-input", &bytes]),
            logical_digest: Digest::sha256(&bytes),
            encoded_length: bytes.len() as u64,
            media_type: "application/octet-stream".into(),
            value_schema: "test.scoped-input.v1".into(),
            encryption_context: Digest::sha256(b"scoped-input"),
            kind: ValueRefKind::DriverObject,
            authorization_scope: scope_name.clone(),
            resident_fence: None,
        };
        pool.register_execution_object(reference.clone(), bytes)
            .unwrap();
        references.push(reference);
    }
    let root = references[0].clone();
    let chunk = references[1].clone();
    program.arguments = vec![ValuePayload::Object(Box::new(root.clone()))];
    assert!(pool
        .register_program_input_inventory(&program, vec![chunk.clone()])
        .is_err());
    let mut foreign = root.clone();
    foreign.authorization_scope = "foreign".into();
    assert!(pool
        .register_program_input_inventory(&program, vec![foreign])
        .is_err());
    let mut conflict = root.clone();
    conflict.encoded_length += 1;
    assert!(pool
        .register_program_input_inventory(&program, vec![root.clone(), conflict])
        .is_err());
    pool.register_program_input_inventory(&program, vec![root.clone(), chunk.clone()])
        .unwrap();
    assert!(pool
        .register_program_input_inventory(&program, vec![root.clone()])
        .is_err());
    let received = Arc::new(Mutex::new(Vec::new()));
    pool.add_worker(Arc::new(InputWorker {
        spec: WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: WorkerId::derive(&[b"cold"]),
            pool_id,
            resources: inventory(1000),
        },
        bundle: Arc::new(bundle),
        received: Arc::clone(&received),
    }))
    .await
    .unwrap();
    let submission = TaskSubmission {
        request: TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: TaskId::derive(&[b"scoped-task"]),
            scope_id: scope,
            pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact),
            host: crate::test_support::host_requirement(),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            callable: Callable::for_program("test", &program.callable),
            inputs: program.arguments.clone(),
            outputs: OutputContract {
                owned_work: None,
                requested_outputs: 1,
            },
            resources: request(),
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: BTreeSet::new(),
        priority: 0,
    };
    pool.submit(submission.clone(), program.clone())
        .unwrap()
        .wait()
        .await
        .unwrap();
    let actual: BTreeSet<_> = received
        .lock()
        .unwrap()
        .iter()
        .map(|value| value.id)
        .collect();
    assert_eq!(actual, [root.id, chunk.id].into_iter().collect());
    assert_eq!(received.lock().unwrap().len(), 2);
    assert!(pool
        .register_program_input_inventory(&program, references)
        .is_err());
    pool.submit(submission, program)
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(
        received.lock().unwrap().len(),
        2,
        "committed reuse does not retransmit"
    );
}
