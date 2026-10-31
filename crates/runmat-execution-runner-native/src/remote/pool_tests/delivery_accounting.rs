//! A real result-delivery failure must preserve valid original producer consumption only.
use super::*;

struct DeliveryWorker {
    original: ResultObjectWorker,
    foreign_accounting: bool,
}

#[async_trait]
impl RemoteWorkerChannel for DeliveryWorker {
    async fn authenticated_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<super::super::AuthenticatedPhysicalRetirement>> {
        self.original
            .authenticated_physical_retirement(request)
            .await
    }
    async fn acknowledge_physical_retirement(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        self.original.acknowledge_physical_retirement(request).await
    }
    fn node_identity(&self) -> &str {
        self.original.node_identity()
    }
    fn worker(&self) -> &WorkerSpec {
        self.original.worker()
    }
    async fn describe_worker(&self) -> NativeExecutionResult<WorkerSpec> {
        self.original.describe_worker().await
    }
    async fn install_bundle(
        &self,
        digest: Digest,
        bytes: &[u8],
    ) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.original.install_bundle(digest, bytes).await
    }
    async fn activate_bundle(&self, digest: Digest) -> NativeExecutionResult<RemoteBundleReceipt> {
        self.original.activate_bundle(digest).await
    }
    async fn transfer_value(
        &self,
        reference: runmat_execution::value::ValueRef,
        encoded: &[u8],
    ) -> NativeExecutionResult<super::super::RemoteValueReceipt> {
        self.original.transfer_value(reference, encoded).await
    }
    async fn cancel(
        &self,
        request: &runmat_execution_runner::AttemptRequest,
        reason: runmat_execution::CancellationReason,
    ) -> NativeExecutionResult<()> {
        self.original.cancel(request, reason).await
    }
    async fn drain(&self) -> NativeExecutionResult<()> {
        self.original.drain().await
    }
    async fn execute(&self, attempt: RemoteAttempt) -> NativeExecutionResult<AttemptReport> {
        let mut report = self.original.execute(attempt).await?;
        if self.foreign_accounting {
            let AttemptReport::Succeeded { result } = &mut report else {
                unreachable!()
            };
            let AttemptSuccess::Values { resources, .. } = result else {
                panic!("value accounting fixture")
            };
            resources.as_mut().unwrap().worker_id =
                WorkerId::derive(&[b"foreign-delivery-accounting"]);
        }
        Ok(report)
    }
    async fn download_object(
        &self,
        reference: runmat_execution::value::ValueRef,
    ) -> NativeExecutionResult<Vec<u8>> {
        self.original.download_object(reference).await
    }
}

#[tokio::test]
async fn failed_result_delivery_preserves_valid_original_accounting_through_cold_history() {
    for foreign in [false, true] {
        let scope = ExecutionScopeId::derive(&[b"delivery-accounting", &[u8::from(foreign)]]);
        let pool_id = PoolId::derive(&[scope.bytes()]);
        let bundle = Arc::new(executable_bundle().await.2);
        let pool = RemotePoolDriver::new_with_value_scope(
            scope,
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: inventory(1_000),
            },
            43,
            bundle.as_ref().clone(),
            "delivery-accounting",
        )
        .unwrap();
        let bytes = Arc::new(b"original result bytes".to_vec());
        let reference = object_reference("delivery-accounting", &bytes);
        pool.add_worker(Arc::new(DeliveryWorker {
            original: ResultObjectWorker {
                spec: WorkerSpec {
                    host: crate::test_support::host_inventory(),
                    id: WorkerId::derive(&[scope.bytes(), b"producer"]),
                    pool_id,
                    resources: inventory(1_000),
                },
                bundle,
                reference: reference.clone(),
                bytes,
                corrupt_download: true,
            },
            foreign_accounting: foreign,
        }))
        .await
        .unwrap();
        let (program, artifact_id) = program().await;
        let task_id = TaskId::derive(&[scope.bytes(), b"delivery-task"]);
        let completion = pool
            .submit(submission(scope, pool_id, task_id, artifact_id), program)
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), completion.wait())
                .await
                .unwrap()
                .is_err()
        );
        let evidence = pool.task_execution_evidence(task_id).unwrap();
        let original = &evidence.attempts[0];
        let terminal = original.terminal.as_ref().unwrap();
        assert_eq!(
            terminal.disposition,
            runmat_execution_runner::task::AttemptDisposition::Lost
        );
        let retirement = original.unverified_retirement.as_ref().unwrap();
        assert_eq!(
            retirement.retired_at_driver_millis,
            Some(terminal.closed_at_driver_millis)
        );
        assert!(evidence.task.committed.is_none());
        assert!(pool.execution_object(&reference).unwrap().is_none());
        if foreign {
            assert!(terminal.resources.is_none());
            assert!(evidence.resource_aggregate(1).is_err());
        } else {
            let producer = AttemptSuccess::new(
                &original.request,
                vec![ValuePayload::Object(Box::new(reference.clone()))],
                vec![reference.clone()],
                1,
                reference.encoded_length,
                0,
            )
            .unwrap();
            assert_eq!(terminal.resources.as_deref(), producer.resources());
            // Exact retirement closes the original failed delivery without
            // erasing the producer's validated resource accounting.
            assert!(evidence.resource_aggregate(1).is_ok());
        }
        let snapshot = pool.snapshot();
        let cold: runmat_execution_runner::DriverSnapshot =
            serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
        cold.validate().unwrap();
        assert_eq!(cold.task_execution_evidence(task_id).unwrap(), evidence);
    }
}
