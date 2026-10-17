use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
    HostNumericDataOwned, HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
    NumericElementType, ProviderExecutionError, ProviderExecutionResult, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderOperationFamily, ProviderResourceEstimate,
    ProviderUploadPhysicalEvidence,
};
use runmat_execution::ResourceByteBound;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum Outcome {
    Double,
    Single,
    Alias,
    Failure,
    Unsupported,
}

struct PermutationOwner {
    owner: TestAllocationOwner,
    outcome: Outcome,
    raw_uploads: AtomicUsize,
    source: Mutex<Option<GpuTensorHandle>>,
    results: Mutex<Vec<RetirementProbe>>,
}

impl AccelProvider for PermutationOwner {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned permutation fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("legacy upload must not publish a GPU child permutation")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("permutation ownership test must not download") })
    }
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        if query.operation.0 == "transfer.upload"
            && query.family == ProviderOperationFamily::Upload
            && query.inputs.len() == 1
            && query.outputs.len() == 1
            && query.outputs[0].element_type == runmat_accelerate_api::ProviderElementType::F64
            && query.outputs[0].shape == [1, 1]
        {
            ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: Some(0),
                output_bytes: Some(8),
                dispatches: Some(0),
            })
        } else {
            ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "fixture.permutation.upload",
            )
        }
    }
    fn upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderUploadPhysicalEvidence> {
        self.query_feasibility(query)
            .is_supported()
            .then_some(ProviderUploadPhysicalEvidence {
                output_requested_bytes: 8,
                host_workspace_bytes: ResourceByteBound::Known(0),
                provider_scratch_bytes: ResourceByteBound::Known(0),
                completion_queue_slots: 0,
            })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.raw_uploads.fetch_add(1, Ordering::SeqCst);
        assert_eq!(host.shape, &[1, 1]);
        assert_eq!(host.data.element_type(), NumericElementType::F64);
        let allocation = match self.outcome {
            Outcome::Double => self.owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0]),
                shape: host.shape.to_vec(),
                storage: GpuTensorStorage::Real,
            }),
            Outcome::Single => self.owner.allocate_zeroed(
                host.shape.to_vec(),
                NumericElementType::F32,
                GpuTensorStorage::Real,
            ),
            Outcome::Alias => return Ok(self.source.lock().unwrap().as_ref().unwrap().clone()),
            Outcome::Failure => {
                return Err(ProviderExecutionError::from(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "permutation upload failed",
                ))
                .with_unproven_effects())
            }
            Outcome::Unsupported => {
                return Err(ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "native Double upload unavailable before effects",
                    ),
                ))
            }
        }
        .map_err(|cause| ProviderExecutionError::from(cause).with_unproven_effects())?;
        self.results.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn like_uses_exact_owner_and_rejects_wrong_upload_without_replay() {
    let _guard = random::test_guard();
    random::reset_rng();
    let cpu_before = random::snapshot().expect("CPU stream before GPU permutation");
    random::gpu::replace(random::gpu::GpuRandomSnapshot::default())
        .expect("reset independent GPU child");
    crate::builtins::common::test_support::with_test_provider(|ambient| {
        for outcome in [
            Outcome::Double,
            Outcome::Single,
            Outcome::Alias,
            Outcome::Failure,
            Outcome::Unsupported,
        ] {
            let owner = Box::leak(Box::new(PermutationOwner {
                owner: TestAllocationOwner::new(),
                outcome,
                raw_uploads: AtomicUsize::new(0),
                source: Mutex::new(None),
                results: Mutex::new(Vec::new()),
            }));
            // SAFETY: the fixture and its device namespace live for the process.
            unsafe { runmat_accelerate_api::register_device_provider(owner) };
            let source = owner
                .owner
                .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let input = source
                .handle
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            *owner.source.lock().unwrap() = Some(input.clone());
            assert_ne!(ambient.device_id(), input.device_id());
            let result = futures::executor::block_on(gpu_child::like(&input, 1, 1));
            match outcome {
                Outcome::Double => {
                    let Value::GpuTensor(output) = result.unwrap() else {
                        panic!("valid permutation must remain resident")
                    };
                    assert_eq!(output.device_id(), input.device_id());
                    assert_eq!(
                        runmat_accelerate_api::handle_precision(&output),
                        Some(runmat_accelerate_api::ProviderPrecision::F64)
                    );
                    assert!(runmat_accelerate_api::handle_is_explicit(&output));
                    assert!(!output.shares_allocation_with(&input));
                }
                Outcome::Single => {
                    let error = result.expect_err("wrong class must be rejected");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    assert_eq!(owner.results.lock().unwrap()[0].retirements(), 1);
                }
                Outcome::Alias => {
                    let error = result.expect_err("borrowed input is not fresh output");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                }
                Outcome::Failure => {
                    let error = result.expect_err("operational failure is terminal");
                    assert!(error.message().contains("permutation upload failed"));
                }
                Outcome::Unsupported => {
                    let error = result.expect_err("explicit input cannot return host output");
                    assert!(error.message().contains("native upload failed"));
                }
            }
            assert_eq!(owner.raw_uploads.load(Ordering::SeqCst), 1);
            assert_eq!(source.probe.retirements(), 0);
        }
    });
    let cpu_after = random::snapshot().expect("CPU stream after GPU permutation");
    assert_eq!(cpu_before.state, cpu_after.state);
    assert_eq!(
        random::gpu::snapshot()
            .expect("GPU stream after attempts")
            .position,
        5
    );
}
