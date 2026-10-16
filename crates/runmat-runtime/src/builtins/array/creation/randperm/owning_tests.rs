use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuHandleProvenance, HostNumericDataOwned,
    HostNumericTensorOwned, HostTensorView, NumericElementType, ProviderExecutionError,
    ProviderExecutionResult,
};
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
    uploads: AtomicUsize,
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
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Err(runmat_accelerate_api::unsupported_provider_operation(
            "Double upload unavailable",
        ))
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("permutation ownership test must not download") })
    }

    fn random_permutation_like(
        &self,
        prototype: &GpuTensorHandle,
        _: usize,
        _: usize,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let allocation = match self.outcome {
            Outcome::Double => self.owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            }),
            Outcome::Single => self.owner.allocate_zeroed(
                vec![1, 1],
                NumericElementType::F32,
                GpuTensorStorage::Real,
            ),
            Outcome::Alias => return Ok(prototype.clone()),
            Outcome::Failure => {
                return Err(ProviderExecutionError::from(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "permutation execution failed",
                ))
                .with_unproven_effects())
            }
            Outcome::Unsupported => {
                return Err(ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "permutation kernel unavailable before effects",
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
fn like_uses_exact_owner_and_rejects_wrong_class_without_replay() {
    let _guard = random::test_guard();
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
                uploads: AtomicUsize::new(0),
                results: Mutex::new(Vec::new()),
            }));
            // SAFETY: leaked fixture and its minted device namespace live for
            // the process lifetime; ambient selection remains a different owner.
            unsafe { runmat_accelerate_api::register_device_provider(owner) };
            let source = owner
                .owner
                .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let input = source
                .handle
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            assert_ne!(ambient.device_id(), input.device_id());
            let result = randperm_gpu(&input, 1, 1);
            match outcome {
                Outcome::Double => {
                    let Value::GpuTensor(output) = result.unwrap() else {
                        panic!("valid permutation must remain resident")
                    };
                    assert_eq!(output.device_id(), input.device_id());
                    assert_eq!(
                        runmat_accelerate_api::handle_precision(&output),
                        Some(ProviderPrecision::F64)
                    );
                    assert!(runmat_accelerate_api::handle_is_explicit(&output));
                    assert!(!output.shares_allocation_with(&input));
                }
                Outcome::Single => {
                    let error = result.expect_err("wrong physical class must be rejected");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    assert_eq!(owner.results.lock().unwrap()[0].retirements(), 1);
                    assert_eq!(owner.uploads.load(Ordering::SeqCst), 0);
                }
                Outcome::Alias => {
                    let error = result.expect_err("borrowed input is not a fresh permutation");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    assert_eq!(owner.uploads.load(Ordering::SeqCst), 0);
                }
                Outcome::Failure => {
                    let error = result.expect_err("operational failure is terminal");
                    assert!(error.message().contains("permutation execution failed"));
                    assert_eq!(owner.uploads.load(Ordering::SeqCst), 0);
                }
                Outcome::Unsupported => {
                    let error = result.expect_err("explicit input cannot return a host result");
                    assert!(error.message().contains("failed to restore GPU result"));
                    assert_eq!(owner.uploads.load(Ordering::SeqCst), 1);
                }
            }
            assert_eq!(source.probe.retirements(), 0);
        }
    });
}
