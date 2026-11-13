use std::sync::Mutex;

use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType,
};
use runmat_builtins::PLUS_CATALOG_ENTRY;

use crate::builtins::common::test_support;

use super::super::ArithmeticProviderOperation;
use super::try_pair;

#[derive(Clone, Copy)]
enum AddBehavior {
    Unsupported,
    Failed,
    InvalidOutput,
}

struct ControlledProvider {
    behavior: AddBehavior,
    owner: TestAllocationOwner,
    output_probe: Mutex<Option<RetirementProbe>>,
}

impl ControlledProvider {
    fn new(behavior: AddBehavior) -> Self {
        Self {
            behavior,
            owner: TestAllocationOwner::new(),
            output_probe: Mutex::new(None),
        }
    }
}

impl AccelProvider for ControlledProvider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("dispatch contract tests construct resident handles directly")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async {
            Err(anyhow::anyhow!(
                "fallback download is outside this unit test"
            ))
        })
    }

    fn device_info(&self) -> String {
        "controlled arithmetic provider".to_owned()
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn elem_add<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.behavior {
                AddBehavior::Unsupported => Err(
                    runmat_accelerate_api::unsupported_provider_operation("addition unavailable")
                        .into(),
                ),
                AddBehavior::Failed => Err(anyhow::anyhow!("device lost").into()),
                AddBehavior::InvalidOutput => {
                    let allocation = self
                        .owner
                        .allocate_zeroed(
                            vec![1, 1],
                            NumericElementType::F64,
                            GpuTensorStorage::Real,
                        )
                        .expect("actual invalid-shape output allocation");
                    *self.output_probe.lock().unwrap() = Some(allocation.probe);
                    Ok(allocation.handle)
                }
            }
        })
    }
}

fn handle(provider: &ControlledProvider, shape: Vec<usize>) -> GpuTensorHandle {
    provider
        .owner
        .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
        .expect("owned operand")
        .handle
}

fn with_provider<R>(
    behavior: AddBehavior,
    test: impl FnOnce(&'static ControlledProvider) -> R,
) -> R {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(ControlledProvider::new(behavior)));
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    test(provider)
}

#[test]
fn typed_unsupported_operation_permits_host_fallback() {
    with_provider(AddBehavior::Unsupported, |provider| {
        let left = handle(provider, vec![1, 2]);
        let right = handle(provider, vec![1, 2]);
        let result = block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .expect("typed unsupported hook");
        assert!(result.is_none());
    });
}

#[test]
fn operational_provider_failure_is_not_reclassified_as_unsupported() {
    with_provider(AddBehavior::Failed, |provider| {
        let left = handle(provider, vec![1, 2]);
        let right = handle(provider, vec![1, 2]);
        let error = block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .expect_err("terminal provider failure");
        assert!(error.message().contains("device lost"));
    });
}

#[test]
fn invalid_provider_output_is_rejected_and_released() {
    with_provider(AddBehavior::InvalidOutput, |provider| {
        let left = handle(provider, vec![1, 2]);
        let right = handle(provider, vec![1, 2]);
        let left_lease = left.allocation_lease().downgrade();
        let right_lease = right.allocation_lease().downgrade();
        let error = block_on(try_pair(
            ArithmeticProviderOperation::Add,
            PLUS_CATALOG_ENTRY.identity,
            &left,
            &right,
        ))
        .expect_err("invalid output metadata");
        assert!(error
            .message()
            .contains("invalid element-wise output metadata"));
        drop(error);
        assert_eq!(
            provider
                .output_probe
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
        assert!(left_lease.acquire().is_some());
        assert!(right_lease.acquire().is_some());
        drop((left, right));
        assert!(left_lease.acquire().is_none());
        assert!(right_lease.acquire().is_none());
    });
}
