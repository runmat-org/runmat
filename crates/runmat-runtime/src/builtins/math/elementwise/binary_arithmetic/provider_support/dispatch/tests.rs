use std::sync::atomic::{AtomicUsize, Ordering};

use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, GpuTensorStorage,
    HostTensorView, NumericElementType,
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
    free_count: AtomicUsize,
}

impl ControlledProvider {
    fn new(behavior: AddBehavior) -> Self {
        Self {
            behavior,
            free_count: AtomicUsize::new(0),
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

    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        self.free_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn device_info(&self) -> String {
        "controlled arithmetic provider".to_owned()
    }

    fn device_id(&self) -> u32 {
        71
    }

    fn elem_add<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.behavior {
                AddBehavior::Unsupported => Err(
                    runmat_accelerate_api::unsupported_provider_operation("addition unavailable"),
                ),
                AddBehavior::Failed => Err(anyhow::anyhow!("device lost")),
                AddBehavior::InvalidOutput => Ok(handle(self, 99, vec![1, 1])),
            }
        })
    }
}

fn handle(provider: &dyn AccelProvider, buffer_id: u64, shape: Vec<usize>) -> GpuTensorHandle {
    GpuTensorHandle {
        shape,
        device_id: provider.device_id(),
        buffer_id,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real)
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
        let left = handle(provider, 1, vec![1, 2]);
        let right = handle(provider, 2, vec![1, 2]);
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
        let left = handle(provider, 1, vec![1, 2]);
        let right = handle(provider, 2, vec![1, 2]);
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
        let left = handle(provider, 1, vec![1, 2]);
        let right = handle(provider, 2, vec![1, 2]);
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
        assert_eq!(provider.free_count.load(Ordering::SeqCst), 1);
    });
}
