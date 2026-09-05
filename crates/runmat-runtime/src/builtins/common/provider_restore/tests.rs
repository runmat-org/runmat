use super::upload_value_protected;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct F32OnlyProvider {
    upload_calls: AtomicUsize,
}

impl runmat_accelerate_api::AccelProvider for F32OnlyProvider {
    fn upload(
        &self,
        _host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.upload_calls.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("unexpected upload"))
    }

    fn download<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow::anyhow!("download unsupported")) })
    }

    fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn device_info(&self) -> String {
        "f32-only-test".to_string()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F32
    }
}

#[test]
fn preserves_non_f32_exact_double_without_metadata_relabeling() {
    test_support::with_test_provider(|provider| {
        let value = 1.0000000000000002_f64;
        let output = upload_value_protected(provider, Value::Num(value), "restore-test", &[])
            .expect("double restore");
        let Value::GpuTensor(handle) = output else {
            panic!("expected resident output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&handle),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        let downloaded = block_on(provider.download(&handle)).expect("download");
        assert_eq!(downloaded.data, vec![value]);
    });
}

#[test]
fn rejects_double_when_f32_only_provider_declines_typed_upload() {
    let provider = F32OnlyProvider {
        upload_calls: AtomicUsize::new(0),
    };
    let error = upload_value_protected(
        &provider,
        Value::Num(1.0000000000000002),
        "restore-test",
        &[],
    )
    .expect_err("f32-only provider cannot supply a double result");
    assert!(error.message().contains("failed to restore result"));
    assert_eq!(provider.upload_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn accepts_single_when_provider_supports_typed_upload() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::from_f32(vec![1.0000001], vec![1, 1]).expect("single");
        let output = upload_value_protected(provider, Value::Tensor(tensor), "restore-test", &[])
            .expect("typed single result");
        let Value::GpuTensor(handle) = output else {
            panic!("expected resident single output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&handle),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
    });
}
