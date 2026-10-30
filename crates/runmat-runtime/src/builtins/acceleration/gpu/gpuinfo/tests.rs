use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorView,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn call() -> crate::BuiltinResult<Value> {
    block_on(gpu_info_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
}

struct MetadataProbeProvider {
    reads: AtomicUsize,
}

static METADATA_PROBE: MetadataProbeProvider = MetadataProbeProvider {
    reads: AtomicUsize::new(0),
};

impl AccelProvider for MetadataProbeProvider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn device_info(&self) -> String {
        "metadata-probe".into()
    }

    fn device_id(&self) -> u32 {
        97
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        self.reads.fetch_add(1, Ordering::SeqCst);
        ApiDeviceInfo {
            device_id: self.device_id(),
            name: self.device_info(),
            vendor: "fixture".into(),
            backend: None,
            memory_bytes: None,
        }
    }
}

#[test]
fn extra_argument_has_own_error_before_provider_metadata_access() {
    let _guard = test_support::accel_test_lock();
    METADATA_PROBE.reads.store(0, Ordering::SeqCst);
    let _selected = ThreadProviderGuard::set(Some(&METADATA_PROBE));
    let error = block_on(gpu_info_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect_err("gpuInfo accepts no input arguments");
    assert_eq!(
        error.identifier(),
        GPU_INFO_ERROR_TOO_MANY_INPUTS.identifier
    );
    assert_eq!(error.to_string(), GPU_INFO_ERROR_TOO_MANY_INPUTS.message);
    assert_eq!(METADATA_PROBE.reads.load(Ordering::SeqCst), 0);
    assert!(
        matches!(call(), Ok(Value::String(runmat_value::StringElement::Text(text))) if text.contains("metadata-probe"))
    );
    assert_eq!(METADATA_PROBE.reads.load(Ordering::SeqCst), 1);
}

#[test]
fn gpu_info_without_provider_has_typed_placeholder() {
    let _guard = test_support::accel_test_lock();
    assert_eq!(
        call().expect("absence is a valid status"),
        Value::String("GPU[no provider]".into())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_info_with_provider_formats_summary() {
    test_support::with_test_provider(|_| {
        let value = call().expect("gpuInfo");
        match value {
            Value::String(runmat_value::StringElement::Text(text)) => {
                assert!(text.starts_with("GPU["), "unexpected prefix: {text}");
                assert!(
                    text.contains("name='InProcess'"),
                    "expected provider name in: {text}"
                );
                assert!(
                    text.contains("vendor='RunMat'"),
                    "expected vendor in: {text}"
                );
            }
            other => panic!("expected string result, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_info_placeholder_matches_expectation() {
    let placeholder = format_summary(None);
    assert_eq!(placeholder, "GPU[no provider]");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_info_format_summary_handles_core_value_types() {
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    info.insert("device_id", Value::Int(IntValue::U32(0)));
    info.insert("index", Value::Int(IntValue::U32(1)));
    info.insert("name", Value::String("Ada'GPU".into()));
    info.insert("vendor", Value::String("RunMat".into()));
    info.insert("memory_bytes", Value::Int(IntValue::U64(12)));
    info.insert("load", Value::Num(0.5));
    info.insert("precision", Value::String("double".into()));
    info.insert("supports_double", Value::Bool(true));
    info.insert("ignored", Value::Struct(StructValue::new()));
    let info = info.finish();

    let summary = format_summary(Some(&info));
    assert!(
        summary.starts_with("GPU["),
        "summary should start with GPU[: {summary}"
    );
    assert!(
        summary.contains("device_id=0"),
        "expected device_id field in: {summary}"
    );
    assert!(
        summary.contains("index=1"),
        "expected index field in: {summary}"
    );
    assert!(
        summary.contains("name='Ada''GPU'"),
        "expected escaped name in: {summary}"
    );
    assert!(
        summary.contains("vendor='RunMat'"),
        "expected vendor field in: {summary}"
    );
    assert!(
        summary.contains("memory_bytes=12"),
        "expected memory_bytes field in: {summary}"
    );
    assert!(
        summary.contains("load=0.5"),
        "expected numeric load field in: {summary}"
    );
    assert!(
        summary.contains("precision='double'"),
        "expected precision field in: {summary}"
    );
    assert!(
        summary.contains("supports_double=true"),
        "expected supports_double field in: {summary}"
    );
    assert!(
        !summary.contains("ignored"),
        "unexpected field 'ignored' present in: {summary}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires an available WGPU adapter; run as a required hardware witness"]
fn gpu_info_wgpu_provider_reports_backend() {
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    let provider =
        register_wgpu_provider(WgpuProviderOptions::default()).expect("required WGPU adapter");

    let value = call().expect("gpuInfo");
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            assert!(text.starts_with("GPU["), "unexpected prefix: {text}");
            let info = provider.device_info_struct();
            let runmat_accelerate_api::ApiDeviceInfo {
                vendor, backend, ..
            } = info;
            let expected_vendor = escape_single_quotes(&vendor);
            assert!(
                text.contains(&format!("vendor='{}'", expected_vendor)),
                "expected vendor '{}' in summary: {}",
                vendor,
                text
            );
            if let Some(backend) = backend {
                if !backend.is_empty() {
                    assert!(
                        text.contains(&format!("backend='{}'", backend)),
                        "expected backend '{}' in summary: {}",
                        backend,
                        text
                    );
                }
            }
        }
        other => panic!("expected string result, got {other:?}"),
    }
}
