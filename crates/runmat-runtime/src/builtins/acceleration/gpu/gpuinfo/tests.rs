use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;

fn call() -> crate::BuiltinResult<Value> {
    block_on(gpu_info_builtin())
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
            Value::String(text) => {
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
    let mut info = StructValue::new();
    info.insert("device_id", Value::Int(IntValue::U32(0)));
    info.insert("index", Value::Int(IntValue::U32(1)));
    info.insert("name", Value::String("Ada'GPU".into()));
    info.insert("vendor", Value::String("RunMat".into()));
    info.insert("memory_bytes", Value::Int(IntValue::U64(12)));
    info.insert("load", Value::Num(0.5));
    info.insert("precision", Value::String("double".into()));
    info.insert("supports_double", Value::Bool(true));
    info.insert("ignored", Value::Struct(StructValue::new()));

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

    let options = WgpuProviderOptions {
        force_fallback_adapter: true,
        ..Default::default()
    };

    let provider = register_wgpu_provider(options).expect("required WGPU adapter");

    let value = call().expect("gpuInfo");
    match value {
        Value::String(text) => {
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
