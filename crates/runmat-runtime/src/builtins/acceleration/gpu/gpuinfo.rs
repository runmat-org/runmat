//! MATLAB-compatible `gpuInfo` builtin that formats the active GPU device metadata.

use super::gpudevice::{self, active_device_struct};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::acceleration::gpu::type_resolvers::gpuinfo_type;

use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gpuinfo")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuInfo",
    op_kind: GpuOpKind::Custom("device-summary"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Formats metadata reported by the active provider; no GPU kernels are launched.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::acceleration::gpu::gpuinfo")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuInfo",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion—returns a host-resident string.",
};

#[runtime_builtin(
    name = "gpuInfo",
    category = "acceleration/gpu",
    summary = "Return a formatted status string that describes the active GPU provider.",
    keywords = "gpu,gpuInfo,device,info,accelerate",
    examples = "disp(gpuInfo())",
    type_resolver(gpuinfo_type),
    builtin_path = "crate::builtins::acceleration::gpu::gpuinfo"
)]
async fn gpu_info_builtin() -> crate::BuiltinResult<Value> {
    match active_device_struct() {
        Ok(info) => Ok(Value::String(format_summary(Some(&info)))),
        Err(err) if err.message() == gpudevice::ERR_NO_PROVIDER => {
            Ok(Value::String(format_summary(None)))
        }
        Err(err) => Err(err),
    }
}

fn format_summary(info: Option<&StructValue>) -> String {
    match info {
        Some(struct_value) => {
            let mut parts = Vec::new();
            for (key, value) in struct_value.fields.iter() {
                if let Some(fragment) = format_field(key, value) {
                    parts.push(fragment);
                }
            }
            format!("GPU[{}]", parts.join(", "))
        }
        None => "GPU[no provider]".to_string(),
    }
}

fn format_field(key: &str, value: &Value) -> Option<String> {
    match value {
        Value::Int(intv) => Some(format!("{key}={}", int_to_string(intv))),
        Value::Num(n) => Some(format!("{key}={}", Value::Num(*n))),
        Value::Bool(flag) => Some(format!("{key}={}", if *flag { "true" } else { "false" })),
        Value::String(text) => Some(format!("{key}='{}'", escape_single_quotes(text))),
        _ => None,
    }
}

fn int_to_string(value: &IntValue) -> String {
    match value {
        IntValue::I8(v) => v.to_string(),
        IntValue::I16(v) => v.to_string(),
        IntValue::I32(v) => v.to_string(),
        IntValue::I64(v) => v.to_string(),
        IntValue::U8(v) => v.to_string(),
        IntValue::U16(v) => v.to_string(),
        IntValue::U32(v) => v.to_string(),
        IntValue::U64(v) => v.to_string(),
    }
}

fn escape_single_quotes(text: &str) -> String {
    text.replace('\'', "''")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::Type;

    fn call() -> crate::BuiltinResult<Value> {
        block_on(gpu_info_builtin())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[allow(non_snake_case)]
    fn gpuInfo_with_provider_formats_summary() {
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
    #[allow(non_snake_case)]
    fn gpuInfo_placeholder_matches_expectation() {
        let placeholder = format_summary(None);
        assert_eq!(placeholder, "GPU[no provider]");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[allow(non_snake_case)]
    fn gpuInfo_format_summary_handles_core_value_types() {
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

    #[test]
    fn gpuinfo_type_is_string() {
        assert_eq!(gpuinfo_type(&[]), Type::String);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    #[allow(non_snake_case)]
    fn gpuInfo_wgpu_provider_reports_backend() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let options = WgpuProviderOptions {
            force_fallback_adapter: true,
            ..Default::default()
        };

        let provider = match register_wgpu_provider(options) {
            Ok(p) => p,
            Err(err) => {
                tracing::warn!("Skipping gpuInfo WGPU provider test: {err}");
                return;
            }
        };

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
}
