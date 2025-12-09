//! MATLAB-compatible `gpuInfo` builtin that formats the active GPU device metadata.

use super::gpudevice::{self, active_device_struct};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "gpuInfo")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "gpuInfo"
category: "acceleration/gpu"
keywords: ["gpuInfo", "gpu", "device info", "accelerate", "summary"]
summary: "Return a formatted status string that describes the active GPU provider."
references:
  - https://www.mathworks.com/help/parallel-computing/gpudevice.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Pure query builtin that formats provider metadata. When no provider is registered it returns `GPU[no provider]`."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::acceleration::gpu::gpuinfo::tests::gpu_info_with_provider_formats_summary"
  fallback: "builtins::acceleration::gpu::gpuinfo::tests::gpu_info_placeholder_matches_expectation"
  doc: "builtins::acceleration::gpu::gpuinfo::tests::doc_examples_present"
  wgpu: "builtins::acceleration::gpu::gpuinfo::tests::gpuInfo_wgpu_provider_reports_backend"
---

# What does the `gpuInfo` function do in MATLAB / RunMat?
`gpuInfo()` returns a concise, human-readable string that summarises the active GPU acceleration
provider. It is a convenience wrapper around [`gpuDevice`](./gpuDevice), intended for logging or
displaying status information in the REPL and notebooks.

## How does the `gpuInfo` function behave in MATLAB / RunMat?
- Queries the same device metadata as `gpuDevice()` and formats the fields into
  `GPU[key=value, ...]`.
- Includes identifiers (`device_id`, `index`), descriptive strings (`name`, `vendor`, `backend`)
  and capability hints (`precision`, `supports_double`, `memory_bytes` when available).
- Escapes string values using MATLAB-style single quote doubling so they are display-friendly.
- When no acceleration provider is registered, returns the placeholder string `GPU[no provider]`
  instead of throwing an error, making it safe to call unconditionally.
- Propagates unexpected errors (for example, if a provider fails while reporting metadata) so they
  can be diagnosed.

## GPU residency in RunMat (Do I need `gpuArray`?)
`gpuInfo` is a pure metadata query and never changes residency. Arrays stay wherever they already
live (GPU or CPU). Use `gpuArray`, `gather`, or RunMat Accelerate's auto-offload heuristics to move
data between devices as needed.

## Examples of using the `gpuInfo` function in MATLAB / RunMat

### Displaying GPU status in the REPL
```matlab
disp(gpuInfo());
```
Expected output (in-process provider):
```matlab
GPU[device_id=0, index=1, name='InProcess', vendor='RunMat', backend='inprocess', precision='double', supports_double=true]
```

### Emitting a log line before a computation
```matlab
fprintf("Running on %s\n", gpuInfo());
```
Expected output:
```matlab
Running on GPU[device_id=0, index=1, name='InProcess', vendor='RunMat', backend='inprocess', precision='double', supports_double=true]
```

### Checking for double precision support quickly
```matlab
summary = gpuInfo();
if contains(summary, "supports_double=true")
    disp("Double precision kernels available.");
else
    disp("Falling back to single precision.");
end
```

### Handling missing providers gracefully
```matlab
% Safe even when acceleration is disabled
status = gpuInfo();
if status == "GPU[no provider]"
    warning("GPU acceleration is currently disabled.");
end
```

### Combining `gpuInfo` with `gpuDevice` for structured data
```matlab
info = gpuDevice();
summary = gpuInfo();
if isfield(info, 'memory_bytes')
    fprintf("%s (memory: %.2f GB)\n", summary, info.memory_bytes / 1e9);
else
    fprintf("%s (memory: unknown)\n", summary);
end
```
Expected output (when the provider reports `memory_bytes`):
```matlab
GPU[device_id=0, index=1, name='InProcess', vendor='RunMat', backend='inprocess', precision='double', supports_double=true] (memory: 15.99 GB)
```
If the provider omits memory metadata, the fallback branch prints:
```matlab
GPU[device_id=0, index=1, name='InProcess', vendor='RunMat', backend='inprocess', precision='double', supports_double=true] (memory: unknown)
```

## FAQ

### Does `gpuInfo` change GPU state?
No. It only reads metadata and formats it into a string.

### Will `gpuInfo` throw an error when no provider is registered?
No. It returns `GPU[no provider]` so caller code can branch without exception handling.

### How is `gpuInfo` different from `gpuDevice`?
`gpuDevice` returns a struct that you can inspect programmatically. `gpuInfo` formats the same
information into a single string that is convenient for logging and display.

### Does the output order of fields stay stable?
Yes. Fields are emitted in the same order as the `gpuDevice` struct: identifiers, descriptive
strings, optional metadata, precision, and capability flags.

### Are strings escaped in MATLAB style?
Yes. Single quotes are doubled (e.g., `Ada'GPU` becomes `Ada''GPU`) so the summary can be pasted
back into MATLAB code without breaking literal syntax.

## See Also
[gpuDevice](./gpuDevice), [gpuArray](./gpuArray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec]
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

#[runmat_macros::register_fusion_spec]
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
    examples = "disp(gpuInfo())"
)]
fn gpu_info_builtin() -> Result<Value, String> {
    match active_device_struct() {
        Ok(info) => Ok(Value::String(format_summary(Some(&info)))),
        Err(err) if err == gpudevice::ERR_NO_PROVIDER => Ok(Value::String(format_summary(None))),
        Err(err) => Err(err.to_string()),
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
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;

    #[test]
    #[allow(non_snake_case)]
    fn gpuInfo_with_provider_formats_summary() {
        test_support::with_test_provider(|_| {
            let value = gpu_info_builtin().expect("gpuInfo");
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

    #[test]
    #[allow(non_snake_case)]
    fn gpuInfo_placeholder_matches_expectation() {
        let placeholder = format_summary(None);
        assert_eq!(placeholder, "GPU[no provider]");
    }

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
    #[allow(non_snake_case)]
    fn gpuInfo_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty(), "expected at least one MATLAB example");
    }

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
                eprintln!("Skipping gpuInfo WGPU provider test: {err}");
                return;
            }
        };

        let value = gpu_info_builtin().expect("gpuInfo");
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
