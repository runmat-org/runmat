//! MATLAB-compatible `gpuDevice` builtin with provider-aware semantics.

use runmat_accelerate_api::{ApiDeviceInfo, ProviderPrecision};
use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

/// Error used when no acceleration provider is registered.
pub(crate) const ERR_NO_PROVIDER: &str = "gpuDevice: no acceleration provider registered";
const ERR_TOO_MANY_INPUTS: &str = "gpuDevice: too many input arguments";
const ERR_UNSUPPORTED_ARGUMENT: &str = "gpuDevice: unsupported input argument";
const ERR_RESET_NOT_SUPPORTED: &str = "gpuDevice: reset is not supported by the active provider";
const ERR_INVALID_INDEX: &str = "gpuDevice: device index must be a positive integer";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "gpuDevice")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "gpuDevice"
category: "acceleration/gpu"
keywords: ["gpuDevice", "gpu", "device info", "accelerate", "provider"]
summary: "Query metadata about the active GPU provider and return it as a MATLAB struct."
references:
  - https://www.mathworks.com/help/parallel-computing/gpudevice.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Pure query builtin; it never enqueues GPU kernels. Providers that omit metadata simply leave the corresponding struct fields absent."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::acceleration::gpu::gpudevice::tests::gpu_device_returns_struct"
  doc: "builtins::acceleration::gpu::gpudevice::tests::gpu_device_doc_examples_present"
  integration: "tests::gpu::gpu_device_returns_struct"
  wgpu: "builtins::acceleration::gpu::gpudevice::tests::gpu_device_wgpu_reports_metadata"
---

# What does the `gpuDevice` function do in MATLAB / RunMat?
`info = gpuDevice()` queries the active accelerator provider and returns a MATLAB struct that
describes the GPU (or GPU-like backend) that RunMat is currently using. The struct mirrors
MathWorks MATLAB's `gpuDevice` metadata, exposing identifiers, vendor information, memory hints,
and precision support so you can adapt algorithms at runtime.

The returned struct contains a subset of these fields (providers may omit ones they cannot
populate):

- `device_id` — zero-based identifier reported by the provider.
- `index` — MATLAB-style one-based index derived from `device_id`.
- `name` — human-readable adapter name.
- `vendor` — provider-reported vendor or implementation name.
- `backend` — backend identifier such as `inprocess` or `Vulkan` (optional).
- `memory_bytes` — total device memory in bytes when known (optional).
- `precision` — string describing the scalar precision used for kernels (`"double"` or `"single"`).
- `supports_double` — logical flag that is `true` when double precision kernels are available.

The builtin raises `gpuDevice: no acceleration provider registered` when no provider is active.

## How does the `gpuDevice` function behave in MATLAB / RunMat?
- Requires an acceleration provider that implements RunMat Accelerate's `AccelProvider` trait.
- Returns a struct so you can access fields with dot notation: `gpuDevice().name`.
- Does not mutate GPU state or enqueue kernels—it is safe to call frequently.
- Accepts a scalar device index; `gpuDevice(1)` returns the active provider, while any other index
  raises the MATLAB-style error `gpuDevice: GPU device with index N not available`.
- Requests to reset the provider using `gpuDevice('reset')` or `gpuDevice([])` currently raise
  `gpuDevice: reset is not supported by the active provider`.
- Hooks into `gpuInfo` so the string-form summary stays in sync with the struct fields.

## GPU residency in RunMat (Do I need `gpuArray`?)
`gpuDevice` purely reports metadata and does not change residency. Arrays remain on the GPU or CPU
exactly as they were prior to the call. Use `gpuArray`, `gather`, and the planner-controlled
automatic residency features to move data as needed.

## Examples of using the `gpuDevice` function in MATLAB / RunMat

### Inspecting the active GPU provider
```matlab
info = gpuDevice();
disp(info.name);
```
Expected output (in-process provider):
```matlab
InProcess
```

### Displaying vendor and backend metadata
```matlab
info = gpuDevice();
fprintf("Vendor: %s (backend: %s)\n", info.vendor, info.backend);
```
Expected output:
```matlab
Vendor: RunMat (backend: inprocess)
```

### Checking whether double precision is supported
```matlab
info = gpuDevice();
if info.supports_double
    disp("Double precision kernels are available.");
else
    disp("Provider only exposes single precision.");
end
```

### Formatting a user-facing status message
```matlab
summary = gpuInfo();
disp("Active GPU summary:");
disp(summary);
```
Expected output:
```matlab
Active GPU summary:
GPU[device_id=0, index=1, name='InProcess', vendor='RunMat', backend='inprocess', precision='double', supports_double=true]
```

### Handling missing providers gracefully
```matlab
try
    info = gpuDevice();
catch ex
    warning("GPU unavailable: %s", ex.message);
end
```
If no provider is registered:
```matlab
Warning: GPU unavailable: gpuDevice: no acceleration provider registered
```

## FAQ

### Do I need to call `gpuDevice` before using other GPU builtins?
No. RunMat initialises the active provider during startup. `gpuDevice` is purely informational and
can be called at any time to inspect the current provider.

### Why are some fields missing from the struct?
Providers only fill metadata they can reliably supply. For example, the in-process test provider
does not report `memory_bytes`. Real GPU backends typically populate additional fields.

### What happens if there is no GPU provider?
RunMat raises `gpuDevice: no acceleration provider registered`. You can catch this error and fall
back to CPU code, as shown in the examples above.

### Does `gpuDevice` support selecting or resetting devices?
RunMat currently exposes a single provider. `gpuDevice(1)` returns that provider, matching MATLAB's
first-device semantics, while any other index raises `gpuDevice: GPU device with index N not available`.
Reset requests (`gpuDevice('reset')` or `gpuDevice([])`) are not implemented yet and return
`gpuDevice: reset is not supported by the active provider`.

### How can I get a quick string summary instead of a struct?
Use `gpuInfo()`. It internally calls `gpuDevice` and formats the struct fields into a concise status
string that is convenient for logging or display.

## See Also
[gpuArray](./gpuArray), [gather](./gather), [gpuInfo](./gpuInfo)
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuDevice",
    op_kind: GpuOpKind::Custom("device-info"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Pure metadata query; does not enqueue GPU kernels. Returns an error when no provider is registered.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuDevice",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; the builtin returns a host-resident struct.",
};

/// Query the active provider and return a metadata struct describing the GPU device.
#[cfg_attr(
    feature = "doc_export",
    runtime_builtin(
        name = "gpuDevice",
        category = "acceleration/gpu",
        summary = "Return information about the active GPU device/provider.",
        keywords = "gpu,device,info,accelerate"
    )
)]
#[cfg_attr(
    not(feature = "doc_export"),
    runtime_builtin(
        name = "gpuDevice",
        category = "acceleration/gpu",
        summary = "Return information about the active GPU device/provider.",
        keywords = "gpu,device,info,accelerate"
    )
)]
fn gpu_device_builtin(args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [] => active_device_struct()
            .map(Value::Struct)
            .map_err(|err| err.to_string()),
        [arg] => handle_single_argument(arg),
        _ => Err(ERR_TOO_MANY_INPUTS.to_string()),
    }
}

/// Internal helper that queries the provider and returns a populated struct.
pub(crate) fn active_device_struct() -> Result<StructValue, &'static str> {
    let provider = runmat_accelerate_api::provider().ok_or(ERR_NO_PROVIDER)?;
    let info = provider.device_info_struct();
    let precision = provider.precision();
    Ok(build_struct(&info, precision))
}

fn build_struct(info: &ApiDeviceInfo, precision: ProviderPrecision) -> StructValue {
    let mut st = StructValue::new();
    st.insert("device_id", Value::Int(IntValue::U32(info.device_id)));
    st.insert(
        "index",
        Value::Int(IntValue::U32(info.device_id.saturating_add(1))),
    );
    st.insert("name", Value::String(info.name.clone()));
    st.insert("vendor", Value::String(info.vendor.clone()));
    if let Some(backend) = info.backend.as_ref() {
        st.insert("backend", Value::String(backend.clone()));
    }
    if let Some(bytes) = info.memory_bytes {
        st.insert("memory_bytes", Value::Int(IntValue::U64(bytes)));
    }
    st.insert(
        "precision",
        Value::String(
            match precision {
                ProviderPrecision::F64 => "double",
                ProviderPrecision::F32 => "single",
            }
            .to_string(),
        ),
    );
    st.insert(
        "supports_double",
        Value::Bool(matches!(precision, ProviderPrecision::F64)),
    );
    st
}

fn is_keyword(value: &Value, keyword: &str) -> bool {
    match value {
        Value::String(s) => s.trim().eq_ignore_ascii_case(keyword),
        Value::CharArray(ca) if ca.rows == 1 => {
            let collected: String = ca.data.iter().collect();
            collected.trim().eq_ignore_ascii_case(keyword)
        }
        _ => false,
    }
}

fn handle_single_argument(arg: &Value) -> Result<Value, String> {
    if is_reset_arg(arg) {
        return Err(ERR_RESET_NOT_SUPPORTED.to_string());
    }

    match parse_device_index(arg)? {
        Some(index) => {
            let info = active_device_struct().map_err(|err| err.to_string())?;
            let current_index = struct_device_index(&info).unwrap_or(1);
            if index == current_index {
                Ok(Value::Struct(info))
            } else {
                Err(format!(
                    "gpuDevice: GPU device with index {} not available",
                    index
                ))
            }
        }
        None => Err(ERR_UNSUPPORTED_ARGUMENT.to_string()),
    }
}

fn struct_device_index(info: &StructValue) -> Option<u32> {
    info.fields.get("index").and_then(|value| match value {
        Value::Int(intv) => {
            let idx = intv.to_i64();
            if idx >= 0 && idx <= u32::MAX as i64 {
                Some(idx as u32)
            } else {
                None
            }
        }
        Value::Num(n) if n.is_finite() && *n >= 0.0 => {
            let rounded = n.round();
            if (rounded - n).abs() <= 1e-9 {
                Some(rounded as u32)
            } else {
                None
            }
        }
        _ => None,
    })
}

fn is_reset_arg(value: &Value) -> bool {
    if is_keyword(value, "reset") {
        return true;
    }
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(la) => la.data.is_empty(),
        _ => false,
    }
}

fn parse_device_index(value: &Value) -> Result<Option<u32>, String> {
    match value {
        Value::Int(i) => int_to_index(i.to_i64()),
        Value::Num(n) => num_to_index(*n),
        Value::Bool(b) => {
            if *b {
                Ok(Some(1))
            } else {
                Err(ERR_INVALID_INDEX.to_string())
            }
        }
        Value::Tensor(t) => match t.data.len() {
            0 => Ok(None),
            1 => num_to_index(t.data[0]),
            _ => Err(ERR_INVALID_INDEX.to_string()),
        },
        Value::LogicalArray(la) => match la.data.len() {
            0 => Ok(None),
            1 => {
                if la.data[0] != 0 {
                    Ok(Some(1))
                } else {
                    Err(ERR_INVALID_INDEX.to_string())
                }
            }
            _ => Err(ERR_INVALID_INDEX.to_string()),
        },
        _ => Ok(None),
    }
}

fn int_to_index(raw: i64) -> Result<Option<u32>, String> {
    if raw <= 0 {
        return Err(ERR_INVALID_INDEX.to_string());
    }
    if raw > u32::MAX as i64 {
        return Err(ERR_INVALID_INDEX.to_string());
    }
    Ok(Some(raw as u32))
}

fn num_to_index(raw: f64) -> Result<Option<u32>, String> {
    if !raw.is_finite() {
        return Err(ERR_INVALID_INDEX.to_string());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-9 {
        return Err(ERR_INVALID_INDEX.to_string());
    }
    let idx = rounded as i64;
    int_to_index(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn gpu_device_returns_struct() {
        test_support::with_test_provider(|_| {
            let value = gpu_device_builtin(Vec::new()).expect("gpuDevice");
            match value {
                Value::Struct(s) => {
                    assert!(s.fields.contains_key("device_id"));
                    assert!(s.fields.contains_key("index"));
                    assert!(s.fields.contains_key("name"));
                    assert!(s.fields.contains_key("vendor"));
                    assert!(s.fields.contains_key("precision"));
                    assert!(s.fields.contains_key("supports_double"));
                }
                other => panic!("expected struct, got {other:?}"),
            }
        });
    }

    #[test]
    fn gpu_device_accepts_current_index() {
        test_support::with_test_provider(|_| {
            let tensor_scalar =
                runmat_builtins::Tensor::new(vec![1.0], vec![1, 1]).expect("scalar tensor");
            let logical_scalar = runmat_builtins::LogicalArray::new(vec![1u8], vec![1]).unwrap();
            let cases = vec![
                Value::Int(IntValue::I32(1)),
                Value::Num(1.0),
                Value::Bool(true),
                Value::Tensor(tensor_scalar),
                Value::LogicalArray(logical_scalar),
            ];
            for case in cases {
                let value = gpu_device_builtin(vec![case]).expect("gpuDevice");
                assert!(matches!(value, Value::Struct(_)));
            }
        });
    }

    #[test]
    fn gpu_device_out_of_range_index_errors() {
        test_support::with_test_provider(|_| {
            let err = gpu_device_builtin(vec![Value::Num(2.0)]).unwrap_err();
            assert!(
                err.contains("gpuDevice: GPU device with index 2 not available"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn gpu_device_unsupported_argument_errors() {
        test_support::with_test_provider(|_| {
            let err = gpu_device_builtin(vec![Value::from("status")]).unwrap_err();
            assert_eq!(err, ERR_UNSUPPORTED_ARGUMENT);
        });
    }

    #[test]
    fn gpu_device_reset_argument_reports_not_supported() {
        test_support::with_test_provider(|_| {
            let err = gpu_device_builtin(vec![Value::from(" RESET ")]).unwrap_err();
            assert_eq!(err, ERR_RESET_NOT_SUPPORTED);
        });
    }

    #[test]
    fn gpu_device_reset_char_array_argument_reports_not_supported() {
        test_support::with_test_provider(|_| {
            let chars = runmat_builtins::CharArray::new("reset".chars().collect(), 1, 5).unwrap();
            let err = gpu_device_builtin(vec![Value::CharArray(chars)]).unwrap_err();
            assert_eq!(err, ERR_RESET_NOT_SUPPORTED);
        });
    }

    #[test]
    fn gpu_device_empty_array_argument_reports_not_supported() {
        test_support::with_test_provider(|_| {
            let empty = runmat_builtins::Tensor::zeros(vec![0, 0]);
            let err = gpu_device_builtin(vec![Value::Tensor(empty)]).unwrap_err();
            assert_eq!(err, ERR_RESET_NOT_SUPPORTED);
        });
    }

    #[test]
    fn gpu_device_invalid_index_rejected() {
        test_support::with_test_provider(|_| {
            let cases = vec![
                Value::Num(0.0),
                Value::Int(IntValue::I32(0)),
                Value::Num(-1.0),
                Value::Num(1.5),
                Value::Bool(false),
                Value::LogicalArray(
                    runmat_builtins::LogicalArray::new(vec![0u8], vec![1]).unwrap(),
                ),
                Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            ];
            for case in cases {
                let err = gpu_device_builtin(vec![case]).unwrap_err();
                assert_eq!(err, ERR_INVALID_INDEX);
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gpu_device_wgpu_reports_metadata() {
        use runmat_accelerate::backend::wgpu::provider as wgpu_provider;

        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let value = gpu_device_builtin(Vec::new()).expect("gpuDevice");
        match value {
            Value::Struct(info) => {
                let name = info
                    .fields
                    .get("name")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s),
                        _ => None,
                    })
                    .expect("name field");
                assert!(
                    !name.is_empty(),
                    "expected non-empty adapter name from wgpu provider"
                );

                let backend = info.fields.get("backend").and_then(|v| match v {
                    Value::String(s) => Some(s),
                    _ => None,
                });
                assert!(backend.is_some(), "expected backend field to be present");

                if let Some(Value::Int(memory)) = info.fields.get("memory_bytes") {
                    assert!(
                        memory.to_i64() > 0,
                        "expected positive memory_bytes, got {:?}",
                        memory
                    );
                }

                if let Some(Value::Bool(supports_double)) = info.fields.get("supports_double") {
                    if *supports_double {
                        assert_eq!(
                            info.fields.get("precision"),
                            Some(&Value::String("double".to_string()))
                        );
                    }
                }
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[test]
    fn gpu_device_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(blocks.len() >= 5, "expected at least five doc examples");
    }
}
