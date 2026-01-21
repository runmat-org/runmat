//! MATLAB-compatible `compose` builtin that formats data into string arrays.
use runmat_builtins::{StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::core::string::{
    extract_format_spec, format_from_spec, FormatSpecData,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "compose",
        builtin_path = "crate::builtins::strings::core::compose"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "compose"
category: "strings/core"
keywords: ["compose", "format", "string array", "sprintf", "gpu"]
summary: "Format numeric, logical, and text data into MATLAB string arrays using printf-style placeholders."
references:
  - https://www.mathworks.com/help/matlab/ref/compose.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Formatting runs on the CPU. GPU inputs are gathered to host memory before substitution."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::compose::tests"
  integration: "builtins::strings::core::compose::tests::compose_gpu_argument"
---

# What does the `compose` function do in MATLAB / RunMat?
`compose(formatSpec, A1, ..., An)` substitutes data into MATLAB-compatible `%` placeholders and
returns the result as a string array. It combines `sprintf`-style formatting with string-array
broadcasting so you can generate multiple strings in one call.

## How does the `compose` function behave in MATLAB / RunMat?
- `formatSpec` must be text: a string scalar, string array, character vector, character array, or
  cell array of character vectors.
- If `formatSpec` is scalar and any argument array has more than one element, RunMat broadcasts the
  scalar specification over the array dimensions.
- When `formatSpec` is a string or character array with multiple elements, the output has the same
  shape as the specification. Each element uses the corresponding row or cell during formatting.
- Arguments can be numeric, logical, string, or text-like cell arrays. Non-text arguments are
  converted using MATLAB-compatible rules (logical values become `1` or `0`, complex numbers use the
  `a + bi` form).
- When you omit additional arguments, `compose(formatSpec)` simply converts the specification into a
  string array, preserving the original structure.
- Errors are raised if argument shapes are incompatible with the specification or if format specifiers
  are incomplete.

## `compose` Function GPU Execution Behaviour
`compose` is a residency sink. When inputs include GPU-resident tensors, RunMat gathers the data
back to host memory using the active acceleration provider before performing the formatting logic.
All formatted strings live in host memory, so acceleration providers do not need compose-specific
kernels.

## Examples of using the `compose` function in MATLAB / RunMat

### Formatting A Scalar Value Into A Sentence
```matlab
msg = compose("The answer is %d.", 42);
```
Expected output:
```matlab
msg = "The answer is 42."
```

### Broadcasting A Scalar Format Spec Over A Vector
```matlab
result = compose("Trial %d", 1:4);
```
Expected output:
```matlab
result = 1×4 string
    "Trial 1"    "Trial 2"    "Trial 3"    "Trial 4"
```

### Using A String Array Of Formats
```matlab
spec = ["max: %0.2f", "min: %0.2f"];
values = compose(spec, [3.14159, 0.125]);
```
Expected output:
```matlab
values = 1×2 string
    "max: 3.14"    "min: 0.12"
```

### Formatting Each Row Of A Character Array
```matlab
C = ['Row %02d'; 'Row %02d'; 'Row %02d'];
idx = compose(C, (1:3).');
```
Expected output:
```matlab
idx = 3×1 string
    "Row 01"
    "Row 02"
    "Row 03"
```

### Combining Real And Imaginary Parts
```matlab
Z = [1+2i, 3-4i];
txt = compose("z = %s", Z);
```
Expected output:
```matlab
txt = 1×2 string
    "z = 1+2i"    "z = 3-4i"
```

### Using A Cell Array Of Format Specs
```matlab
specs = {'%0.1f volts', '%0.1f amps'};
readings = compose(specs, {12.6, 3.4});
```
Expected output:
```matlab
readings = 2×1 string
    "12.6 volts"
    "3.4 amps"
```

### Formatting GPU-Resident Data
```matlab
G = gpuArray([10 20 30]);
labels = compose("Value %d", G);
```
Expected output:
```matlab
labels = 1×3 string
    "Value 10"    "Value 20"    "Value 30"
```
RunMat gathers `G` from the GPU before formatting, so the behaviour matches CPU inputs.

## FAQ

### What happens if the number of format arguments does not match the placeholders?
RunMat raises `compose: format data arguments must be scalars or match formatSpec size`. Ensure that
each placeholder has a corresponding value or broadcast the specification appropriately.

### Can `compose` handle complex numbers?
Yes. Complex numbers use MATLAB's canonical `a + bi` formatting, so `%s` specifiers receive the
string form of the complex scalar.

### How does `compose` treat logical inputs?
Logical values are converted to numeric `1` or `0` before formatting so they work with `%d`, `%i`,
or `%f` placeholders.

### Does `compose` modify the shape of the output?
No. The output matches the broadcasted size between `formatSpec` and the input arguments. Scalar
specifications broadcast across non-scalar arguments.

### What if I pass GPU arrays?
Inputs that reside on the GPU are automatically gathered to host memory before formatting. The
resulting string array always lives on the CPU.

### How do I emit literal percent signs?
Use `%%` inside `formatSpec` just like `sprintf`. The formatter converts `%%` into a single `%`.

### Can I mix scalars and arrays in the arguments list?
Yes, as long as non-scalar arguments all share the same number of elements or match the size of
`formatSpec`. Scalars broadcast across the target shape.

### What happens when `formatSpec` is empty?
`compose(formatSpec)` returns an empty string array with the same shape as `formatSpec`. When
`formatSpec` and arguments have zero elements, the output is `0×0`.

## See Also
`string`, `sprintf`, `strcat`, `join`
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::compose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "compose",
    op_kind: GpuOpKind::Custom("format"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Formatting always executes on the CPU; GPU tensors are gathered before substitution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::compose")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "compose",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Formatting builtin; not eligible for fusion and materialises host string arrays.",
};

fn compose_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("compose").build()
}

fn remap_compose_flow(mut err: RuntimeError) -> RuntimeError {
    err = map_control_flow_with_builtin(err, "compose");
    if let Some(message) = err.message.strip_prefix("string: ") {
        err.message = format!("compose: {message}");
        return err;
    }
    if !err.message.starts_with("compose: ") {
        err.message = format!("compose: {}", err.message);
    }
    err
}

#[runtime_builtin(
    name = "compose",
    category = "strings/core",
    summary = "Format values into MATLAB string arrays using printf-style placeholders.",
    keywords = "compose,format,string array,gpu",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::compose"
)]
async fn compose_builtin(format_spec: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let format_value = gather_if_needed_async(&format_spec)
        .await
        .map_err(remap_compose_flow)?;
    let mut gathered_args = Vec::with_capacity(rest.len());
    for arg in rest {
        let gathered = gather_if_needed_async(&arg)
            .await
            .map_err(remap_compose_flow)?;
        gathered_args.push(gathered);
    }

    if gathered_args.is_empty() {
        let spec = extract_format_spec(format_value)
            .await
            .map_err(remap_compose_flow)?;
        let array = format_spec_data_to_string_array(spec)?;
        return Ok(Value::StringArray(array));
    }

    let formatted = format_from_spec(format_value, gathered_args)
        .await
        .map_err(remap_compose_flow)?;
    Ok(Value::StringArray(formatted))
}

fn format_spec_data_to_string_array(spec: FormatSpecData) -> BuiltinResult<StringArray> {
    let shape = if spec.shape.is_empty() {
        match spec.specs.len() {
            0 => vec![0, 0],
            1 => vec![1, 1],
            len => vec![len, 1],
        }
    } else {
        spec.shape
    };
    StringArray::new(spec.specs, shape).map_err(|e| compose_flow(format!("compose: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    fn compose_builtin(format_spec: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::compose_builtin(format_spec, rest))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_scalar_numeric() {
        let result = compose_builtin(Value::from("Count %d"), vec![Value::Int(IntValue::I32(7))])
            .expect("compose");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["Count 7".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_broadcasts_scalar_spec() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = compose_builtin(Value::from("Item %0.0f"), vec![Value::Tensor(tensor)])
            .expect("compose");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(sa.data, vec!["Item 1".to_string(), "Item 2".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_zero_arguments_returns_spec() {
        let spec = Value::StringArray(
            StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap(),
        );
        let result = compose_builtin(spec, Vec::new()).expect("compose");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(sa.data, vec!["alpha".to_string(), "beta".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_mismatched_lengths_errors() {
        let spec = Value::StringArray(
            StringArray::new(vec!["%d".into(), "%d".into()], vec![1, 2]).unwrap(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = error_message(compose_builtin(spec, vec![Value::Tensor(tensor)]).unwrap_err());
        assert!(
            err.starts_with("compose: "),
            "expected compose prefix, got {err}"
        );
        assert!(
            err.contains("format data arguments must be scalars or match formatSpec size"),
            "unexpected error text: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn compose_gpu_argument() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                compose_builtin(Value::from("Value %0.0f"), vec![Value::GpuTensor(handle)])
                    .expect("compose");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![1, 3]);
                    assert_eq!(
                        sa.data,
                        vec![
                            "Value 1".to_string(),
                            "Value 2".to_string(),
                            "Value 3".to_string()
                        ]
                    );
                }
                other => panic!("expected string array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_parse() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn compose_wgpu_numeric_tensor_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.25, 2.5, 3.75], vec![1, 3]).unwrap();
        let cpu = compose_builtin(
            Value::from("Value %0.2f"),
            vec![Value::Tensor(tensor.clone())],
        )
        .expect("cpu compose");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("gpu upload");
        let gpu = compose_builtin(Value::from("Value %0.2f"), vec![Value::GpuTensor(handle)])
            .expect("gpu compose");
        match (cpu, gpu) {
            (Value::StringArray(expect), Value::StringArray(actual)) => {
                assert_eq!(actual.shape, expect.shape);
                assert_eq!(actual.data, expect.data);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }
}
