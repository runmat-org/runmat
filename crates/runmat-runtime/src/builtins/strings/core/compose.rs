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
use crate::builtins::strings::type_resolvers::string_array_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

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
    type_resolver(string_array_type),
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
    use runmat_builtins::{IntValue, ResolveContext, Tensor, Type};

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

    #[test]
    fn compose_type_is_string_array() {
        assert_eq!(
            string_array_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
    }
}
