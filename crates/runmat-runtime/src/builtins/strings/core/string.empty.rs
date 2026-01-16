//! MATLAB-compatible `string.empty` builtin for RunMat.

use runmat_builtins::{StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const LABEL: &str = "string.empty";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::string_empty")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "string.empty",
    op_kind: GpuOpKind::Custom("constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only constructor that returns a new empty string array without contacting GPU providers.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::core::string_empty"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "string.empty",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Pure constructor; fusion planner treats calls as non-fusable sinks.",
};

#[runtime_builtin(
    name = "string.empty",
    category = "strings/core",
    summary = "Construct an empty string array with MATLAB-compatible dimensions.",
    keywords = "string.empty,empty,string array,preallocate",
    accel = "none",
    builtin_path = "crate::builtins::strings::core::string_empty"
)]
fn string_empty_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let shape = parse_shape(&rest)?;
    let total: usize = shape.iter().product();
    debug_assert_eq!(total, 0, "string.empty must produce an empty array");
    let data = Vec::<String>::new();
    let array = StringArray::new(data, shape).map_err(|e| format!("{LABEL}: {e}"))?;
    Ok(Value::StringArray(array))
}

fn parse_shape(args: &[Value]) -> Result<Vec<usize>, String> {
    if args.is_empty() {
        return Ok(vec![0, 0]);
    }

    let mut explicit_dims: Vec<usize> = Vec::new();
    let mut like_shape: Option<Vec<usize>> = None;
    let mut idx = 0;

    while idx < args.len() {
        let arg_host = gather_if_needed(&args[idx]).map_err(|e| format!("{LABEL}: {e}"))?;

        if let Some(keyword) = keyword_of(&arg_host) {
            if keyword.as_str() == "like" {
                if like_shape.is_some() {
                    return Err(format!(
                        "{LABEL}: multiple 'like' prototypes are not supported"
                    ));
                }
                let Some(proto_raw) = args.get(idx + 1) else {
                    return Err(format!("{LABEL}: expected prototype after 'like'"));
                };
                let proto = gather_if_needed(proto_raw).map_err(|e| format!("{LABEL}: {e}"))?;
                like_shape = Some(prototype_dims(&proto));
                idx += 2;
                continue;
            }
            // Unrecognized keywords are treated as non-keyword inputs and will
            // be validated under numeric size parsing below.
        }

        if let Some(parsed) = extract_dims(&arg_host, LABEL)? {
            if explicit_dims.is_empty() {
                explicit_dims = parsed;
            } else {
                explicit_dims.extend(parsed);
            }
            idx += 1;
            continue;
        }

        return Err(format!(
            "{LABEL}: size inputs must be numeric scalars or size vectors"
        ));
    }

    let shape = if !explicit_dims.is_empty() {
        shape_from_explicit_dims(&explicit_dims)
    } else if let Some(proto_shape) = like_shape {
        shape_from_like(&proto_shape)
    } else {
        vec![0, 0]
    };
    ensure_empty_shape(&shape)?;
    Ok(shape)
}

fn shape_from_explicit_dims(dims: &[usize]) -> Vec<usize> {
    match dims.len() {
        0 => vec![0, 0],
        1 => vec![0, dims[0]],
        _ => {
            let mut shape = Vec::with_capacity(dims.len());
            shape.push(0);
            shape.extend_from_slice(&dims[1..]);
            shape
        }
    }
}

fn shape_from_like(proto: &[usize]) -> Vec<usize> {
    if proto.is_empty() {
        return vec![0, 0];
    }
    if proto.len() == 1 {
        return vec![0, proto[0]];
    }
    let mut shape = Vec::with_capacity(proto.len());
    shape.push(0);
    shape.extend_from_slice(&proto[1..]);
    shape
}

fn ensure_empty_shape(shape: &[usize]) -> Result<(), String> {
    if shape.iter().product::<usize>() != 0 {
        return Err(format!(
            "{LABEL}: at least one dimension must be zero to construct an empty string array"
        ));
    }
    Ok(())
}

fn prototype_dims(proto: &Value) -> Vec<usize> {
    match proto {
        Value::StringArray(sa) => sa.shape.clone(),
        Value::CharArray(ca) => vec![ca.rows, ca.cols],
        Value::Tensor(t) => t.shape.clone(),
        Value::ComplexTensor(t) => t.shape.clone(),
        Value::LogicalArray(l) => l.shape.clone(),
        Value::Cell(cell) => cell.shape.clone(),
        Value::GpuTensor(handle) => handle.shape.clone(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => vec![1, 1],
        Value::String(_) => vec![1, 1],
        _ => vec![1, 1],
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{StringArray, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_is_zero_by_zero() {
        let result = string_empty_builtin(Vec::new()).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 0]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_dimension_creates_zero_by_n() {
        let result = string_empty_builtin(vec![Value::from(5)]).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 5]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn multiple_dimensions_respect_trailing_sizes() {
        let args = vec![Value::from(3), Value::from(4), Value::from(2)];
        let result = string_empty_builtin(args).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 4, 2]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_vector_argument_supported() {
        let tensor = Tensor::new(vec![0.0, 5.0, 3.0], vec![1, 3]).unwrap();
        let result = string_empty_builtin(vec![Value::Tensor(tensor)]).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 5, 3]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_vector_from_nonempty_array_drops_leading_extent() {
        let tensor = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
        let result = string_empty_builtin(vec![Value::Tensor(tensor)]).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 2]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_zero_in_any_position() {
        let args = vec![Value::from(3), Value::from(4), Value::from(0)];
        let result = string_empty_builtin(args).expect("string.empty");
        match result {
            Value::StringArray(sa) => assert_eq!(sa.shape, vec![0, 4, 0]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn like_prototype_without_explicit_dims() {
        let proto = StringArray::new(vec!["alpha".to_string(); 6], vec![2, 3]).unwrap();
        let result = string_empty_builtin(vec![Value::from("like"), Value::StringArray(proto)])
            .expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 3]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn like_prototype_with_scalar_shape() {
        let proto = StringArray::new(vec!["foo".to_string()], vec![1, 1]).unwrap();
        let result = string_empty_builtin(vec![Value::from("like"), Value::StringArray(proto)])
            .expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 1]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn like_with_numeric_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = string_empty_builtin(vec![Value::from("like"), Value::Tensor(tensor)])
            .expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 1]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn like_with_explicit_dims_prefers_dimensions() {
        let proto = StringArray::new(Vec::new(), vec![0, 2]).unwrap();
        let args = vec![
            Value::from(0),
            Value::from(7),
            Value::from("like"),
            Value::StringArray(proto),
        ];
        let result = string_empty_builtin(args).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 7]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn missing_like_prototype_errors() {
        let err = string_empty_builtin(vec![Value::from("like")]).expect_err("expected error");
        assert!(
            err.contains("expected prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn duplicate_like_errors() {
        let proto = StringArray::new(Vec::new(), vec![0, 2]).unwrap();
        let err = string_empty_builtin(vec![
            Value::from("like"),
            Value::StringArray(proto.clone()),
            Value::from("like"),
            Value::StringArray(proto),
        ])
        .expect_err("expected error");
        assert!(err.contains("multiple 'like'"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_dimension_inputs() {
        let err =
            string_empty_builtin(vec![Value::String("oops".into())]).expect_err("expected error");
        assert!(
            err.contains("size inputs must be numeric"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn like_gathers_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=6).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                string_empty_builtin(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
                    .expect("string.empty");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![0, 3]);
                    assert_eq!(sa.data.len(), 0);
                }
                other => panic!("expected string array, got {other:?}"),
            }
            let _ = provider.free(&handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_dimension_arguments_are_gathered() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![0.0, 5.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &dims.data,
                shape: &dims.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                string_empty_builtin(vec![Value::GpuTensor(handle.clone())]).expect("string.empty");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![0, 5, 3]);
                    assert_eq!(sa.data.len(), 0);
                }
                other => panic!("expected string array, got {other:?}"),
            }
            let _ = provider.free(&handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_negative_dimension() {
        let err = string_empty_builtin(vec![Value::from(-1.0)]).expect_err("expected error");
        assert!(
            err.contains("matrix dimensions must be non-negative"),
            "unexpected error: {err}"
        );
    }
}
