//! MATLAB-compatible `numel` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::{value_dimensions, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{Tensor, Type, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::numel")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "numel",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Counts elements using tensor metadata; gathers once only if provider metadata is missing.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::numel"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "numel",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner treats this builtin as a host scalar.",
};

fn numel_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("numel").build()
}

fn numel_type(args: &[Type]) -> Type {
    if args.is_empty() {
        Type::Unknown
    } else {
        Type::Int
    }
}

#[runtime_builtin(
    name = "numel",
    category = "array/introspection",
    summary = "Count the number of elements in scalars, vectors, matrices, and N-D arrays.",
    keywords = "numel,number of elements,array length,gpu metadata,dimensions",
    accel = "metadata",
    type_resolver(numel_type),
    builtin_path = "crate::builtins::array::introspection::numel"
)]
async fn numel_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Ok(Value::Num(value_numel(&value).await? as f64));
    }

    let dims = parse_dimension_args(&rest)?;
    let shape = value_dimensions(&value).await?;

    let mut product = 1usize;
    for dim in dims {
        let extent = dimension_extent(&shape, dim);
        product = product.saturating_mul(extent);
    }

    Ok(Value::Num(product as f64))
}

fn parse_dimension_args(args: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for arg in args {
        match arg {
            Value::Int(_) | Value::Num(_) => {
                dims.push(tensor::parse_dimension(arg, "numel").map_err(|e| numel_error(e))?);
            }
            Value::Tensor(t) => {
                ensure_dim_vector(t)?;
                if t.data.is_empty() {
                    return Err(numel_error(
                        "numel: dimension vector must contain at least one element",
                    ));
                }
                let parsed = t
                    .data
                    .iter()
                    .map(|&raw| parse_dim_scalar(raw))
                    .collect::<crate::BuiltinResult<Vec<_>>>()?;
                dims.extend(parsed);
            }
            _ => {
                return Err(numel_error(
                    "numel: dimension arguments must be numeric scalars or vectors",
                ));
            }
        }
    }
    if dims.is_empty() {
        return Err(numel_error(
            "numel: dimension list must contain at least one element",
        ));
    }
    Ok(dims)
}

fn ensure_dim_vector(t: &Tensor) -> crate::BuiltinResult<()> {
    let non_unit = t.shape.iter().filter(|&&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(numel_error(
            "numel: dimension vector must be a vector of positive integers",
        ))
    }
}

fn parse_dim_scalar(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(numel_error("numel: dimension must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(numel_error("numel: dimension must be an integer"));
    }
    if rounded < 1.0 {
        return Err(numel_error("numel: dimension must be >= 1"));
    }
    Ok(rounded as usize)
}

fn dimension_extent(dimensions: &[usize], dim: usize) -> usize {
    dimensions.get(dim.saturating_sub(1)).copied().unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn numel_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::numel_builtin(value, rest))
    }
    use runmat_builtins::{CellArray, CharArray, Tensor};

    #[test]
    fn numel_type_returns_int() {
        assert_eq!(numel_type(&[Type::Tensor { shape: None }]), Type::Int);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_scalar_is_one() {
        let result = numel_builtin(Value::Num(42.0), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_matrix_counts_elements() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = numel_builtin(Value::Tensor(tensor), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_cell_array_counts_cells() {
        let cells = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let cell_array = CellArray::new(cells, 2, 2).unwrap();
        let result = numel_builtin(Value::Cell(cell_array), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_char_array_counts_characters() {
        let chars = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = numel_builtin(Value::CharArray(chars), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_selected_dimensions_multiplies_extents() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let args = vec![Value::from(1.0), Value::from(2.0)];
        let result = numel_builtin(Value::Tensor(tensor), args).expect("numel");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_vector_argument_supported() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result =
            numel_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("numel");
        assert_eq!(result, Value::Num(8.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_gpu_tensor_uses_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = numel_builtin(Value::GpuTensor(handle), Vec::new()).expect("numel");
            assert_eq!(result, Value::Num(12.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_must_be_positive_integer() {
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::from(0.0)])
            .expect_err("expected dimension error");
        assert!(
            err.to_string().contains("dimension must be >= 1"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_vector_requires_vector_shape() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)])
            .expect_err("expected vector shape error");
        assert!(
            err.to_string()
                .contains("dimension vector must be a vector"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numel_dimension_arguments_must_be_numeric() {
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")])
            .expect_err("expected numeric argument error");
        assert!(
            err.to_string()
                .contains("dimension arguments must be numeric"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn numel_wgpu_counts_elements() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0; 18], vec![3, 3, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let result = numel_builtin(Value::GpuTensor(handle), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(18.0));
    }
}
