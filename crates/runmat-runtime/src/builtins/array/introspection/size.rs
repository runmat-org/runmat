//! MATLAB-compatible `size` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::{dims_to_row_tensor, value_dimensions};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::size")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "size",
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
        "Reads dimension metadata from tensor handles; no kernels or provider hooks are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::introspection::size")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "size",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner bypasses this builtin.",
};

fn size_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("size").build()
}

#[runtime_builtin(
    name = "size",
    category = "array/introspection",
    summary = "Get the dimensions of scalars, vectors, matrices, and N-D arrays.",
    keywords = "size,dimensions,shape,gpu,introspection",
    builtin_path = "crate::builtins::array::introspection::size"
)]
async fn size_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let dims = value_dimensions(&value).await?;
    match rest.len() {
        0 => dimensions_to_value(&dims),
        1 => match parse_dim_selection(&rest[0])? {
            DimSelection::Single(dim) => {
                let extent = dimension_extent(&dims, dim);
                Ok(Value::Num(extent as f64))
            }
            DimSelection::Multiple(dimensions) => {
                let extents: Vec<usize> = dimensions
                    .into_iter()
                    .map(|dim| dimension_extent(&dims, dim))
                    .collect();
                dimensions_to_value(&extents)
            }
        },
        _ => Err(size_error("size: too many input arguments")),
    }
}

fn dimensions_to_value(dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = dims_to_row_tensor(dimensions)
        .map_err(|e| size_error(format!("size: failed to build output: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

enum DimSelection {
    Single(usize),
    Multiple(Vec<usize>),
}

fn parse_dim_selection(arg: &Value) -> crate::BuiltinResult<DimSelection> {
    match arg {
        Value::Int(_) | Value::Num(_) => {
            let dim = tensor::parse_dimension(arg, "size").map_err(|e| size_error(e))?;
            Ok(DimSelection::Single(dim))
        }
        Value::Tensor(t) => {
            ensure_dim_vector(t)?;
            if t.data.is_empty() {
                return Err(size_error(
                    "size: dimension vector must contain at least one element",
                ));
            }
            let dims = t
                .data
                .iter()
                .map(|&raw| parse_dim_scalar(raw))
                .collect::<crate::BuiltinResult<Vec<_>>>()?;
            Ok(DimSelection::Multiple(dims))
        }
        _ => Err(size_error(
            "size: dimension argument must be a numeric scalar or vector",
        )),
    }
}

fn ensure_dim_vector(t: &Tensor) -> crate::BuiltinResult<()> {
    let non_unit_dims = t.shape.iter().filter(|&&dim| dim > 1).count();
    if non_unit_dims <= 1 {
        Ok(())
    } else {
        Err(size_error(
            "size: dimension vector must be a vector of positive integers",
        ))
    }
}

fn parse_dim_scalar(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(size_error("size: dimension must be finite"));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err(size_error("size: dimension must be an integer"));
    }
    if rounded < 1.0 {
        return Err(size_error("size: dimension must be >= 1"));
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

    fn size_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::size_builtin(value, rest))
    }
    use runmat_builtins::Tensor;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_matrix_returns_row_vector() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = size_builtin(Value::Tensor(tensor), Vec::new()).expect("size");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_with_dimension_scalar_returns_extent() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let result = size_builtin(Value::Tensor(tensor), vec![Value::from(1.0)]).expect("size dim");
        match result {
            Value::Num(v) => assert_eq!(v, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_with_dimension_vector_returns_row_vector() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let dims_arg = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result = size_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims_arg)])
            .expect("size dims vector");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![2.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_gpu_tensor_uses_handle_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0; 8], vec![2, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = size_builtin(Value::GpuTensor(handle), Vec::new()).expect("size gpu");
            match result {
                Value::Tensor(out) => {
                    assert_eq!(out.shape, vec![1, 2]);
                    assert_eq!(out.data, vec![2.0, 4.0]);
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn size_wgpu_preserves_shape_metadata() {
        struct EnvGuard(Option<String>);
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                match &self.0 {
                    Some(prev) => std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", prev.as_str()),
                    None => std::env::remove_var("RUNMAT_WGPU_FORCE_PRECISION"),
                }
            }
        }
        let previous = std::env::var("RUNMAT_WGPU_FORCE_PRECISION").ok();
        std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
        let _guard = EnvGuard(previous);

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };

        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload to device");

        let result = size_builtin(Value::GpuTensor(handle), Vec::new()).expect("size");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_rejects_non_numeric_dimension() {
        let err = size_builtin(Value::Num(1.0), vec![Value::from("dim")]).unwrap_err();
        assert!(
            err.to_string().contains("dimension argument"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_dimension_beyond_rank_returns_one() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = size_builtin(Value::Tensor(tensor), vec![Value::from(5.0)]).expect("size dim");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_dimension_vector_requires_positive_integers() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.5], vec![1, 2]).unwrap();
        let err = size_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)])
            .expect_err("non-int dim");
        assert!(err.to_string().contains("dimension must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_dimension_vector_must_not_be_matrix() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = size_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)])
            .expect_err("matrix dims");
        assert!(err
            .to_string()
            .contains("dimension vector must be a vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn size_dimension_vector_must_not_be_empty() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![], vec![1, 0]).unwrap();
        let err =
            size_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect_err("empty dims");
        assert!(err
            .to_string()
            .contains("must contain at least one element"));
    }
}
