//! MATLAB-compatible `cumprod` builtin with GPU-aware semantics for RunMat.

use crate::call::BuiltinArguments;
use crate::context::RuntimeContext;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, CumulativeKind, CumulativeMissingMode as CumprodNanMode,
    CumulativeScanDirection as CumprodDirection, CUMPROD_DESCRIPTOR, CUMPROD_ERROR_INTERNAL,
    CUMPROD_ERROR_INVALID_ARGUMENT, CUMPROD_ERROR_INVALID_INPUT,
    CUMPROD_GPU_NANFLAG_EXTENSION as GPU_NANFLAG_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use super::floating_cumulative_arithmetic::{
    self, CumulativeDirection, CumulativeNanMode, CumulativeOperation,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cumprod";

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cumprod")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cumprod",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cumprod_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The exact input owner may expose a class-preserving scan. Only a certified no-attempt permits admitted automatic host fallback.",
};

fn cumprod_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cumprod_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cumprod_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cumprod_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    cumprod_error_with_detail(&CUMPROD_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cumprod")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cumprod",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cumprod to the runtime implementation; providers can substitute specialised scan kernels.",
};

#[runtime_builtin(
    name = "cumprod",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::cumprod",
    context = true
)]
async fn cumprod_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    let _roots = super::cumulative::admission::roots(
        runtime.as_ref(),
        std::iter::once(&value).chain(rest.iter()),
    )?;
    crate::output_count::validate_fixed_outputs(
        &CUMPROD_DESCRIPTOR,
        &CUMPROD_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    super::cumulative::admission::input(CumulativeKind::Product, &value)?;
    let parsed = super::cumulative::options::parse(rest.view())
        .map_err(|e| cumprod_error_with_detail(&CUMPROD_ERROR_INVALID_ARGUMENT, e))?;
    if matches!(&value, Value::GpuTensor(_)) && parsed.explicit_missing {
        crate::compatibility::ensure_builtin_extension_enabled(&GPU_NANFLAG_EXTENSION, NAME)?;
    }
    match value {
        Value::GpuTensor(handle) => {
            cumprod_gpu(
                handle,
                parsed.dimension,
                parsed.direction,
                parsed.missing,
                runtime.as_ref(),
            )
            .await
        }
        other => {
            host_value(
                other,
                parsed.dimension,
                parsed.direction,
                parsed.missing,
                runtime.as_ref(),
            )
            .await
        }
    }
}

async fn host_value(
    value: Value,
    dim: Option<usize>,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    super::cumulative::admission::host(CumulativeKind::Product, runtime)?;
    match value {
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(cumprod_internal_error)?;
            host_complex(&tensor, dim.unwrap_or(1), direction, nan_mode, runtime).await
        }
        Value::ComplexTensor(tensor) => {
            let dimension = dim.unwrap_or_else(|| default_dimension_from_shape(tensor.shape()));
            host_complex(&tensor, dimension, direction, nan_mode, runtime).await
        }
        other => cumprod_host(other, dim, direction, nan_mode),
    }
}
async fn host_complex(
    tensor: &ComplexTensor,
    dimension: usize,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let options = super::cumulative::options::Options {
        dimension: Some(dimension),
        direction,
        missing: nan_mode,
        explicit_missing: false,
    };
    let result = super::cumulative::complex::cumulative(
        tensor,
        dimension,
        &options,
        CumulativeKind::Product,
        runtime,
    )
    .await?;
    Ok(complex_tensor_into_value(result))
}
fn cumprod_host(
    value: Value,
    dim: Option<usize>,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<Value> {
    match value {
        Value::Int(value) => {
            let storage =
                crate::builtins::math::reduction::integer_native::storage_from_scalar(&value);
            crate::builtins::math::reduction::integer_native::cumulative(
                &storage,
                &[1, 1],
                dim.unwrap_or(1),
                integer_direction(direction),
                crate::builtins::math::reduction::integer_native::CumulativeOperation::Product,
            )
            .map_err(cumprod_internal_error)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
            crate::builtins::math::reduction::integer_native::cumulative(
                tensor.integer_storage().expect("checked integer storage"),
                tensor.shape(),
                target_dim,
                integer_direction(direction),
                crate::builtins::math::reduction::integer_native::CumulativeOperation::Product,
            )
            .map_err(cumprod_internal_error)
        }
        other => cumprod_host_floating(other, dim, direction, nan_mode),
    }
}

fn cumprod_host_floating(
    value: Value,
    dim: Option<usize>,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cumprod", value)
        .map_err(|err| cumprod_error_with_detail(&CUMPROD_ERROR_INVALID_INPUT, err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let result = cumprod_tensor(tensor, target_dim, direction, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

fn integer_direction(
    direction: CumprodDirection,
) -> crate::builtins::math::reduction::integer_native::CumulativeDirection {
    match direction {
        CumprodDirection::Forward => {
            crate::builtins::math::reduction::integer_native::CumulativeDirection::Forward
        }
        CumprodDirection::Reverse => {
            crate::builtins::math::reduction::integer_native::CumulativeDirection::Reverse
        }
    }
}

async fn cumprod_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let target = dim.unwrap_or_else(|| default_dimension_from_shape(handle.shape()));
    if target == 0 {
        return Err(cumprod_error_with_detail(
            &CUMPROD_ERROR_INVALID_ARGUMENT,
            "dimension must be positive",
        ));
    }
    if target > handle.shape().len() {
        super::cumulative::provider::identity(CumulativeKind::Product, &handle, runtime)?;
        return Ok(Value::GpuTensor(handle));
    }
    let result = super::cumulative::provider::execute(
        CumulativeKind::Product,
        &handle,
        target - 1,
        (direction, nan_mode),
        runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
        runtime,
    )
    .await?;
    if let Some(result) = result {
        return match result {
            super::cumulative::provider::ResultValue::Values(value) => {
                gpu_helpers::resident_gpu_value(value)
            }
            other => Err(other.reject(super::cumulative::failure(
                CumulativeKind::Product,
                "provider returned an invalid scan result family",
            ))),
        };
    }
    let value =
        super::cumulative::provider::fallback(CumulativeKind::Product, &handle, runtime).await?;
    let result = host_value(value, Some(target), direction, nan_mode, runtime).await?;
    gpu_helpers::restore_class_preserving_value(&handle, result, NAME)
}

fn cumprod_tensor(
    tensor: Tensor,
    dim: usize,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(cumprod_internal_error)?;
    floating_cumulative_arithmetic::cumulative(
        storage,
        shape,
        dim,
        match direction {
            CumprodDirection::Forward => CumulativeDirection::Forward,
            CumprodDirection::Reverse => CumulativeDirection::Reverse,
        },
        match nan_mode {
            CumprodNanMode::Include => CumulativeNanMode::Include,
            CumprodNanMode::Omit => CumulativeNanMode::Omit,
        },
        CumulativeOperation::Product,
    )
    .map_err(|error| cumprod_error_with_detail(&CUMPROD_ERROR_INVALID_ARGUMENT, error))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if let Some([value]) = tensor.as_f64_slice() {
        Value::Complex(value.0, value.1)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn default_dimension(tensor: &Tensor) -> usize {
    default_dimension_from_shape(tensor.shape())
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor as BuiltinsTensor};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    fn cumprod_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cumprod_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn cumprod_descriptor_signatures_and_errors() {
        let labels: Vec<&str> = CUMPROD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"B = cumprod(A)"));
        assert!(labels.contains(&"B = cumprod(A, dim)"));
        assert!(labels.contains(&"B = cumprod(A, direction)"));
        assert!(labels.contains(&"B = cumprod(A, nanflag)"));
        assert!(labels.contains(&"B = cumprod(A, dim, direction)"));
        assert!(labels.contains(&"B = cumprod(A, direction, dim)"));
        assert!(labels.contains(&"B = cumprod(A, dim, nanflag)"));
        assert!(labels.contains(&"B = cumprod(A, nanflag, dim)"));
        assert!(labels.contains(&"B = cumprod(A, direction, nanflag)"));
        assert!(labels.contains(&"B = cumprod(A, nanflag, direction)"));
        assert!(labels.contains(&"B = cumprod(A, dim, direction, nanflag)"));
        assert!(labels.contains(&"B = cumprod(A, nanflag, direction, dim)"));
        assert!(CUMPROD_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMPROD_ERROR_INVALID_ARGUMENT.code));
        assert!(CUMPROD_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMPROD_ERROR_INVALID_INPUT.code));
        assert!(CUMPROD_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMPROD_ERROR_INTERNAL.code));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_scalar_num() {
        let result = cumprod_builtin(Value::Num(7.0), Vec::new()).expect("cumprod scalar");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn cumprod_preserves_native_single_with_omitnan() {
        let input = BuiltinsTensor::from_f32(vec![f32::NAN, 2.0, 3.0], vec![3, 1]).expect("input");
        let result =
            cumprod_builtin(Value::Tensor(input), vec![Value::from("omitnan")]).expect("cumprod");
        let Value::Tensor(result) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.into_numeric_storage().expect("native storage"),
            NumericStorage::F32(vec![1.0, 2.0, 6.0])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_native_integer_preserves_storage_for_dimensions_and_reverse() {
        let input = BuiltinsTensor::new_integer(
            runmat_value::IntegerStorage::I8(vec![2, 100, 3, 2]),
            vec![2, 2],
        )
        .expect("input");
        assert_eq!(
            cumprod_builtin(Value::Tensor(input.clone()), Vec::new()).expect("default cumprod"),
            Value::Tensor(
                BuiltinsTensor::new_integer(
                    runmat_value::IntegerStorage::I8(vec![2, 127, 3, 6]),
                    vec![2, 2],
                )
                .expect("default output"),
            )
        );
        assert_eq!(
            cumprod_builtin(
                Value::Tensor(input),
                vec![
                    Value::Num(2.0),
                    Value::from("reverse"),
                    Value::from("omitnan")
                ],
            )
            .expect("reverse cumprod"),
            Value::Tensor(
                BuiltinsTensor::new_integer(
                    runmat_value::IntegerStorage::I8(vec![6, 127, 3, 2]),
                    vec![2, 2],
                )
                .expect("reverse output"),
            )
        );
    }

    #[test]
    fn cumprod_parses_typed_integer_dimension_without_mirror() {
        let input = BuiltinsTensor::new_integer(IntegerStorage::I16(vec![2, 3, 4, 5]), vec![2, 2])
            .expect("input");
        let dim = BuiltinsTensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1])
            .expect("dimension");

        let result = cumprod_builtin(Value::Tensor(input), vec![Value::Tensor(dim)])
            .expect("cumprod dimension from typed integer tensor");

        assert_eq!(
            result,
            Value::Tensor(
                BuiltinsTensor::new_integer(IntegerStorage::I16(vec![2, 3, 8, 15]), vec![2, 2],)
                    .expect("dimension two output"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_native_integer_scalar_and_empty_retains_class() {
        assert_eq!(
            cumprod_builtin(Value::Int(IntValue::U64(u64::MAX)), Vec::new()).expect("scalar"),
            Value::Int(IntValue::U64(u64::MAX))
        );
        let empty =
            BuiltinsTensor::new_integer(runmat_value::IntegerStorage::U32(Vec::new()), vec![0, 1])
                .expect("empty input");
        assert_eq!(
            cumprod_builtin(Value::Tensor(empty), Vec::new()).expect("empty cumprod"),
            Value::Tensor(
                BuiltinsTensor::new_integer(
                    runmat_value::IntegerStorage::U32(Vec::new()),
                    vec![0, 1]
                )
                .expect("empty output"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_matrix_default_dimension() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), Vec::new()).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![1.0, 4.0, 2.0, 10.0, 3.0, 18.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_matrix_dimension_two() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = cumprod_builtin(Value::Tensor(tensor), args).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![1.0, 4.0, 2.0, 20.0, 6.0, 120.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_reverse_direction() {
        let tensor = BuiltinsTensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![4, 1]).unwrap();
        let result =
            cumprod_builtin(Value::Tensor(tensor), vec![Value::from("reverse")]).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![120.0, 60.0, 20.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_omit_nan_forward() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![4, 1]).expect("tensor");
        let result =
            cumprod_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 2.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_include_nan_propagates() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), Vec::new()).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert!((out.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!(out.materialize_f64()[1].is_nan());
                assert!(out.materialize_f64()[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_dim_greater_than_ndims_returns_input() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = cumprod_builtin(Value::Tensor(tensor.clone()), args).expect("cumprod");
        match result {
            Value::Tensor(out) => assert_eq!(out, tensor),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_complex_tensor() {
        let data = vec![(1.0, 2.0), (3.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 1]).unwrap();
        let result = cumprod_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("cumprod");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert!((out.materialize_f64()[0].0 - 1.0).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - 2.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1].0 - 5.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1].1 - 5.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cumprod_builtin(Value::GpuTensor(handle), Vec::new()).expect("cumprod");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 6.0, 24.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_reverse_with_omit_nan() {
        let tensor =
            BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).expect("tensor");
        let result = cumprod_builtin(
            Value::Tensor(tensor),
            vec![Value::from("omitnan"), Value::from("reverse")],
        )
        .expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![8.0, 8.0, 8.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_accepts_omitmissing_synonym() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 2.0, 3.0, f64::NAN], vec![4, 1]).expect("tensor");
        let result = cumprod_builtin(Value::Tensor(tensor), vec![Value::from("omitmissing")])
            .expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 6.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_accepts_includemissing_synonym() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")])
            .expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert!((out.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!(out.materialize_f64()[1].is_nan());
                assert!(out.materialize_f64()[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_dimension_placeholder_is_ignored() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let placeholder = BuiltinsTensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::from("reverse")];
        let result = cumprod_builtin(Value::Tensor(tensor), args).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![4.0, 4.0, 10.0, 5.0, 18.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_dimension_zero_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]);
        match result {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMPROD_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(
                    err.message()
                        .contains(CUMPROD_ERROR_INVALID_ARGUMENT.message),
                    "unexpected result: {err}"
                );
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_native_integer_gpu_reverse_omitnan_stays_resident() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U8(&[20, 4, 5, 2]),
                    shape: &[2, 2],
                })
                .expect("upload native integer");
            let result = cumprod_builtin(
                Value::GpuTensor(handle),
                vec![
                    Value::Int(IntValue::I32(1)),
                    Value::from("reverse"),
                    Value::from("omitnan"),
                ],
            )
            .expect("cumprod");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident GPU tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out),
                Some(runmat_accelerate_api::IntegerElementType::U8)
            );
            assert_eq!(
                block_on(provider.download_integer(&out))
                    .expect("download native integer result")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::U8(vec![80, 4, 10, 2])
            );
        });
    }

    #[test]
    fn cumprod_gpu_nanflag_follows_compatibility_mode() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 3.0], vec![2, 1]).expect("input");
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: tensor.shape(),
                })
                .expect("upload");
            let args = vec![Value::from("omitnan")];
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(super::cumprod_builtin(
                    None,
                    Value::GpuTensor(handle.clone()),
                    crate::builtins::common::test_support::builtin_arguments(args.clone()),
                ))
                .expect_err("MATLAB-compatible mode");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:CumprodGpuNanflagExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                block_on(super::cumprod_builtin(
                    None,
                    Value::GpuTensor(handle),
                    crate::builtins::common::test_support::builtin_arguments(args),
                ))
                .expect("RunMat extension mode");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumprod_wgpu_forward_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();

        let cpu = cumprod_host(
            Value::Tensor(tensor.clone()),
            Some(1),
            CumprodDirection::Forward,
            CumprodNanMode::Include,
        )
        .unwrap();

        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(cumprod_gpu(
            handle,
            Some(1),
            CumprodDirection::Forward,
            CumprodNanMode::Include,
            None,
        ))
        .expect("cumprod gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape(), ct.shape());
                assert_eq!(gathered.materialize_f64(), ct.materialize_f64());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumprod_wgpu_reverse_omitnan_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).unwrap();

        let cpu = cumprod_host(
            Value::Tensor(tensor.clone()),
            Some(1),
            CumprodDirection::Reverse,
            CumprodNanMode::Omit,
        )
        .unwrap();

        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(cumprod_gpu(
            handle,
            Some(1),
            CumprodDirection::Reverse,
            CumprodNanMode::Omit,
            None,
        ))
        .expect("cumprod gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape(), ct.shape());
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gathered
                    .materialize_f64()
                    .iter()
                    .zip(ct.materialize_f64().iter())
                {
                    if b.is_nan() {
                        assert!(a.is_nan());
                    } else {
                        assert!((a - b).abs() < tol);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn canonical_complex_single_prefixes_do_not_widen_or_copy_input() {
        let input =
            ComplexTensor::from_f32(vec![(1., 1.), (2., 0.), (-1., 1.)], vec![3, 1]).unwrap();
        let pointer = input.as_f32_slice().unwrap().as_ptr();
        let output = block_on(super::cumprod_builtin(
            None,
            Value::ComplexTensor(input.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert_eq!(input.as_f32_slice().unwrap().as_ptr(), pointer);
        let Value::ComplexTensor(output) = output else {
            panic!("complex tensor output")
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape(), &[3, 1]);
        assert_eq!(
            output.as_f32_slice().unwrap(),
            &[
                runmat_value::ComplexElement(1., 1.),
                runmat_value::ComplexElement(2., 2.),
                runmat_value::ComplexElement(-4., 0.)
            ]
        );
    }
}
