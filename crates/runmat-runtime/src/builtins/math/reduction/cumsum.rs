//! MATLAB-compatible `cumsum` builtin with GPU-aware semantics for RunMat.

use crate::call::BuiltinArguments;
use crate::context::RuntimeContext;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, CumulativeKind, CumulativeMissingMode as CumsumNanMode,
    CumulativeScanDirection as CumsumDirection, CUMSUM_DESCRIPTOR, CUMSUM_ERROR_INTERNAL,
    CUMSUM_ERROR_INVALID_ARGUMENT, CUMSUM_ERROR_INVALID_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use super::floating_cumulative_arithmetic::{
    self, CumulativeDirection, CumulativeNanMode, CumulativeOperation,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cumsum";

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cumsum")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cumsum",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cumsum_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The exact input owner may expose a class-preserving scan. Only a certified no-attempt permits admitted automatic host fallback.",
};

fn cumsum_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cumsum_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cumsum_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cumsum_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    cumsum_error_with_detail(&CUMSUM_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cumsum")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cumsum",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cumsum to the runtime implementation; providers can substitute specialised scan kernels.",
};

#[runtime_builtin(
    name = "cumsum",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::cumsum",
    context = true
)]
async fn cumsum_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    let _roots = super::cumulative::admission::roots(
        runtime.as_ref(),
        std::iter::once(&value).chain(rest.iter()),
    )?;
    crate::output_count::validate_fixed_outputs(
        &CUMSUM_DESCRIPTOR,
        &CUMSUM_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    super::cumulative::admission::input(CumulativeKind::Sum, &value)?;
    let parsed = super::cumulative::options::parse(rest.view())
        .map_err(|e| cumsum_error_with_detail(&CUMSUM_ERROR_INVALID_ARGUMENT, e))?;

    match value {
        Value::GpuTensor(handle) => {
            cumsum_gpu(
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
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    super::cumulative::admission::host(CumulativeKind::Sum, runtime)?;
    match value {
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(cumsum_internal_error)?;
            host_complex(&tensor, dim.unwrap_or(1), direction, nan_mode, runtime).await
        }
        Value::ComplexTensor(tensor) => {
            let dimension = dim.unwrap_or_else(|| default_dimension_from_shape(tensor.shape()));
            host_complex(&tensor, dimension, direction, nan_mode, runtime).await
        }
        other => cumsum_host(other, dim, direction, nan_mode),
    }
}
async fn host_complex(
    tensor: &ComplexTensor,
    dimension: usize,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
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
        CumulativeKind::Sum,
        runtime,
    )
    .await?;
    Ok(complex_tensor_into_value(result))
}
fn cumsum_host(
    value: Value,
    dim: Option<usize>,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
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
                crate::builtins::math::reduction::integer_native::CumulativeOperation::Sum,
            )
            .map_err(cumsum_internal_error)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
            crate::builtins::math::reduction::integer_native::cumulative(
                tensor.integer_storage().expect("checked integer storage"),
                tensor.shape(),
                target_dim,
                integer_direction(direction),
                crate::builtins::math::reduction::integer_native::CumulativeOperation::Sum,
            )
            .map_err(cumsum_internal_error)
        }
        other => cumsum_host_floating(other, dim, direction, nan_mode),
    }
}

fn cumsum_host_floating(
    value: Value,
    dim: Option<usize>,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cumsum", value)
        .map_err(|err| cumsum_error_with_detail(&CUMSUM_ERROR_INVALID_INPUT, err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let result = cumsum_tensor(tensor, target_dim, direction, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

fn integer_direction(
    direction: CumsumDirection,
) -> crate::builtins::math::reduction::integer_native::CumulativeDirection {
    match direction {
        CumsumDirection::Forward => {
            crate::builtins::math::reduction::integer_native::CumulativeDirection::Forward
        }
        CumsumDirection::Reverse => {
            crate::builtins::math::reduction::integer_native::CumulativeDirection::Reverse
        }
    }
}

async fn cumsum_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let target = dim.unwrap_or_else(|| default_dimension_from_shape(handle.shape()));
    if target == 0 {
        return Err(cumsum_error_with_detail(
            &CUMSUM_ERROR_INVALID_ARGUMENT,
            "dimension must be positive",
        ));
    }
    if target > handle.shape().len() {
        super::cumulative::provider::identity(CumulativeKind::Sum, &handle, runtime)?;
        return Ok(Value::GpuTensor(handle));
    }
    let result = super::cumulative::provider::execute(
        CumulativeKind::Sum,
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
                CumulativeKind::Sum,
                "provider returned an invalid scan result family",
            ))),
        };
    }
    let value =
        super::cumulative::provider::fallback(CumulativeKind::Sum, &handle, runtime).await?;
    let result = host_value(value, Some(target), direction, nan_mode, runtime).await?;
    gpu_helpers::restore_class_preserving_value(&handle, result, NAME)
}

fn cumsum_tensor(
    tensor: Tensor,
    dim: usize,
    direction: CumsumDirection,
    nan_mode: CumsumNanMode,
) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(cumsum_internal_error)?;
    floating_cumulative_arithmetic::cumulative(
        storage,
        shape,
        dim,
        match direction {
            CumsumDirection::Forward => CumulativeDirection::Forward,
            CumsumDirection::Reverse => CumulativeDirection::Reverse,
        },
        match nan_mode {
            CumsumNanMode::Include => CumulativeNanMode::Include,
            CumsumNanMode::Omit => CumulativeNanMode::Omit,
        },
        CumulativeOperation::Sum,
    )
    .map_err(|error| cumsum_error_with_detail(&CUMSUM_ERROR_INVALID_ARGUMENT, error))
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

pub(in crate::builtins::math::reduction) fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

pub(in crate::builtins::math) use super::cumulative::geometry::{
    scan_geometry, ScanGeometry, ScanRows,
};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericStorage,
        Tensor as BuiltinsTensor,
    };

    #[test]
    fn cumsum_complex_scalar_finalizer_keeps_typed_integer_storage_without_mirror() {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::U32(vec![11]), IntegerStorage::U32(vec![5]))
                .expect("matching complex integer storage");
        let input = ComplexTensor::new_integer(storage.clone(), vec![1, 1])
            .expect("typed complex integer input");

        let value = complex_tensor_into_value(input);
        let Value::ComplexTensor(output) = value else {
            panic!("typed complex integer scalar must not collapse to double complex");
        };
        assert_eq!(output.integer_storage().cloned(), Some(storage));
    }

    fn cumsum_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cumsum_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn cumsum_descriptor_signatures_and_errors() {
        let labels: Vec<&str> = CUMSUM_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"B = cumsum(A)"));
        assert!(labels.contains(&"B = cumsum(A, dim)"));
        assert!(labels.contains(&"B = cumsum(A, direction)"));
        assert!(labels.contains(&"B = cumsum(A, nanflag)"));
        assert!(labels.contains(&"B = cumsum(A, dim, direction)"));
        assert!(labels.contains(&"B = cumsum(A, direction, dim)"));
        assert!(labels.contains(&"B = cumsum(A, dim, nanflag)"));
        assert!(labels.contains(&"B = cumsum(A, nanflag, dim)"));
        assert!(labels.contains(&"B = cumsum(A, direction, nanflag)"));
        assert!(labels.contains(&"B = cumsum(A, nanflag, direction)"));
        assert!(labels.contains(&"B = cumsum(A, dim, direction, nanflag)"));
        assert!(labels.contains(&"B = cumsum(A, nanflag, direction, dim)"));
        assert!(CUMSUM_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMSUM_ERROR_INVALID_ARGUMENT.code));
        assert!(CUMSUM_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMSUM_ERROR_INVALID_INPUT.code));
        assert!(CUMSUM_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMSUM_ERROR_INTERNAL.code));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_scalar_num() {
        let result = cumsum_builtin(Value::Num(7.0), Vec::new()).expect("cumsum scalar");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn cumsum_preserves_native_single_with_reverse_omitnan() {
        let input = BuiltinsTensor::from_f32(vec![1.0, f32::NAN, 3.0], vec![3, 1]).expect("input");
        let result = cumsum_builtin(
            Value::Tensor(input),
            vec![Value::from("reverse"), Value::from("omitnan")],
        )
        .expect("cumsum");
        let Value::Tensor(result) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.into_numeric_storage().expect("native storage"),
            NumericStorage::F32(vec![4.0, 3.0, 3.0])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_native_integer_preserves_storage_for_dimensions_and_reverse() {
        let input = BuiltinsTensor::new_integer(
            runmat_value::IntegerStorage::U8(vec![250, 10, 2, 3]),
            vec![2, 2],
        )
        .expect("input");
        assert_eq!(
            cumsum_builtin(Value::Tensor(input.clone()), Vec::new()).expect("default cumsum"),
            Value::Tensor(
                BuiltinsTensor::new_integer(
                    runmat_value::IntegerStorage::U8(vec![250, 255, 2, 5]),
                    vec![2, 2],
                )
                .expect("default output"),
            )
        );
        assert_eq!(
            cumsum_builtin(
                Value::Tensor(input),
                vec![
                    Value::Num(2.0),
                    Value::from("reverse"),
                    Value::from("omitnan")
                ],
            )
            .expect("reverse cumsum"),
            Value::Tensor(
                BuiltinsTensor::new_integer(
                    runmat_value::IntegerStorage::U8(vec![252, 13, 2, 3]),
                    vec![2, 2],
                )
                .expect("reverse output"),
            )
        );
    }

    #[test]
    fn cumsum_parses_typed_integer_dimension_without_mirror() {
        let input = BuiltinsTensor::new_integer(IntegerStorage::I16(vec![1, 4, 2, 5]), vec![2, 2])
            .expect("input");
        let dim = BuiltinsTensor::new_integer(IntegerStorage::I32(vec![2]), vec![1, 1])
            .expect("dimension");

        let result = cumsum_builtin(Value::Tensor(input), vec![Value::Tensor(dim)])
            .expect("cumsum dimension from typed integer tensor");

        assert_eq!(
            result,
            Value::Tensor(
                BuiltinsTensor::new_integer(IntegerStorage::I16(vec![1, 4, 3, 9]), vec![2, 2],)
                    .expect("dimension two output"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_native_integer_scalar_and_empty_retains_class() {
        assert_eq!(
            cumsum_builtin(Value::Int(IntValue::I64(i64::MIN)), Vec::new()).expect("scalar"),
            Value::Int(IntValue::I64(i64::MIN))
        );
        let empty =
            BuiltinsTensor::new_integer(runmat_value::IntegerStorage::U32(Vec::new()), vec![0, 1])
                .expect("empty input");
        assert_eq!(
            cumsum_builtin(Value::Tensor(empty), Vec::new()).expect("empty cumsum"),
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
    fn cumsum_matrix_default_dimension() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), Vec::new()).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![1.0, 5.0, 2.0, 7.0, 3.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_matrix_dimension_two() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = cumsum_builtin(Value::Tensor(tensor), args).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![1.0, 4.0, 3.0, 9.0, 6.0, 15.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_reverse_direction() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result =
            cumsum_builtin(Value::Tensor(tensor), vec![Value::from("reverse")]).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![10.0, 9.0, 7.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_omit_nan_forward() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 1.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let result =
            cumsum_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_include_nan_propagates() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), Vec::new()).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert!(out.materialize_f64()[1].is_nan());
                assert!(out.materialize_f64()[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_dim_greater_than_ndims_returns_input() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = cumsum_builtin(Value::Tensor(tensor.clone()), args).expect("cumsum");
        match result {
            Value::Tensor(out) => assert_eq!(out, tensor),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_complex_tensor() {
        let data = vec![(1.0, 2.0), (3.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 1]).unwrap();
        let result = cumsum_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("cumsum");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [2, 1]);
                assert!((out.materialize_f64()[0].0 - 1.0).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - 2.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1].0 - 4.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1].1 - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cumsum_builtin(Value::GpuTensor(handle), Vec::new()).expect("cumsum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), [4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 6.0, 10.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_reverse_with_omit_nan() {
        let tensor =
            BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).expect("tensor");
        let result = cumsum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("omitnan"), Value::from("reverse")],
        )
        .expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![7.0, 6.0, 6.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_accepts_omitmissing_synonym() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 2.0, 3.0, f64::NAN], vec![4, 1]).expect("tensor");
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::from("omitmissing")])
            .expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![0.0, 2.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_accepts_includemissing_synonym() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")])
            .expect("cumsum");
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
    fn cumsum_dimension_placeholder_is_ignored() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let placeholder = BuiltinsTensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::from("reverse")];
        let result = cumsum_builtin(Value::Tensor(tensor), args).expect("cumsum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![5.0, 4.0, 7.0, 5.0, 9.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_dimension_zero_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]);
        match result {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMSUM_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(
                    err.message()
                        .contains(CUMSUM_ERROR_INVALID_ARGUMENT.message),
                    "unexpected result: {err}"
                );
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_duplicate_direction_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumsum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("forward"), Value::from("reverse")],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_unknown_option_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumsum_builtin(Value::Tensor(tensor), vec![Value::from("bogus")]);
        match result {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMSUM_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(
                    err.message().contains("unrecognised option"),
                    "unexpected result: {err}"
                );
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_complex_omit_nan() {
        let data = vec![(1.0, 0.5), (f64::NAN, 2.0), (3.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![3, 1]).unwrap();
        let result = cumsum_builtin(Value::ComplexTensor(tensor), vec![Value::from("omitnan")])
            .expect("cumsum");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), [3, 1]);
                assert!((out.materialize_f64()[0].0 - 1.0).abs() < 1e-12);
                assert!((out.materialize_f64()[0].1 - 0.5).abs() < 1e-12);
                assert!((out.materialize_f64()[1].0 - 1.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1].1 - 0.5).abs() < 1e-12);
                assert!((out.materialize_f64()[2].0 - 4.0).abs() < 1e-12);
                assert!((out.materialize_f64()[2].1 - (-0.5)).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_gpu_omit_nan_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![f64::NAN, 2.0, 3.0, f64::NAN], vec![4, 1])
                .expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cumsum_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")])
                .expect("cumsum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![0.0, 2.0, 5.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_gpu_dimension_exceeds_rank_returns_handle() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let original_id = handle.buffer_id();
            let result = cumsum_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(3))],
            )
            .expect("cumsum");
            match result {
                Value::GpuTensor(out) => {
                    assert_eq!(out.shape(), tensor.shape());
                    assert_eq!(out.buffer_id(), original_id);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumsum_native_integer_gpu_reverse_omitnan_stays_resident() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I8(&[120, 10, 2, 4]),
                    shape: &[2, 2],
                })
                .expect("upload native integer");
            let result = cumsum_builtin(
                Value::GpuTensor(handle),
                vec![
                    Value::Int(IntValue::I32(1)),
                    Value::from("reverse"),
                    Value::from("omitnan"),
                ],
            )
            .expect("cumsum");
            let Value::GpuTensor(out) = result else {
                panic!("expected resident GPU tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&out),
                Some(runmat_accelerate_api::IntegerElementType::I8)
            );
            assert_eq!(
                block_on(provider.download_integer(&out))
                    .expect("download native integer result")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::I8(vec![127, 10, 6, 4])
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumsum_wgpu_matches_cpu_forward_dim1() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let cpu = cumsum_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu_result = cumsum_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let gathered = test_support::gather(gpu_result).expect("gather gpu");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape(), gathered.shape());
                for (a, b) in ct
                    .materialize_f64()
                    .iter()
                    .zip(gathered.materialize_f64().iter())
                {
                    assert!((a - b).abs() < 1e-9);
                }
            }
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumsum_wgpu_reverse_omitnan_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).unwrap();
        let cpu = cumsum_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("reverse"), Value::from("omitnan")],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).unwrap();
        let gpu = cumsum_builtin(
            Value::GpuTensor(handle),
            vec![Value::from("reverse"), Value::from("omitnan")],
        )
        .unwrap();
        let gathered = test_support::gather(gpu).expect("gather gpu");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape(), gathered.shape());
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in ct
                    .materialize_f64()
                    .iter()
                    .zip(gathered.materialize_f64().iter())
                {
                    assert!((a - b).abs() < tol);
                }
            }
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[test]
    fn canonical_complex_single_prefixes_do_not_widen_or_copy_input() {
        let input =
            ComplexTensor::from_f32(vec![(1., 1.), (2., 0.), (-1., 1.)], vec![3, 1]).unwrap();
        let pointer = input.as_f32_slice().unwrap().as_ptr();
        let output = block_on(super::cumsum_builtin(
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
                runmat_value::ComplexElement(3., 1.),
                runmat_value::ComplexElement(2., 2.)
            ]
        );
    }
}
