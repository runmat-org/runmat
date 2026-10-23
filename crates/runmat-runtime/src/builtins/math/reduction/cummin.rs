//! MATLAB-compatible `cummin` builtin with GPU-aware semantics for RunMat.

use crate::call::{BuiltinArguments, FunctionArgumentView};
use crate::context::RuntimeContext;
use crate::execution::RootedValueLease;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, CumulativeKind, CumulativeMissingMode as CumminNanMode,
    CumulativeScanDirection as CumminDirection, CUMMIN_DESCRIPTOR, CUMMIN_ERROR_INTERNAL,
    CUMMIN_ERROR_INVALID_ARGUMENT, CUMMIN_ERROR_INVALID_INPUT,
    CUMMIN_GPU_NANFLAG_EXTENSION as GPU_NANFLAG_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use super::complex_cumulative_extrema;
use super::floating_cumulative_extrema::{
    self, CumulativeDirection, CumulativeExtrema, CumulativeNanMode,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cummin";

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cummin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cummin",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cummin_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers return cumulative values and optional public double indices. A certified no-attempt permits only admitted automatic host fallback; execution failures are terminal.",
};

fn cummin_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cummin_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cummin_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cummin_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    cummin_error_with_detail(&CUMMIN_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cummin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cummin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cummin to the runtime implementation; providers can substitute specialised scan kernels when available.",
};

/// Evaluation artifact returned by `cummin` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct CumminEvaluation {
    values: Value,
    indices: Value,
}

impl CumminEvaluation {
    /// Consume the evaluation and return only the running minima (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both minima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming the evaluation.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "cummin",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::cummin",
    context = true
)]
async fn cummin_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    let _roots = super::cumulative::admission::roots(
        runtime.as_ref(),
        std::iter::once(&value).chain(rest.iter()),
    )?;
    crate::output_count::validate_bounded_outputs(
        &CUMMIN_DESCRIPTOR,
        &CUMMIN_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let parsed = admitted_controls(&value, rest.view())?;
    let publication =
        super::cumulative::output::Publication::admit(CumulativeKind::Minimum, runtime.as_ref())?;
    let result = evaluate_parsed(value, parsed, runtime.as_ref(), publication.demand()).await?;
    let (values, indices) = result.into_pair();
    publication.finish(values, indices)
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub async fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CumminEvaluation> {
    let _roots = RootedValueLease::for_values(std::iter::once(&value).chain(rest.iter()))
        .map_err(|e| e.into_runtime_error())?;
    let parsed = admitted_controls(&value, FunctionArgumentView::from_slice(rest))?;
    evaluate_parsed(
        value,
        parsed,
        None,
        runmat_accelerate_api::ProviderExtremaDemand::ValuesAndIndices,
    )
    .await
}
fn admitted_controls(
    value: &Value,
    rest: FunctionArgumentView<'_>,
) -> BuiltinResult<super::cumulative::options::Options> {
    super::cumulative::admission::input(CumulativeKind::Minimum, value)?;
    let parsed = super::cumulative::options::parse(rest)
        .map_err(|e| cummin_error_with_detail(&CUMMIN_ERROR_INVALID_ARGUMENT, e))?;
    if matches!(value, Value::GpuTensor(_)) && parsed.explicit_missing {
        crate::compatibility::ensure_builtin_extension_enabled(&GPU_NANFLAG_EXTENSION, NAME)?;
    }
    Ok(parsed)
}
async fn evaluate_parsed(
    value: Value,
    parsed: super::cumulative::options::Options,
    runtime: Option<&RuntimeContext>,
    demand: runmat_accelerate_api::ProviderExtremaDemand,
) -> BuiltinResult<CumminEvaluation> {
    match value {
        Value::GpuTensor(handle) => {
            cummin_gpu(
                handle,
                parsed.dimension,
                parsed.direction,
                parsed.missing,
                demand,
                runtime,
            )
            .await
        }
        other => {
            host_value(
                other,
                parsed.dimension,
                parsed.direction,
                parsed.missing,
                runtime,
            )
            .await
        }
    }
}

async fn host_value(
    value: Value,
    dim: Option<usize>,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CumminEvaluation> {
    super::cumulative::admission::host(CumulativeKind::Minimum, runtime)?;
    match value {
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(cummin_internal_error)?;
            host_complex(&tensor, dim.unwrap_or(1), direction, nan_mode, runtime).await
        }
        Value::ComplexTensor(tensor) => {
            let dimension = dim.unwrap_or_else(|| default_dimension_from_shape(tensor.shape()));
            host_complex(&tensor, dimension, direction, nan_mode, runtime).await
        }
        other => cummin_host(other, dim, direction, nan_mode),
    }
}
async fn host_complex(
    tensor: &ComplexTensor,
    dimension: usize,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
    _runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CumminEvaluation> {
    let (values, indices) = cummin_complex_tensor(tensor, dimension, direction, nan_mode)?;
    Ok(CumminEvaluation {
        values: complex_tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}
fn cummin_host(
    value: Value,
    dim: Option<usize>,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<CumminEvaluation> {
    match value {
        Value::Int(value) => {
            let storage =
                crate::builtins::math::reduction::integer_native::storage_from_scalar(&value);
            integer_cummin(&storage, vec![1, 1], dim.unwrap_or(1), direction)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
            integer_cummin(
                tensor.integer_storage().expect("checked integer storage"),
                tensor.shape().to_vec(),
                target_dim,
                direction,
            )
        }
        other => cummin_host_floating(other, dim, direction, nan_mode),
    }
}

fn cummin_host_floating(
    value: Value,
    dim: Option<usize>,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<CumminEvaluation> {
    let tensor = tensor::value_into_tensor_for("cummin", value)
        .map_err(|err| cummin_error_with_detail(&CUMMIN_ERROR_INVALID_INPUT, err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let (values, indices) = cummin_tensor(tensor, target_dim, direction, nan_mode)?;
    Ok(CumminEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn integer_cummin(
    storage: &runmat_value::IntegerStorage,
    shape: Vec<usize>,
    dim: usize,
    direction: CumminDirection,
) -> BuiltinResult<CumminEvaluation> {
    let result = crate::builtins::math::reduction::integer_native::cumulative_extrema(
        storage,
        &shape,
        dim,
        match direction {
            CumminDirection::Forward => {
                crate::builtins::math::reduction::integer_native::CumulativeDirection::Forward
            }
            CumminDirection::Reverse => {
                crate::builtins::math::reduction::integer_native::CumulativeDirection::Reverse
            }
        },
        crate::builtins::math::reduction::integer_native::CumulativeExtremaDirection::Min,
    )
    .map_err(cummin_internal_error)?;
    Ok(CumminEvaluation {
        values: result.values,
        indices: result.indices,
    })
}

async fn cummin_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
    demand: runmat_accelerate_api::ProviderExtremaDemand,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CumminEvaluation> {
    let target = dim.unwrap_or_else(|| default_dimension_from_shape(handle.shape()));
    if target == 0 {
        return Err(cummin_error_with_detail(
            &CUMMIN_ERROR_INVALID_ARGUMENT,
            "dimension must be positive",
        ));
    }
    if target > handle.shape().len() {
        super::cumulative::provider::identity(CumulativeKind::Minimum, &handle, runtime)?;
        super::cumulative::admission::host(CumulativeKind::Minimum, runtime)?;
        let indices = ones_indices(handle.shape())?;
        return Ok(CumminEvaluation {
            values: Value::GpuTensor(handle),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let result = super::cumulative::provider::execute(
        CumulativeKind::Minimum,
        &handle,
        target - 1,
        (direction, nan_mode),
        demand,
        runtime,
    )
    .await?;
    if let Some(result) = result {
        return match result {
            super::cumulative::provider::ResultValue::Pair { values, indices } => {
                Ok(CumminEvaluation {
                    values: Value::GpuTensor(values),
                    indices: Value::GpuTensor(indices),
                })
            }
            other => Err(other.reject(super::cumulative::failure(
                CumulativeKind::Minimum,
                "provider returned an invalid scan result family",
            ))),
        };
    }
    let value =
        super::cumulative::provider::fallback(CumulativeKind::Minimum, &handle, runtime).await?;
    host_value(value, Some(target), direction, nan_mode, runtime).await
}

fn cummin_tensor(
    tensor: Tensor,
    dim: usize,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<(Tensor, Tensor)> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(cummin_internal_error)?;
    floating_cumulative_extrema::cumulative_extrema(
        storage,
        shape,
        dim,
        match direction {
            CumminDirection::Forward => CumulativeDirection::Forward,
            CumminDirection::Reverse => CumulativeDirection::Reverse,
        },
        match nan_mode {
            CumminNanMode::Include => CumulativeNanMode::Include,
            CumminNanMode::Omit => CumulativeNanMode::Omit,
        },
        CumulativeExtrema::Min,
    )
    .map_err(|error| cummin_error_with_detail(&CUMMIN_ERROR_INVALID_ARGUMENT, error))
}

fn cummin_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CumminDirection,
    nan_mode: CumminNanMode,
) -> BuiltinResult<(ComplexTensor, Tensor)> {
    complex_cumulative_extrema::cumulative_extrema(
        tensor.clone().into_complex_storage(),
        tensor.shape().to_vec(),
        dim,
        match direction {
            CumminDirection::Forward => complex_cumulative_extrema::Direction::Forward,
            CumminDirection::Reverse => complex_cumulative_extrema::Direction::Reverse,
        },
        match nan_mode {
            CumminNanMode::Include => complex_cumulative_extrema::NanMode::Include,
            CumminNanMode::Omit => complex_cumulative_extrema::NanMode::Omit,
        },
        complex_cumulative_extrema::Extrema::Min,
    )
    .map_err(|error| cummin_error_with_detail(&CUMMIN_ERROR_INVALID_ARGUMENT, error))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if let Some([value]) = tensor.as_f64_slice() {
        Value::Complex(value.0, value.1)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn ones_indices(shape: &[usize]) -> BuiltinResult<Tensor> {
    let len = tensor::element_count(shape);
    let data = if len == 0 {
        Vec::new()
    } else {
        vec![1.0f64; len]
    };
    Tensor::new(data, shape.to_vec()).map_err(cummin_internal_error)
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
    use runmat_value::{IntValue, IntegerStorage, NumericStorage};

    fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CumminEvaluation> {
        block_on(super::evaluate(value, rest))
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn cummin_descriptor_signatures_and_errors() {
        let labels: Vec<&str> = CUMMIN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"[M, I] = cummin(A)"));
        assert!(labels.contains(&"[M, I] = cummin(A, dim)"));
        assert!(labels.contains(&"[M, I] = cummin(A, direction)"));
        assert!(labels.contains(&"[M, I] = cummin(A, nanflag)"));
        assert!(labels.contains(&"[M, I] = cummin(A, dim, direction, nanflag)"));
        assert_eq!(labels.len(), 16);
        assert!(CUMMIN_DESCRIPTOR
            .signatures
            .iter()
            .all(|signature| signature.outputs.len() == 2));
        assert_eq!(
            CUMMIN_DESCRIPTOR.output_mode,
            runmat_builtins::BuiltinOutputMode::ByRequestedOutputCount
        );
        assert!(CUMMIN_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMIN_ERROR_INVALID_ARGUMENT.code));
        assert!(CUMMIN_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMIN_ERROR_INVALID_INPUT.code));
        assert!(CUMMIN_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMIN_ERROR_INTERNAL.code));
    }

    #[test]
    fn cummin_rejects_a_third_public_output() {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(super::cummin_builtin(
            None,
            Value::Num(7.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:cummin:InvalidArgument")
        );
    }

    #[test]
    fn cummin_complex_single_preserves_native_storage() {
        let input =
            ComplexTensor::from_f32(vec![(1.0, 0.0), (3.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
        let values = evaluate(Value::ComplexTensor(input), &[])
            .expect("cummin")
            .into_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex tensor");
        };
        assert_eq!(
            values.complex_storage(),
            &runmat_value::ComplexStorage::F32(vec![(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)].into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_scalar_returns_value_and_index() {
        let eval = evaluate(Value::Num(7.0), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(7.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[test]
    fn cummin_preserves_native_single_with_omitnan() {
        let input =
            Tensor::from_f32(vec![f32::NAN, 5.0, f32::NAN, 3.0], vec![4, 1]).expect("input");
        let (values, indices) = evaluate(Value::Tensor(input), &[Value::from("omitnan")])
            .expect("cummin")
            .into_pair();
        let Value::Tensor(values) = values else {
            panic!("expected values tensor");
        };
        let NumericStorage::F32(values) = values.into_numeric_storage().expect("native storage")
        else {
            panic!("expected native single values");
        };
        assert!(values[0].is_nan());
        assert_eq!(&values[1..], &[5.0, 5.0, 3.0]);
        let Value::Tensor(indices) = indices else {
            panic!("expected indices tensor");
        };
        assert!(indices.materialize_f64()[0].is_nan());
        assert_eq!(&indices.materialize_f64()[1..], &[2.0, 2.0, 4.0]);
    }

    #[test]
    fn cummin_integer_storage_and_indices_remain_exact() {
        let input = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 4, 5, 3]),
            vec![2, 2],
        )
        .unwrap();
        let (values, indices) = evaluate(Value::Tensor(input), &[]).unwrap().into_pair();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(
                    runmat_value::IntegerStorage::U64(vec![u64::MAX, 4, 5, 3]),
                    vec![2, 2]
                )
                .unwrap()
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_matrix_default_dimension() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![4.0, 3.0, 2.0, 2.0, 7.0, 1.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.shape(), [2, 3]);
                assert_eq!(idx.materialize_f64(), vec![1.0, 2.0, 1.0, 1.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_dimension_two_tracks_rows() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![4.0, 3.0, 2.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 2.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummin_parses_typed_integer_dimension_without_mirror() {
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![4, 3, 2, 5]), vec![2, 2]).expect("input");
        let dim = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("dimension");

        let eval = evaluate(Value::Tensor(input), &[Value::Tensor(dim)])
            .expect("cummin dimension from typed integer tensor");
        let (values, indices) = eval.into_pair();

        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![4, 3, 2, 3]), vec![2, 2])
                    .expect("dimension two values"),
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 2.0, 1.0], vec![2, 2]).expect("indices"))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_reverse_direction() {
        let tensor = Tensor::new(vec![8.0, 3.0, 6.0, 2.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![4.0, 4.0, 4.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_omit_nan_behaviour() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert!(out.materialize_f64()[0].is_nan());
                assert_eq!(out.materialize_f64()[1], 5.0);
                assert_eq!(out.materialize_f64()[2], 5.0);
                assert_eq!(out.materialize_f64()[3], 3.0);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert!(idx.materialize_f64()[0].is_nan());
                assert_eq!(idx.materialize_f64()[1], 2.0);
                assert_eq!(idx.materialize_f64()[2], 2.0);
                assert_eq!(idx.materialize_f64()[3], 4.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64()[0], 1.0);
                assert!(out.materialize_f64()[1].is_nan());
                assert!(out.materialize_f64()[2].is_nan());
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64()[0], 1.0);
                assert_eq!(idx.materialize_f64()[1], 2.0);
                assert_eq!(idx.materialize_f64()[2], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_dimension_greater_than_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let eval = evaluate(Value::Tensor(tensor.clone()), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), tensor.materialize_f64()),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert!(idx.materialize_f64().iter().all(|v| *v == 1.0)),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_allows_empty_dimension_placeholder() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = [Value::Tensor(placeholder), Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.0, 1.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![2.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = [Value::Int(IntValue::I32(0))];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected dimension error"),
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMIN_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err
                    .message()
                    .contains(CUMMIN_ERROR_INVALID_ARGUMENT.message));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_duplicate_direction_errors() {
        let tensor = Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap();
        let args = [Value::from("reverse"), Value::from("forward")];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected duplicate direction error"),
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMIN_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err.message().contains("direction specified more than once"));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_reverse_omitnan_combination() {
        let tensor =
            Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN, 3.0], vec![5, 1]).expect("tensor");
        let args = [Value::from("reverse"), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![2.0, 2.0, 2.0, 3.0, 3.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![3.0, 3.0, 3.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_complex_vector() {
        let tensor =
            ComplexTensor::new(vec![(3.0, 0.0), (2.0, 0.0), (2.0, 1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("cummin");
        let (values, indices) = eval.into_pair();
        match values {
            Value::ComplexTensor(out) => {
                assert_eq!(out.materialize_f64()[0], (3.0, 0.0));
                assert_eq!(out.materialize_f64()[1], (2.0, 0.0));
                assert_eq!(out.materialize_f64()[2], (2.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 2.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummin");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.materialize_f64(), vec![4.0, 2.0, 2.0, 1.0]);
            assert_eq!(gathered_indices.materialize_f64(), vec![1.0, 2.0, 2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_native_integer_gpu_values_and_indices_stay_resident() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I64(&[4, 2, 2, 7]),
                    shape: &[4, 1],
                })
                .expect("upload native integer");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummin");
            let (values, indices) = eval.into_pair();
            let Value::GpuTensor(value_handle) = values else {
                panic!("expected GPU value tensor");
            };
            let Value::GpuTensor(index_handle) = indices else {
                panic!("expected GPU index tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&value_handle),
                Some(runmat_accelerate_api::IntegerElementType::I64)
            );
            assert_eq!(
                block_on(provider.download_integer(&value_handle))
                    .expect("download native integer values")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::I64(vec![4, 2, 2, 2])
            );
            let gathered_indices =
                test_support::gather(Value::GpuTensor(index_handle)).expect("gather indices");
            assert_eq!(gathered_indices.materialize_f64(), vec![1.0, 2.0, 2.0, 2.0]);
        });
    }

    #[test]
    fn cummin_gpu_nanflag_follows_compatibility_mode() {
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
                let error = evaluate(Value::GpuTensor(handle.clone()), &args)
                    .expect_err("MATLAB-compatible mode");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:CumminGpuNanflagExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                evaluate(Value::GpuTensor(handle), &args).expect("RunMat extension mode");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummin_gpu_dimension_exceeds_rank_returns_indices() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::Int(IntValue::I32(5))];
            let eval = evaluate(Value::GpuTensor(handle), &args).expect("cummin");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.materialize_f64(), tensor.materialize_f64());
            assert!(gathered_indices.materialize_f64().iter().all(|v| *v == 1.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cummin_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0, 5.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cummin cpu");
        let (cpu_vals, cpu_idx) = cpu_eval.into_pair();
        let expected_vals = match cpu_vals {
            Value::Tensor(t) => t,
            other => panic!("expected tensor values from cpu eval, got {other:?}"),
        };
        let expected_idx = match cpu_idx {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices from cpu eval, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let values_only = block_on(cummin_gpu(
            handle.clone(),
            None,
            CumminDirection::Forward,
            CumminNanMode::Include,
            runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            None,
        ))
        .expect("resident values-only scan");
        assert!(matches!(values_only.values, Value::GpuTensor(_)));
        let gathered = test_support::gather(values_only.values).expect("gather values-only scan");
        assert_eq!(gathered.materialize_f64(), expected_vals.materialize_f64());

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummin gpu");
        let (gpu_vals, gpu_idx) = gpu_eval.into_pair();

        match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => {
                assert!(matches!(
                    (&gpu_vals, &gpu_idx),
                    (Value::GpuTensor(_), Value::GpuTensor(_))
                ));
            }
            runmat_accelerate_api::ProviderPrecision::F32 => {
                // This backend cannot publish double indices. Its two-output
                // route declines before execution; automatic input gathers.
                let (Value::Tensor(values), Value::Tensor(indices)) = (&gpu_vals, &gpu_idx) else {
                    panic!("expected class-preserving host fallback");
                };
                assert_eq!(values.numeric_dtype(), runmat_value::NumericDType::F32);
                assert_eq!(indices.numeric_dtype(), runmat_value::NumericDType::F64);
            }
        }

        let gathered_vals = test_support::gather(gpu_vals).expect("gather values");
        let gathered_idx = test_support::gather(gpu_idx).expect("gather indices");

        assert_eq!(gathered_vals.shape(), expected_vals.shape());
        assert_eq!(
            gathered_vals.materialize_f64(),
            expected_vals.materialize_f64()
        );
        assert_eq!(gathered_idx.shape(), expected_idx.shape());
        assert_eq!(
            gathered_idx.materialize_f64(),
            expected_idx.materialize_f64()
        );
    }

    #[test]
    fn canonical_second_output_is_real_double() {
        let _guard = crate::output_count::push_output_count(Some(2)).unwrap();
        let result = block_on(super::cummin_builtin(
            None,
            Value::Num(7.),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::OutputList(values) = result else {
            panic!("requested output list")
        };
        assert_eq!(values.as_slice(), &[Value::Num(7.), Value::Num(1.)]);
    }
}
