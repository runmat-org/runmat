//! MATLAB-compatible `cummax` builtin with GPU-aware semantics for RunMat.

use crate::call::{BuiltinArguments, FunctionArgumentView};
use crate::context::RuntimeContext;
use crate::execution::RootedValueLease;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, CumulativeKind, CumulativeMissingMode as CummaxNanMode,
    CumulativeScanDirection as CummaxDirection, CUMMAX_DESCRIPTOR, CUMMAX_ERROR_INTERNAL,
    CUMMAX_ERROR_INVALID_ARGUMENT, CUMMAX_ERROR_INVALID_INPUT,
    CUMMAX_GPU_NANFLAG_EXTENSION as GPU_NANFLAG_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use super::complex_cumulative_extrema;
use super::floating_cumulative_extrema::{
    self, CumulativeDirection, CumulativeExtrema, CumulativeNanMode,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cummax";

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cummax")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cummax",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cummax_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers return cumulative values and optional public double indices. A certified no-attempt permits only admitted automatic host fallback; execution failures are terminal.",
};

fn cummax_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cummax_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cummax_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cummax_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    cummax_error_with_detail(&CUMMAX_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cummax")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cummax",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cummax to the runtime implementation; providers can substitute specialised scan kernels when available.",
};

/// Evaluation artifact returned by `cummax` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct CummaxEvaluation {
    values: Value,
    indices: Value,
}

impl CummaxEvaluation {
    /// Consume the evaluation and return only the running maxima (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both maxima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming the evaluation.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "cummax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::reduction::cummax",
    context = true
)]
async fn cummax_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    let _roots = super::cumulative::admission::roots(
        runtime.as_ref(),
        std::iter::once(&value).chain(rest.iter()),
    )?;
    crate::output_count::validate_bounded_outputs(
        &CUMMAX_DESCRIPTOR,
        &CUMMAX_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let parsed = admitted_controls(&value, rest.view())?;
    let publication =
        super::cumulative::output::Publication::admit(CumulativeKind::Maximum, runtime.as_ref())?;
    let result = evaluate_parsed(value, parsed, runtime.as_ref(), publication.demand()).await?;
    let (values, indices) = result.into_pair();
    publication.finish(values, indices)
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub async fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CummaxEvaluation> {
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
    super::cumulative::admission::input(CumulativeKind::Maximum, value)?;
    let parsed = super::cumulative::options::parse(rest)
        .map_err(|e| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, e))?;
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
) -> BuiltinResult<CummaxEvaluation> {
    match value {
        Value::GpuTensor(handle) => {
            cummax_gpu(
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
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CummaxEvaluation> {
    super::cumulative::admission::host(CumulativeKind::Maximum, runtime)?;
    match value {
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(cummax_internal_error)?;
            host_complex(&tensor, dim.unwrap_or(1), direction, nan_mode, runtime).await
        }
        Value::ComplexTensor(tensor) => {
            let dimension = dim.unwrap_or_else(|| default_dimension_from_shape(tensor.shape()));
            host_complex(&tensor, dimension, direction, nan_mode, runtime).await
        }
        other => cummax_host(other, dim, direction, nan_mode),
    }
}
async fn host_complex(
    tensor: &ComplexTensor,
    dimension: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
    _runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CummaxEvaluation> {
    let (values, indices) = cummax_complex_tensor(tensor, dimension, direction, nan_mode)?;
    Ok(CummaxEvaluation {
        values: complex_tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}
fn cummax_host(
    value: Value,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<CummaxEvaluation> {
    match value {
        Value::Int(value) => {
            let storage =
                crate::builtins::math::reduction::integer_native::storage_from_scalar(&value);
            integer_cummax(&storage, vec![1, 1], dim.unwrap_or(1), direction)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
            integer_cummax(
                tensor.integer_storage().expect("checked integer storage"),
                tensor.shape().to_vec(),
                target_dim,
                direction,
            )
        }
        other => cummax_host_floating(other, dim, direction, nan_mode),
    }
}

fn cummax_host_floating(
    value: Value,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<CummaxEvaluation> {
    let tensor = tensor::value_into_tensor_for("cummax", value)
        .map_err(|err| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_INPUT, err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let (values, indices) = cummax_tensor(tensor, target_dim, direction, nan_mode)?;
    Ok(CummaxEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn integer_cummax(
    storage: &runmat_value::IntegerStorage,
    shape: Vec<usize>,
    dim: usize,
    direction: CummaxDirection,
) -> BuiltinResult<CummaxEvaluation> {
    let result = crate::builtins::math::reduction::integer_native::cumulative_extrema(
        storage,
        &shape,
        dim,
        match direction {
            CummaxDirection::Forward => {
                crate::builtins::math::reduction::integer_native::CumulativeDirection::Forward
            }
            CummaxDirection::Reverse => {
                crate::builtins::math::reduction::integer_native::CumulativeDirection::Reverse
            }
        },
        crate::builtins::math::reduction::integer_native::CumulativeExtremaDirection::Max,
    )
    .map_err(cummax_internal_error)?;
    Ok(CummaxEvaluation {
        values: result.values,
        indices: result.indices,
    })
}

async fn cummax_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
    demand: runmat_accelerate_api::ProviderExtremaDemand,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<CummaxEvaluation> {
    let target = dim.unwrap_or_else(|| default_dimension_from_shape(handle.shape()));
    if target == 0 {
        return Err(cummax_error_with_detail(
            &CUMMAX_ERROR_INVALID_ARGUMENT,
            "dimension must be positive",
        ));
    }
    if target > handle.shape().len() {
        super::cumulative::provider::identity(CumulativeKind::Maximum, &handle, runtime)?;
        super::cumulative::admission::host(CumulativeKind::Maximum, runtime)?;
        let indices = ones_indices(handle.shape())?;
        return Ok(CummaxEvaluation {
            values: Value::GpuTensor(handle),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let result = super::cumulative::provider::execute(
        CumulativeKind::Maximum,
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
                Ok(CummaxEvaluation {
                    values: Value::GpuTensor(values),
                    indices: Value::GpuTensor(indices),
                })
            }
            other => Err(other.reject(super::cumulative::failure(
                CumulativeKind::Maximum,
                "provider returned an invalid scan result family",
            ))),
        };
    }
    let value =
        super::cumulative::provider::fallback(CumulativeKind::Maximum, &handle, runtime).await?;
    host_value(value, Some(target), direction, nan_mode, runtime).await
}

fn cummax_tensor(
    tensor: Tensor,
    dim: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<(Tensor, Tensor)> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(cummax_internal_error)?;
    floating_cumulative_extrema::cumulative_extrema(
        storage,
        shape,
        dim,
        match direction {
            CummaxDirection::Forward => CumulativeDirection::Forward,
            CummaxDirection::Reverse => CumulativeDirection::Reverse,
        },
        match nan_mode {
            CummaxNanMode::Include => CumulativeNanMode::Include,
            CummaxNanMode::Omit => CumulativeNanMode::Omit,
        },
        CumulativeExtrema::Max,
    )
    .map_err(|error| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, error))
}

fn cummax_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<(ComplexTensor, Tensor)> {
    complex_cumulative_extrema::cumulative_extrema(
        tensor.clone().into_complex_storage(),
        tensor.shape().to_vec(),
        dim,
        match direction {
            CummaxDirection::Forward => complex_cumulative_extrema::Direction::Forward,
            CummaxDirection::Reverse => complex_cumulative_extrema::Direction::Reverse,
        },
        match nan_mode {
            CummaxNanMode::Include => complex_cumulative_extrema::NanMode::Include,
            CummaxNanMode::Omit => complex_cumulative_extrema::NanMode::Omit,
        },
        complex_cumulative_extrema::Extrema::Max,
    )
    .map_err(|error| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, error))
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
    Tensor::new(data, shape.to_vec()).map_err(cummax_internal_error)
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

    fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CummaxEvaluation> {
        block_on(super::evaluate(value, rest))
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn cummax_descriptor_signatures_and_errors() {
        let labels: Vec<&str> = CUMMAX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"[M, I] = cummax(A)"));
        assert!(labels.contains(&"[M, I] = cummax(A, dim)"));
        assert!(labels.contains(&"[M, I] = cummax(A, direction)"));
        assert!(labels.contains(&"[M, I] = cummax(A, nanflag)"));
        assert!(labels.contains(&"[M, I] = cummax(A, dim, direction, nanflag)"));
        assert_eq!(labels.len(), 16);
        assert!(CUMMAX_DESCRIPTOR
            .signatures
            .iter()
            .all(|signature| signature.outputs.len() == 2));
        assert_eq!(
            CUMMAX_DESCRIPTOR.output_mode,
            runmat_builtins::BuiltinOutputMode::ByRequestedOutputCount
        );
        assert!(CUMMAX_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMAX_ERROR_INVALID_ARGUMENT.code));
        assert!(CUMMAX_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMAX_ERROR_INVALID_INPUT.code));
        assert!(CUMMAX_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMAX_ERROR_INTERNAL.code));
    }

    #[test]
    fn cummax_rejects_a_third_public_output() {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(super::cummax_builtin(
            None,
            Value::Num(7.0),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:cummax:InvalidArgument")
        );
    }

    #[test]
    fn cummax_complex_single_preserves_native_storage() {
        let input =
            ComplexTensor::from_f32(vec![(3.0, 0.0), (1.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
        let values = evaluate(Value::ComplexTensor(input), &[])
            .expect("cummax")
            .into_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex tensor");
        };
        assert_eq!(
            values.complex_storage(),
            &runmat_value::ComplexStorage::F32(vec![(3.0, 0.0), (3.0, 0.0), (3.0, 0.0)].into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_scalar_returns_value_and_index() {
        let eval = evaluate(Value::Num(7.0), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(7.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[test]
    fn cummax_preserves_native_single_in_reverse() {
        let input = Tensor::from_f32(vec![1.0, 5.0, f32::NAN, 3.0], vec![4, 1]).expect("input");
        let (values, indices) = evaluate(
            Value::Tensor(input),
            &[Value::from("reverse"), Value::from("omitnan")],
        )
        .expect("cummax")
        .into_pair();
        let Value::Tensor(values) = values else {
            panic!("expected values tensor");
        };
        assert_eq!(
            values.into_numeric_storage().expect("native storage"),
            NumericStorage::F32(vec![5.0, 5.0, 3.0, 3.0])
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected indices tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![2.0, 2.0, 4.0, 4.0]);
    }

    #[test]
    fn cummax_integer_storage_and_reverse_indices_remain_exact() {
        let input = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![i64::MIN, -4, -5, -3]),
            vec![2, 2],
        )
        .unwrap();
        let (values, indices) = evaluate(Value::Tensor(input), &[Value::from("reverse")])
            .unwrap()
            .into_pair();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(
                    runmat_value::IntegerStorage::I64(vec![-4, -4, -3, -3]),
                    vec![2, 2]
                )
                .unwrap()
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_matrix_default_dimension() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 3]);
                assert_eq!(out.materialize_f64(), vec![4.0, 4.0, 2.0, 5.0, 7.0, 7.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.shape(), [2, 3]);
                assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_two_tracks_rows() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![4.0, 3.0, 4.0, 5.0, 7.0, 5.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 1.0, 2.0, 3.0, 2.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_parses_typed_integer_dimension_without_mirror() {
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![4, 3, 2, 5]), vec![2, 2]).expect("input");
        let dim = Tensor::new_integer(IntegerStorage::I32(vec![2]), vec![1, 1]).expect("dimension");

        let eval = evaluate(Value::Tensor(input), &[Value::Tensor(dim)])
            .expect("cummax dimension from typed integer tensor");
        let (values, indices) = eval.into_pair();

        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![4, 3, 4, 5]), vec![2, 2])
                    .expect("dimension two values"),
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 2.0], vec![2, 2]).expect("indices"))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_reverse_direction() {
        let tensor = Tensor::new(vec![8.0, 3.0, 6.0, 2.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![8.0, 6.0, 6.0, 2.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 3.0, 3.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_reverse_omitnan_combo() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, f64::NAN, 2.0, 3.0, 4.0], vec![2, 3]).expect("tensor");
        let args = vec![
            Value::Int(IntValue::I32(2)),
            Value::from("reverse"),
            Value::from("omitnan"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![3.0, 5.0, 3.0, 4.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![3.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_placeholder_allows_four_arguments() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, f64::NAN, 2.0, 3.0, 4.0], vec![2, 3]).expect("tensor");
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).expect("placeholder");
        let args = vec![
            Value::Tensor(placeholder),
            Value::Int(IntValue::I32(2)),
            Value::from("reverse"),
            Value::from("omitnan"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![3.0, 5.0, 3.0, 4.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![3.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_omit_nan_behaviour() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert!(out.materialize_f64()[0].is_nan());
                assert_eq!(out.materialize_f64()[1], 5.0);
                assert_eq!(out.materialize_f64()[2], 5.0);
                assert_eq!(out.materialize_f64()[3], 5.0);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert!(idx.materialize_f64()[0].is_nan());
                assert_eq!(idx.materialize_f64()[1], 2.0);
                assert_eq!(idx.materialize_f64()[2], 2.0);
                assert_eq!(idx.materialize_f64()[3], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_duplicate_direction_errors() {
        let err = evaluate(
            Value::Num(1.0),
            &[Value::from("reverse"), Value::from("forward")],
        );
        match err {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMAX_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err.message().contains("direction specified more than once"));
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_duplicate_nanflag_errors() {
        let err = evaluate(
            Value::Num(1.0),
            &[Value::from("omitnan"), Value::from("includenan")],
        );
        match err {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMAX_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err
                    .message()
                    .contains("missing-value handling specified more than once"));
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummax");
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
    fn cummax_dimension_greater_than_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(4))];
        let eval = evaluate(Value::Tensor(tensor.clone()), &args).expect("cummax");
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
    fn cummax_allows_empty_dimension_placeholder() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = [Value::Tensor(placeholder), Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![3.0, 1.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = [Value::Int(IntValue::I32(0))];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected dimension error"),
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMAX_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err
                    .message()
                    .contains(CUMMAX_ERROR_INVALID_ARGUMENT.message));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_reverse_omitnan_combination() {
        let tensor =
            Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN, 3.0], vec![5, 1]).expect("tensor");
        let args = [Value::from("reverse"), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![4.0, 4.0, 3.0, 3.0, 3.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![2.0, 2.0, 5.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_complex_vector() {
        let tensor =
            ComplexTensor::new(vec![(3.0, 0.0), (2.0, 0.0), (2.0, 1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::ComplexTensor(out) => {
                assert_eq!(out.materialize_f64()[0], (3.0, 0.0));
                assert_eq!(out.materialize_f64()[1], (3.0, 0.0));
                assert_eq!(out.materialize_f64()[2], (3.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummax");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.materialize_f64(), vec![4.0, 4.0, 7.0, 7.0]);
            assert_eq!(gathered_indices.materialize_f64(), vec![1.0, 1.0, 3.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_native_integer_gpu_reverse_values_and_indices_stay_resident() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[4, 9, 9, 7]),
                    shape: &[4, 1],
                })
                .expect("upload native integer");
            let eval = evaluate(Value::GpuTensor(handle), &[Value::from("reverse")])
                .expect("cummax reverse");
            let (values, indices) = eval.into_pair();
            let Value::GpuTensor(value_handle) = values else {
                panic!("expected GPU value tensor");
            };
            let Value::GpuTensor(index_handle) = indices else {
                panic!("expected GPU index tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&value_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                block_on(provider.download_integer(&value_handle))
                    .expect("download native integer values")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![9, 9, 9, 7])
            );
            let gathered_indices =
                test_support::gather(Value::GpuTensor(index_handle)).expect("gather indices");
            assert_eq!(gathered_indices.materialize_f64(), vec![3.0, 3.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn cummax_gpu_nanflag_follows_compatibility_mode() {
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
                    Some("RunMat:compatibility:CummaxGpuNanflagExtension")
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
    fn cummax_gpu_dimension_exceeds_rank_returns_indices() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::Int(IntValue::I32(5))];
            let eval = evaluate(Value::GpuTensor(handle), &args).expect("cummax");
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
    fn cummax_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0, 5.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cummax cpu");
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
        let values_only = block_on(cummax_gpu(
            handle.clone(),
            None,
            CummaxDirection::Forward,
            CummaxNanMode::Include,
            runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            None,
        ))
        .expect("resident values-only scan");
        assert!(matches!(values_only.values, Value::GpuTensor(_)));
        let gathered = test_support::gather(values_only.values).expect("gather values-only scan");
        assert_eq!(gathered.materialize_f64(), expected_vals.materialize_f64());

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummax gpu");
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
        let result = block_on(super::cummax_builtin(
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
