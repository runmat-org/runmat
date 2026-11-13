//! MATLAB-compatible `unwrap` builtin for phase-continuity correction.

use runmat_value::NumericDType;
use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, UNWRAP_ERROR_ARG_COUNT, UNWRAP_ERROR_INTERNAL,
    UNWRAP_ERROR_INVALID_DIMENSION, UNWRAP_ERROR_INVALID_INPUT, UNWRAP_ERROR_INVALID_TOLERANCE,
    UNWRAP_INTEGER_PHASE_EXTENSION, UNWRAP_INTEGER_TOLERANCE_EXTENSION,
    UNWRAP_LOGICAL_PHASE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::common::default_dimension;
use crate::builtins::math::signal::common::{SignalCooperation, SIGNAL_WORK_CHUNK};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "unwrap";
const TWO_PI: f64 = 2.0 * PI;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::unwrap")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("phase-unwrap"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Corrects phase discontinuities along one dimension. GPU inputs gather through the active provider until dedicated provider hooks are available.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::unwrap")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Phase unwrapping is stateful along a dimension and is not fusible as an element-wise operation.",
};

fn unwrap_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    unwrap_error_with_message(error.message, error)
}

fn unwrap_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    unwrap_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn unwrap_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn unwrap_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "unwrap",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::unwrap"
)]
async fn unwrap_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    cooperation.completed(0).await?;
    let phase = cooperation
        .prepare_integer_boundary(&value, &UNWRAP_INTEGER_PHASE_EXTENSION, "phase")
        .await?;
    let prepared_phase = phase
        .into_resident_payload()
        .map(|payload| {
            payload.into_value().map_err(|cause| {
                let mut error = unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_INPUT, cause);
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })
        })
        .transpose()?;
    if crate::builtins::common::validation::value_has_logical_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &UNWRAP_LOGICAL_PHASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let prepared_tolerance = if let Some(tolerance) = rest.first() {
        let prepared = cooperation
            .prepare_integer_boundary(tolerance, &UNWRAP_INTEGER_TOLERANCE_EXTENSION, "tolerance")
            .await?;
        prepared
            .into_resident_payload()
            .map(|payload| {
                payload.into_value().map_err(|cause| {
                    let mut error =
                        unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_TOLERANCE, cause);
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })
            })
            .transpose()?
    } else {
        None
    };
    let rest = if let Some(tolerance) = prepared_tolerance {
        let source = rest.into_function_arguments();
        let count = source.len();
        crate::execution::RootedValues::transform_arguments(
            &source,
            count,
            None,
            runtime.as_ref(),
            |index| {
                Ok(if index == 0 {
                    tolerance.clone()
                } else {
                    source.get(index).expect("unwrap argument index").clone()
                })
            },
        )?
        .into()
    } else {
        rest
    };
    let options = parse_arguments(rest.view(), &mut cooperation).await?;
    match prepared_phase.unwrap_or(value) {
        Value::GpuTensor(handle) => unwrap_gpu(handle, options, &mut cooperation).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(unwrap_error_with_detail(
            &UNWRAP_ERROR_INVALID_INPUT,
            "input must be real-valued phase angles",
        )),
        other => unwrap_host(other, options, &mut cooperation).await,
    }
}

#[derive(Clone, Copy, Debug)]
struct UnwrapOptions {
    tolerance: f64,
    dimension: Option<usize>,
}

async fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<UnwrapOptions> {
    match args.len() {
        0 => Ok(UnwrapOptions {
            tolerance: PI,
            dimension: None,
        }),
        1 => Ok(UnwrapOptions {
            tolerance: parse_tolerance(&args[0], cooperation).await?.unwrap_or(PI),
            dimension: None,
        }),
        2 => Ok(UnwrapOptions {
            tolerance: parse_tolerance(&args[0], cooperation).await?.unwrap_or(PI),
            dimension: parse_dimension(&args[1], cooperation).await?,
        }),
        _ => Err(unwrap_error(&UNWRAP_ERROR_ARG_COUNT)),
    }
}

async fn resident_control(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle)
        != runmat_accelerate_api::GpuTensorStorage::Real
        || handle
            .shape()
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            != Some(1)
    {
        let mut error = unwrap_error(error);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    let provider = super::resident_output::owner(handle, || unwrap_error(error))?;
    let host = gpu_helpers::download_value_preserving_residency_async(provider, handle)
        .await
        .map_err(|source| unwrap_error_with_source(error, "control download failed", source))?;
    cooperation.completed(0).await?;
    Ok(Some(host))
}

async fn parse_tolerance(
    value: &Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<f64>> {
    if is_empty_value(value) {
        return Ok(None);
    }
    let host = resident_control(value, &UNWRAP_ERROR_INVALID_TOLERANCE, cooperation).await?;
    let raw = tensor::scalar_f64_from_host_value(host.as_ref().unwrap_or(value))
        .map_err(|detail| unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_TOLERANCE, detail))?
        .ok_or_else(|| unwrap_error(&UNWRAP_ERROR_INVALID_TOLERANCE))?;
    if !raw.is_finite() || raw < 0.0 {
        return Err(unwrap_error_with_detail(
            &UNWRAP_ERROR_INVALID_TOLERANCE,
            format!("got {raw}"),
        ));
    }
    Ok(Some(raw.max(PI)))
}

async fn parse_dimension(
    value: &Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<usize>> {
    if is_empty_value(value) {
        return Ok(None);
    }
    let host = resident_control(value, &UNWRAP_ERROR_INVALID_DIMENSION, cooperation).await?;
    tensor::dimension_from_value_async(host.as_ref().unwrap_or(value), BUILTIN_NAME, false)
        .await
        .map_err(|detail| unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_DIMENSION, detail))?
        .map(Some)
        .ok_or_else(|| unwrap_error(&UNWRAP_ERROR_INVALID_DIMENSION))
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor::tensor_element_len(t) == 0,
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::ComplexTensor(t) => t.materialize_f64().is_empty(),
        _ => false,
    }
}

async fn unwrap_gpu(
    handle: GpuTensorHandle,
    options: UnwrapOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            unwrap_error_with_source(&UNWRAP_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    cooperation.completed(0).await?;
    unwrap_tensor(tensor, options, cooperation)
        .await
        .map(tensor::tensor_into_value)
}

async fn unwrap_host(
    value: Value,
    options: UnwrapOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| unwrap_error_with_detail(&UNWRAP_ERROR_INVALID_INPUT, detail))?;
    unwrap_tensor(tensor, options, cooperation)
        .await
        .map(tensor::tensor_into_value)
}

async fn unwrap_tensor(
    tensor: Tensor,
    options: UnwrapOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Tensor> {
    let input_dtype = tensor.numeric_dtype();
    let output_dtype = match input_dtype {
        NumericDType::F32 => NumericDType::F32,
        _ => NumericDType::F64,
    };
    let mut shape = tensor.shape().to_vec();
    let data = tensor
        .into_numeric_storage()
        .map_err(|err| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, err))?
        .materialize_f64();
    if crate::builtins::common::shape::is_scalar_shape(&shape) {
        shape = crate::builtins::common::shape::normalize_scalar_shape(&shape);
    }
    let dim_index = match options.dimension {
        Some(0) => return Err(unwrap_error(&UNWRAP_ERROR_INVALID_DIMENSION)),
        Some(dim) => dim - 1,
        None => default_dimension(&shape) - 1,
    };

    let mut logical_shape = shape.clone();
    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }

    let len = logical_shape[dim_index];
    if len <= 1 || data.is_empty() {
        cooperation.completed(data.len() as u64).await?;
        return Tensor::new_with_dtype(data, shape, output_dtype)
            .map_err(|err| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, err));
    }

    let inner_stride = checked_product(&logical_shape[..dim_index])?;
    let outer_stride = checked_product(&logical_shape[dim_index + 1..])?;
    let slice_span = len
        .checked_mul(inner_stride)
        .ok_or_else(|| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, "slice span overflow"))?;

    let mut output = data.clone();
    let mut work = 0u64;
    for outer in 0..outer_stride {
        let base = outer.checked_mul(slice_span).ok_or_else(|| {
            unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, "slice offset overflow")
        })?;
        for inner in 0..inner_stride {
            let first_idx = base + inner;
            if first_idx >= data.len() {
                continue;
            }
            let mut correction = 0.0;
            let mut previous = data[first_idx];
            output[first_idx] = previous;
            for k in 1..len {
                let idx = base + inner + k * inner_stride;
                if idx >= data.len() {
                    break;
                }
                let current = data[idx];
                if current.is_finite() && previous.is_finite() {
                    let delta = current - previous;
                    if delta.abs() > options.tolerance {
                        correction += principal_phase_delta(delta) - delta;
                    }
                }
                output[idx] = current + correction;
                previous = current;
                work += 1;
                if work >= SIGNAL_WORK_CHUNK {
                    cooperation.completed(work).await?;
                    work = 0;
                }
            }
        }
    }

    cooperation.completed(work).await?;

    Tensor::new_with_dtype(output, shape, output_dtype)
        .map_err(|err| unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, err))
}

fn checked_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            unwrap_error_with_detail(&UNWRAP_ERROR_INTERNAL, "shape product overflow")
        })
    })
}

fn principal_phase_delta(delta: f64) -> f64 {
    let mut wrapped = (delta + PI).rem_euclid(TWO_PI) - PI;
    if wrapped == -PI && delta > 0.0 {
        wrapped = PI;
    }
    wrapped
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray};

    const TOL: f64 = 1.0e-12;

    fn unwrap_call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(unwrap_builtin(
            None,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn as_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= TOL, "actual={a} expected={e}");
        }
    }

    #[test]
    fn unwrap_row_vector_corrects_two_pi_jump() {
        let input =
            Tensor::new(vec![0.0, 1.0, 2.0, 2.0 - TWO_PI, 3.0 - TWO_PI], vec![1, 5]).unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape(), [1, 5]);
        assert_close(&out.materialize_f64(), &[0.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn unwrap_matrix_defaults_down_columns() {
        let input = Tensor::new(
            vec![0.0, 1.0, 1.0 - TWO_PI, 0.0, -1.0, -1.0 + TWO_PI],
            vec![3, 2],
        )
        .unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape(), [3, 2]);
        assert_close(&out.materialize_f64(), &[0.0, 1.0, 1.0, 0.0, -1.0, -1.0]);
    }

    #[test]
    fn unwrap_dimension_argument_operates_across_rows() {
        let input = Tensor::new(
            vec![0.0, 0.0, 1.0, -1.0, 1.0 - TWO_PI, -1.0 + TWO_PI],
            vec![2, 3],
        )
        .unwrap();
        let out = as_tensor(
            unwrap_call(
                Value::Tensor(input),
                vec![
                    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                    Value::Num(2.0),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.shape(), [2, 3]);
        assert_close(&out.materialize_f64(), &[0.0, 0.0, 1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn unwrap_high_tolerance_preserves_large_jump() {
        let input = Tensor::new(vec![0.0, 1.0, 1.0 - TWO_PI], vec![1, 3]).unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), vec![Value::Num(10.0)]).unwrap());
        assert_eq!(out.shape(), [1, 3]);
        assert_close(&out.materialize_f64(), &[0.0, 1.0, 1.0 - TWO_PI]);
    }

    #[test]
    fn unwrap_nonfinite_values_break_continuity_correction() {
        let input =
            Tensor::new(vec![0.0, f64::NAN, 1.0 - TWO_PI, 2.0 - TWO_PI], vec![1, 4]).unwrap();
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape(), [1, 4]);
        assert_eq!(out.materialize_f64()[0], 0.0);
        assert!(out.materialize_f64()[1].is_nan());
        assert_close(&out.materialize_f64()[2..], &[1.0 - TWO_PI, 2.0 - TWO_PI]);
    }

    #[test]
    fn unwrap_accepts_logical_input() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let out = as_tensor(unwrap_call(Value::LogicalArray(input), Vec::new()).unwrap());
        assert_eq!(out.shape(), [1, 3]);
        assert_close(&out.materialize_f64(), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn unwrap_integer_phase_extension_returns_double_phase_data() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Tensor::new_integer(IntegerStorage::I16(vec![7, 7]), vec![1, 2])
            .expect("integer input");
        let out = as_tensor(unwrap_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(out.materialize_f64(), vec![7.0, 7.0]);
    }

    #[test]
    fn unwrap_integer_phase_and_tolerance_are_gated_but_integer_dimension_is_documented() {
        let compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
        let phase_error = unwrap_call(Value::Int(runmat_value::IntValue::I16(1)), Vec::new())
            .expect_err("typed phase data must be gated");
        assert_eq!(
            phase_error.identifier(),
            UNWRAP_INTEGER_PHASE_EXTENSION.error_identifier
        );
        let tolerance_error = unwrap_call(
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
            vec![Value::Int(runmat_value::IntValue::U8(4))],
        )
        .expect_err("typed tolerance must be gated");
        assert_eq!(
            tolerance_error.identifier(),
            UNWRAP_INTEGER_TOLERANCE_EXTENSION.error_identifier
        );
        let output = as_tensor(
            unwrap_call(
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
                vec![
                    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                    Value::Int(runmat_value::IntValue::U8(2)),
                ],
            )
            .expect("typed dimension is documented"),
        );
        assert_eq!(output.shape(), [1, 2]);
        drop(compatibility);
    }

    #[test]
    fn unwrap_rejects_lossy_wide_integer_phase_data() {
        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = unwrap_call(
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
            Vec::new(),
        )
        .expect_err("lossy phase input must reject");
        assert!(error.message().contains("exactly representable"));
        drop(extensions);
    }

    #[test]
    fn host_phase_loop_observes_invocation_cancellation_before_publication() {
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::num::NonZeroU64;

        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(SIGNAL_WORK_CHUNK).unwrap()),
            source.token(),
            move || {
                source.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let tensor = Tensor::new(
            vec![0.0; SIGNAL_WORK_CHUNK as usize + 2],
            vec![1, SIGNAL_WORK_CHUNK as usize + 2],
        )
        .unwrap();
        let error = block_on(unwrap_tensor(
            tensor,
            UnwrapOptions {
                tolerance: PI,
                dimension: None,
            },
            &mut cooperation,
        ))
        .expect_err("same invocation must observe cancellation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }

    #[test]
    fn unwrap_rejects_complex_input() {
        let input = ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
        let err = unwrap_call(Value::ComplexTensor(input), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:unwrap:InvalidInput"));
    }

    #[test]
    fn unwrap_rejects_invalid_tolerance() {
        let input = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let err = unwrap_call(Value::Tensor(input), vec![Value::Num(f64::INFINITY)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:unwrap:InvalidTolerance"));
    }

    #[test]
    fn unwrap_gpu_input_gathers_and_corrects_phase() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0, 1.0 - TWO_PI], vec![1, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &input.materialize_f64(),
                shape: input.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let out = as_tensor(unwrap_call(Value::GpuTensor(handle), Vec::new()).unwrap());
            assert_eq!(out.shape(), [1, 3]);
            assert_close(&out.materialize_f64(), &[0.0, 1.0, 1.0]);
        });
    }
}
