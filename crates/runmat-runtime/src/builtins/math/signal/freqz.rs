//! Focused MATLAB-compatible `freqz` digital filter response.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, FREQZ_ERROR_ARG_COUNT,
    FREQZ_ERROR_INTERNAL, FREQZ_ERROR_INVALID_COEFFICIENTS, FREQZ_ERROR_INVALID_FS,
    FREQZ_ERROR_INVALID_N, FREQZ_INTEGER_COEFFICIENT_EXTENSION, FREQZ_INTEGER_FS_EXTENSION,
    FREQZ_INTEGER_N_EXTENSION, FREQZ_LOGICAL_COEFFICIENT_EXTENSION, FREQZ_LOGICAL_FS_EXTENSION,
    FREQZ_LOGICAL_N_EXTENSION, FREQZ_RESIDENT_INPUT_EXTENSION, FREQZ_SINGLE_FS_EXTENSION,
    FREQZ_SINGLE_N_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    parse_nonnegative_integer, parse_scalar_f64, value_to_complex_vector, SignalCooperation,
    SIGNAL_WORK_CHUNK,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(feature = "plot-core")]
use runmat_builtins::FREQZ_ERROR_PLOT;

const BUILTIN_NAME: &str = "freqz";

#[cfg(feature = "plot-core")]
mod presentation;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::freqz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "freqz",
    op_kind: GpuOpKind::Custom("frequency-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Interactive resident inputs are a RunMat-only extension; after compatibility preflight they gather through the owning provider and produce host outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::freqz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "freqz",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "freqz materialises response vectors and is not fused.",
};

fn freqz_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    freqz_error_with_message(error.message, error)
}

fn freqz_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    freqz_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn freqz_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, source.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn freqz_error_with_message(
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
    name = "freqz",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::freqz"
)]
async fn freqz_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    b: Value,
    a: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate_in_runtime(runtime.as_ref(), b, a, rest.view()).await
}

pub async fn evaluate(b: Value, a: Value, rest: &[Value]) -> BuiltinResult<Value> {
    evaluate_in_runtime(
        None,
        b,
        a,
        crate::call::FunctionArgumentView::from_slice(rest),
    )
    .await
}

async fn evaluate_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    b: Value,
    a: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    evaluate_with_cooperation(b, a, rest, &mut cooperation).await
}

async fn evaluate_with_cooperation(
    b: Value,
    a: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    cooperation.completed(0).await?;
    if rest.len() > 2 {
        return Err(freqz_error(&FREQZ_ERROR_ARG_COUNT));
    }
    let plot_requested = matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none());
    #[cfg(not(feature = "plot-core"))]
    if plot_requested {
        return Err(crate::plotting_hooks::unavailable_presentation());
    }
    ensure_input_extensions(&b, &a, rest)?;
    let single_response = is_single_value(&b) || is_single_value(&a);
    let b = gather_freqz_value(b).await?;
    let a = gather_freqz_value(a).await?;
    let rest = gather_freqz_values(rest).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&a, BUILTIN_NAME)?;
    for value in &rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)?;
    }
    ensure_exact_integer_boundary(&b, "numerator", &FREQZ_ERROR_INVALID_COEFFICIENTS)?;
    ensure_exact_integer_boundary(&a, "denominator", &FREQZ_ERROR_INVALID_COEFFICIENTS)?;
    let b = value_to_complex_vector(BUILTIN_NAME, "numerator", b)
        .await
        .map_err(|err| freqz_error_with_source(&FREQZ_ERROR_INVALID_COEFFICIENTS, err))?
        .data;
    let a = value_to_complex_vector(BUILTIN_NAME, "denominator", a)
        .await
        .map_err(|err| freqz_error_with_source(&FREQZ_ERROR_INVALID_COEFFICIENTS, err))?
        .data;
    if b.is_empty() || a.is_empty() {
        return Err(freqz_error_with_detail(
            &FREQZ_ERROR_INVALID_COEFFICIENTS,
            "coefficient vectors cannot be empty",
        ));
    }
    let n = if let Some(value) = rest.first() {
        if is_empty_value(value) {
            512
        } else {
            let parsed = parse_nonnegative_integer(BUILTIN_NAME, "n", value)
                .map_err(|err| freqz_error_with_source(&FREQZ_ERROR_INVALID_N, err))?;
            if parsed < 2 {
                return Err(freqz_error(&FREQZ_ERROR_INVALID_N));
            }
            parsed
        }
    } else {
        512
    };
    let fs = if let Some(value) = rest.get(1) {
        ensure_exact_integer_boundary(value, "fs", &FREQZ_ERROR_INVALID_FS)?;
        let fs = parse_scalar_f64(BUILTIN_NAME, "fs", value)
            .map_err(|err| freqz_error_with_source(&FREQZ_ERROR_INVALID_FS, err))?;
        if fs <= 0.0 {
            return Err(freqz_error(&FREQZ_ERROR_INVALID_FS));
        }
        Some(fs)
    } else {
        None
    };

    let eval = evaluate_response(&b, &a, n, fs, single_response, cooperation).await?;
    #[cfg(feature = "plot-core")]
    if plot_requested {
        presentation::plot_response(&eval, fs.is_some())?;
        return Ok(Value::output_list(Vec::new()));
    }
    #[cfg(not(feature = "plot-core"))]
    let _ = plot_requested;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.h_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.h_value()?, eval.w_value()?],
        ));
    }
    eval.h_value()
}

struct FreqzEvaluation {
    h: Vec<Complex<f64>>,
    w: Vec<f64>,
    single_response: bool,
}

impl FreqzEvaluation {
    fn h_value(&self) -> BuiltinResult<Value> {
        let storage = if self.single_response {
            ComplexStorage::F32(self.h.iter().map(|z| (z.re as f32, z.im as f32)).collect())
        } else {
            ComplexStorage::F64(self.h.iter().map(|z| (z.re, z.im)).collect())
        };
        ComplexTensor::from_complex_storage(storage, vec![self.h.len(), 1])
            .map(Value::ComplexTensor)
            .map_err(|e| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, e))
    }

    fn w_value(&self) -> BuiltinResult<Value> {
        Tensor::new(self.w.clone(), vec![self.w.len(), 1])
            .map(Value::Tensor)
            .map_err(|e| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, e))
    }
}

async fn evaluate_response(
    b: &[Complex<f64>],
    a: &[Complex<f64>],
    n: usize,
    fs: Option<f64>,
    single_response: bool,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<FreqzEvaluation> {
    let mut h = Vec::new();
    h.try_reserve_exact(n)
        .map_err(|cause| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, cause.to_string()))?;
    let mut w = Vec::new();
    w.try_reserve_exact(n)
        .map_err(|cause| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, cause.to_string()))?;
    for idx in 0..n {
        let omega = std::f64::consts::PI * idx as f64 / n as f64;
        let point = Complex::new(omega.cos(), -omega.sin());
        let numerator = polynomial_in_z_inverse(b, point, cooperation).await?;
        let denominator = polynomial_in_z_inverse(a, point, cooperation).await?;
        h.push(numerator / denominator);
        w.push(match fs {
            Some(fs) => fs * idx as f64 / (2.0 * n as f64),
            None => omega,
        });
        if (idx + 1) % SIGNAL_WORK_CHUNK as usize == 0 {
            cooperation.completed(SIGNAL_WORK_CHUNK).await?;
        }
    }
    cooperation
        .completed((n % SIGNAL_WORK_CHUNK as usize) as u64)
        .await?;
    Ok(FreqzEvaluation {
        h,
        w,
        single_response,
    })
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn ensure_extension_if(
    condition: bool,
    extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if condition {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    Ok(())
}

fn ensure_input_extensions(
    b: &Value,
    a: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for coefficient in [b, a] {
        ensure_extension_if(
            is_typed_integer_value(coefficient),
            &FREQZ_INTEGER_COEFFICIENT_EXTENSION,
        )?;
        ensure_extension_if(
            is_logical_value(coefficient),
            &FREQZ_LOGICAL_COEFFICIENT_EXTENSION,
        )?;
    }
    if let Some(n) = rest.first().filter(|value| !is_empty_value(value)) {
        ensure_extension_if(is_typed_integer_value(n), &FREQZ_INTEGER_N_EXTENSION)?;
        ensure_extension_if(is_logical_value(n), &FREQZ_LOGICAL_N_EXTENSION)?;
        ensure_extension_if(is_single_value(n), &FREQZ_SINGLE_N_EXTENSION)?;
    }
    if let Some(fs) = rest.get(1) {
        ensure_extension_if(is_typed_integer_value(fs), &FREQZ_INTEGER_FS_EXTENSION)?;
        ensure_extension_if(is_logical_value(fs), &FREQZ_LOGICAL_FS_EXTENSION)?;
        ensure_extension_if(is_single_value(fs), &FREQZ_SINGLE_FS_EXTENSION)?;
    }
    ensure_extension_if(
        std::iter::once(b)
            .chain(std::iter::once(a))
            .chain(rest.iter())
            .any(|value| matches!(value, Value::GpuTensor(_))),
        &FREQZ_RESIDENT_INPUT_EXTENSION,
    )
}

async fn gather_freqz_value(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::builtins::common::gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| freqz_error_with_source(&FREQZ_ERROR_INTERNAL, err))
    } else {
        Ok(value)
    }
}

async fn gather_freqz_values(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(values.len());
    for value in values {
        gathered.push(gather_freqz_value(value.clone()).await?);
    }
    Ok(gathered)
}

fn is_empty_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.is_empty())
}

fn ensure_exact_integer_boundary(
    value: &Value,
    role: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    let representable = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if representable {
        Ok(())
    } else {
        Err(freqz_error_with_detail(
            error,
            format!("integer {role} values must be exactly representable as double"),
        ))
    }
}

async fn polynomial_in_z_inverse(
    coeffs: &[Complex<f64>],
    point: Complex<f64>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Complex<f64>> {
    let mut acc = Complex::new(0.0, 0.0);
    let mut power = Complex::new(1.0, 0.0);
    for (idx, coeff) in coeffs.iter().enumerate() {
        acc += *coeff * power;
        power *= point;
        if (idx + 1) % SIGNAL_WORK_CHUNK as usize == 0 {
            cooperation.completed(SIGNAL_WORK_CHUNK).await?;
        }
    }
    cooperation
        .completed((coeffs.len() % SIGNAL_WORK_CHUNK as usize) as u64)
        .await?;
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, REQUIRED_DEFAULT_BINDING};
    use runmat_value::{IntValue, IntegerStorage};

    fn call(b: Value, a: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs
            .map(|count| crate::output_count::push_output_count(Some(count)))
            .transpose()
            .expect("output-count scope");
        block_on(evaluate(b, a, rest))
    }

    fn assert_error_identifier(result: BuiltinResult<Value>, identifier: &str) {
        let err = result.expect_err("expected freqz error");
        assert_eq!(err.identifier(), Some(identifier));
    }

    #[test]
    fn canonical_descriptor_and_default_binding_are_registered() {
        assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
            .expect("canonical freqz builtin");
        assert_eq!(builtin.bindings, &REQUIRED_DEFAULT_BINDING);
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            builtin.binding_identity(&builtin.bindings[0])
        );
        let descriptor = builtin.descriptor;
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[H, w] = freqz(b, a, n, fs)"));
        assert_eq!(builtin.integer_capabilities.len(), 3);
        assert!(builtin
            .extensions
            .iter()
            .any(|extension| extension.id == "freqz-resident-input"));
    }

    #[test]
    fn simple_fir_response_matches_closed_form() {
        let h = call(
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .unwrap();
        let Value::ComplexTensor(h) = h else {
            panic!("expected complex response");
        };
        assert_eq!(h.shape(), [4, 1]);
        assert!((h.materialize_f64()[0].0 - 2.0).abs() < 1e-12);
        assert!((h.materialize_f64()[2].0 - 1.0).abs() < 1e-12);
        assert!((h.materialize_f64()[2].1 + 1.0).abs() < 1e-12);
    }

    #[test]
    fn iir_response_and_frequency_outputs() {
        let out = call(
            Value::Num(0.2),
            Value::Tensor(Tensor::new(vec![1.0, -0.8], vec![1, 2]).unwrap()),
            &[Value::Num(8.0), Value::Num(1000.0)],
            Some(2),
        )
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        let Value::ComplexTensor(h) = &values[0] else {
            panic!("expected H");
        };
        let Value::Tensor(w) = &values[1] else {
            panic!("expected w");
        };
        assert!((h.materialize_f64()[0].0 - 1.0).abs() < 1e-12);
        assert_eq!(w.shape(), [8, 1]);
        assert!((w.materialize_f64()[1] - 62.5).abs() < 1e-12);
    }

    #[test]
    fn default_response_uses_512_point_radian_grid() {
        let out = call(Value::Num(1.0), Value::Num(1.0), &[], Some(2)).unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        let Value::ComplexTensor(h) = &values[0] else {
            panic!("expected H");
        };
        let Value::Tensor(w) = &values[1] else {
            panic!("expected w");
        };
        assert_eq!(h.shape(), [512, 1]);
        assert_eq!(w.shape(), [512, 1]);
        let frequencies = w.materialize_f64();
        assert_eq!(frequencies[0], 0.0);
        assert!((frequencies[511] - std::f64::consts::PI * 511.0 / 512.0).abs() < 1e-12);
    }

    #[test]
    fn output_count_padding_is_respected() {
        let out = call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Num(2.0)],
            Some(3),
        )
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 3);
        assert!(matches!(values[0], Value::ComplexTensor(_)));
        assert!(matches!(values[1], Value::Tensor(_)));
        assert_eq!(values[2], Value::Num(0.0));
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn zero_output_publishes_magnitude_and_phase_only_after_valid_response() {
        use crate::builtins::plotting::state::{
            clone_figure, current_figure_handle, lock_plot_test_registry, reset_plot_state,
        };
        let _plot_guard = lock_plot_test_registry();
        reset_plot_state();
        let invalid = call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Num(1.0)],
            Some(0),
        );
        assert_error_identifier(invalid, "RunMat:freqz:InvalidN");
        let figure = current_figure_handle();
        assert!(clone_figure(figure).is_none());
        let output = call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Num(8.0)],
            Some(0),
        )
        .unwrap();
        assert_eq!(output, Value::output_list(Vec::new()));
        let plotted = clone_figure(figure).expect("figure");
        assert_eq!(plotted.len(), 2);
        assert_eq!(
            plotted
                .axes_metadata(0)
                .and_then(|axes| axes.y_label.as_deref()),
            Some("Magnitude (dB)")
        );
        assert_eq!(
            plotted
                .axes_metadata(1)
                .and_then(|axes| axes.y_label.as_deref()),
            Some("Phase (rad)")
        );
        reset_plot_state();
        assert!(clone_figure(current_figure_handle()).is_none());
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn cancelled_response_does_not_mutate_the_figure() {
        use crate::builtins::plotting::state::{
            capture_figure_persistence_snapshot, clone_figure, current_figure_handle,
            lock_plot_test_registry, reset_plot_state,
        };
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::num::NonZeroU64;
        use std::rc::Rc;

        let _plot_guard = lock_plot_test_registry();
        reset_plot_state();
        call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Num(8.0)],
            Some(0),
        )
        .expect("seed frequency response");
        let figure = current_figure_handle();
        let before = capture_figure_persistence_snapshot(&[figure], 1, 1024 * 1024)
            .expect("bounded fixture snapshot");
        let source = Rc::new(CancellationSource::new());
        let cancel_on_turn = Rc::clone(&source);
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(SIGNAL_WORK_CHUNK).unwrap()),
            source.token(),
            move || {
                cancel_on_turn.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let error = block_on(evaluate_with_cooperation(
            Value::Num(1.0),
            Value::Num(1.0),
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(SIGNAL_WORK_CHUNK as f64)]),
            &mut cooperation,
        ))
        .expect_err("cancelled response must not publish graphics");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(current_figure_handle(), figure);
        let after = capture_figure_persistence_snapshot(&[figure], 1, 1024 * 1024)
            .expect("unchanged fixture snapshot");
        assert!(before.entries[0]
            .scene_lease
            .shares_lease(&after.entries[0].scene_lease));
        assert_eq!(clone_figure(figure).expect("retained figure").len(), 2);
        reset_plot_state();
        assert!(clone_figure(current_figure_handle()).is_none());
    }

    #[cfg(not(feature = "plot-core"))]
    #[test]
    fn zero_output_refuses_before_response_when_plotting_is_unavailable() {
        let error = call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Num(8.0)],
            Some(0),
        )
        .expect_err("zero-output form requires graphics");
        assert_eq!(
            error.identifier(),
            crate::plotting_hooks::unavailable_presentation().identifier()
        );
    }

    #[test]
    fn rejects_invalid_n_and_empty_coefficients() {
        assert_error_identifier(
            call(Value::Num(1.0), Value::Num(1.0), &[Value::Num(0.0)], None),
            "RunMat:freqz:InvalidN",
        );
        assert_error_identifier(
            call(Value::Num(1.0), Value::Num(1.0), &[Value::Num(1.0)], None),
            "RunMat:freqz:InvalidN",
        );
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        assert_error_identifier(
            call(Value::Tensor(empty), Value::Num(1.0), &[], None),
            "RunMat:freqz:InvalidCoefficients",
        );
    }

    #[test]
    fn rejects_invalid_fs_non_vector_coefficients_and_too_many_inputs() {
        assert_error_identifier(
            call(
                Value::Num(1.0),
                Value::Num(1.0),
                &[Value::Num(8.0), Value::Num(0.0)],
                None,
            ),
            "RunMat:freqz:InvalidFs",
        );
        assert_error_identifier(
            call(
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
                Value::Num(1.0),
                &[],
                None,
            ),
            "RunMat:freqz:InvalidCoefficients",
        );
        assert_error_identifier(
            call(
                Value::Num(1.0),
                Value::Num(1.0),
                &[Value::Num(8.0), Value::Num(1000.0), Value::Num(1.0)],
                None,
            ),
            "RunMat:freqz:ArgCount",
        );
    }

    #[test]
    fn empty_point_count_uses_default_response_length() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty point count");
        let output = call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Tensor(empty)],
            None,
        )
        .expect("empty n defaults");
        let Value::ComplexTensor(output) = output else {
            panic!("expected response");
        };
        assert_eq!(output.shape(), [512, 1]);
    }

    #[test]
    fn single_coefficients_produce_single_complex_response() {
        let numerator = Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).expect("single b");
        let output = call(
            Value::Tensor(numerator),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .expect("single response");
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex response");
        };
        assert!(matches!(output.complex_storage(), ComplexStorage::F32(_)));
    }

    #[test]
    fn integer_coefficient_count_and_sample_rate_extensions_are_independently_gated() {
        let cases = [
            (
                Value::Int(IntValue::U8(1)),
                Value::Num(1.0),
                Vec::new(),
                "RunMat:compatibility:FreqzIntegerCoefficientExtension",
            ),
            (
                Value::Num(1.0),
                Value::Num(1.0),
                vec![Value::Int(IntValue::U16(4))],
                "RunMat:compatibility:FreqzIntegerPointCountExtension",
            ),
            (
                Value::Num(1.0),
                Value::Num(1.0),
                vec![Value::Num(4.0), Value::Int(IntValue::U32(1000))],
                "RunMat:compatibility:FreqzIntegerSampleRateExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (b, a, options, identifier) in cases {
            let error = call(b, a, &options, None).expect_err("integer role must be gated");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn integer_coefficients_cross_one_checked_double_boundary() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let numerator = Tensor::new_integer(IntegerStorage::I16(vec![1, 1]), vec![1, 2])
            .expect("integer numerator");
        let output = call(
            Value::Tensor(numerator),
            Value::Int(IntValue::U8(1)),
            &[Value::Int(IntValue::U16(4))],
            None,
        )
        .expect("integer response");
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex response");
        };
        assert!(matches!(output.complex_storage(), ComplexStorage::F64(_)));

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("wide numerator");
        let error = call(
            Value::Tensor(wide),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .expect_err("lossy coefficient conversion must reject");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn logical_and_single_controls_have_distinct_compatibility_gates() {
        let single_n = Tensor::from_f32(vec![4.0], vec![1, 1]).expect("single n");
        let cases = [
            (
                vec![Value::Bool(true)],
                "RunMat:compatibility:FreqzLogicalPointCountExtension",
            ),
            (
                vec![Value::Tensor(single_n)],
                "RunMat:compatibility:FreqzSinglePointCountExtension",
            ),
            (
                vec![Value::Num(4.0), Value::Bool(true)],
                "RunMat:compatibility:FreqzLogicalSampleRateExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (options, identifier) in cases {
            let error = call(Value::Num(1.0), Value::Num(1.0), &options, None)
                .expect_err("control role must be gated");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn resident_extension_rejects_before_provider_access() {
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("admit owned fixture")
                .handle,
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(resident, Value::Num(1.0), &[], None)
            .expect_err("resident call must reject before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FreqzResidentInputExtension")
        );
    }
}
