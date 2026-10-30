//! MATLAB-compatible `linsolve` builtin with structural hints and GPU-aware fallbacks.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, ProviderLinsolveOptions, ProviderLinsolveResult,
};
#[cfg(test)]
use runmat_builtins::LINSOLVE_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, LINSOLVE_ERROR_INVALID_ARGUMENT, LINSOLVE_ERROR_INVALID_INPUT,
    LINSOLVE_EXPLICIT_GPU_TWO_OUTPUT_EXTENSION, LINSOLVE_INTEGER_INPUT_EXTENSION,
    LINSOLVE_INTEGER_OPTION_EXTENSION, LINSOLVE_LOGICAL_INPUT_EXTENSION, LINSOLVE_RCOND_EXTENSION,
    LINSOLVE_TEXT_TRANSA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexStorage, ComplexTensor, IntValue, IntegerComplexStorage, NumericDType, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, linalg::diagonal_rcond, tensor};
use crate::builtins::math::elementwise::complex_components::conjugate_integer_imaginary_storage;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "linsolve";

mod condition;
mod general;
pub use condition::{square_factor_rcond_for_provider, ProviderSquareFactor};
#[cfg(test)]
use general::binary_scale_preserves_component;
use general::{solve_general_complex, solve_general_real};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::linsolve")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linsolve",
    op_kind: GpuOpKind::Custom("solve"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linsolve")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider linsolve hook; WGPU currently supports triangular solves, real F32 TRANSA='T'/'C' variants, a dedicated real F32 POSDEF/Cholesky path, and selected real F32 QR-backed square and rectangular solves, otherwise it gathers to the host solver and re-uploads the result.",
};

fn linsolve_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    linsolve_error_with_message(message, &LINSOLVE_ERROR_INVALID_INPUT)
}

fn argument_error(message: impl Into<String>) -> RuntimeError {
    linsolve_error_with_message(message, &LINSOLVE_ERROR_INVALID_ARGUMENT)
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::solve::linsolve"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linsolve",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Linear solves are terminal operations and do not fuse with surrounding kernels.",
};

#[runtime_builtin(
    name = "linsolve",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::solve::linsolve"
)]
async fn linsolve_builtin(
    lhs: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let eval = evaluate_args(lhs, rhs, rest.view()).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.solution()]));
        }
        if out_count == 2 {
            return Ok(Value::output_list(vec![
                eval.solution(),
                eval.reciprocal_condition(),
            ]));
        }
        return Err(argument_error(
            "linsolve currently supports at most two outputs",
        ));
    }
    Ok(eval.solution())
}

/// Evaluate `linsolve`, returning both the solution and the estimated reciprocal condition number.
pub async fn evaluate(
    lhs: Value,
    rhs: Value,
    options: SolveOptions,
) -> BuiltinResult<LinsolveEval> {
    if let Some(eval) = try_gpu_linsolve(&lhs, &rhs, &options).await? {
        return Ok(eval);
    }

    let lhs_host = crate::dispatcher::gather_if_needed_async(&lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs_host = crate::dispatcher::gather_if_needed_async(&rhs)
        .await
        .map_err(map_control_flow)?;
    let pair = coerce_numeric_pair(lhs_host, rhs_host).await?;
    match pair {
        NumericPair::Real(lhs_r, rhs_r) => {
            let (solution, rcond) = solve_real(lhs_r, rhs_r, &options)?;
            Ok(LinsolveEval::new(
                tensor::tensor_into_value(solution),
                Some(rcond),
            ))
        }
        NumericPair::Complex(lhs_c, rhs_c) => {
            let (solution, rcond) = solve_complex(lhs_c, rhs_c, &options)?;
            Ok(LinsolveEval::new(
                Value::ComplexTensor(solution),
                Some(rcond),
            ))
        }
    }
}

/// Host implementation shared with acceleration providers that fall back to CPU execution.
pub fn linsolve_host_real_for_provider(
    lhs: &Tensor,
    rhs: &Tensor,
    options: &ProviderLinsolveOptions,
) -> BuiltinResult<(Tensor, f64)> {
    let opts = SolveOptions::from(options);
    let lhs = tensor::integer_tensor_to_f64(lhs.clone()).map_err(builtin_error)?;
    let rhs = tensor::integer_tensor_to_f64(rhs.clone()).map_err(builtin_error)?;
    solve_real(lhs, rhs, &opts)
}

/// Result wrapper that exposes both primary and secondary outputs.
#[derive(Clone)]
pub struct LinsolveEval {
    solution: Value,
    rcond: Option<f64>,
}

impl LinsolveEval {
    fn new(solution: Value, rcond: Option<f64>) -> Self {
        Self { solution, rcond }
    }

    /// Primary solution output.
    pub fn solution(&self) -> Value {
        self.solution.clone()
    }

    /// Estimated reciprocal condition number (second output).
    pub fn reciprocal_condition(&self) -> Value {
        match self.rcond {
            Some(r) => Value::Num(r),
            None => Value::Num(f64::NAN),
        }
    }
}

#[derive(Clone, Default)]
pub struct SolveOptions {
    lower: bool,
    upper: bool,
    rectangular: bool,
    transposed: bool,
    conjugate: bool,
    symmetric: bool,
    posdef: bool,
    rcond: Option<f64>,
}

impl From<&SolveOptions> for ProviderLinsolveOptions {
    fn from(opts: &SolveOptions) -> Self {
        Self {
            lower: opts.lower,
            upper: opts.upper,
            rectangular: opts.rectangular,
            transposed: opts.transposed,
            conjugate: opts.conjugate,
            symmetric: opts.symmetric,
            posdef: opts.posdef,
            need_rcond: false,
            rcond: opts.rcond,
        }
    }
}

impl From<&ProviderLinsolveOptions> for SolveOptions {
    fn from(opts: &ProviderLinsolveOptions) -> Self {
        Self {
            lower: opts.lower,
            upper: opts.upper,
            rectangular: opts.rectangular,
            transposed: opts.transposed,
            conjugate: opts.conjugate,
            symmetric: opts.symmetric,
            posdef: opts.posdef,
            rcond: opts.rcond,
        }
    }
}

fn options_from_rest(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<SolveOptions> {
    match rest.len() {
        0 => Ok(SolveOptions::default()),
        1 => parse_options(&rest[0]),
        _ => Err(argument_error("linsolve: too many input arguments")),
    }
}

/// Public helper for the VM multi-output surface.
pub async fn evaluate_args(
    lhs: Value,
    rhs: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<LinsolveEval> {
    let options = options_from_rest(rest)?;
    ensure_linsolve_extensions(&lhs, &rhs).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&lhs, NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&rhs, NAME)?;
    evaluate(lhs, rhs, options).await
}

async fn ensure_linsolve_extensions(lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    let integer = |value: &Value| {
        matches!(value, Value::Int(_))
            || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
            || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
    };
    if integer(lhs) || integer(rhs) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LINSOLVE_INTEGER_INPUT_EXTENSION,
            NAME,
        )?;
        for value in [lhs, rhs] {
            if integer(value)
                && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(
                    value,
                )
                .await?
            {
                return Err(builtin_error(
                    "linsolve: integer input lies outside the exact binary64 interval",
                ));
            }
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(lhs)
        || crate::builtins::common::validation::value_has_logical_class(rhs)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LINSOLVE_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if matches!(crate::output_count::current_output_count(), Some(2)) && [lhs, rhs].iter().any(|value| matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_is_explicit(h))) {
        crate::compatibility::ensure_builtin_extension_enabled(&LINSOLVE_EXPLICIT_GPU_TWO_OUTPUT_EXTENSION, NAME)?;
    }
    Ok(())
}

async fn try_gpu_linsolve(
    lhs: &Value,
    rhs: &Value,
    options: &SolveOptions,
) -> BuiltinResult<Option<LinsolveEval>> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Ok(None);
    }
    let gpu_handles: Vec<&GpuTensorHandle> = [lhs, rhs]
        .into_iter()
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(handle)
            } else {
                None
            }
        })
        .collect();
    gpu_helpers::require_resident_compute_owners(&gpu_handles, false, || {
        builtin_error("linsolve: resident input has no exact owning provider")
    })?;
    let provider = match gpu_handles
        .first()
        .map(|handle| gpu_helpers::exact_provider_for_handle(handle))
        .unwrap_or_else(runmat_accelerate_api::provider)
    {
        Some(p) => p,
        None => return Ok(None),
    };
    if gpu_handles.iter().any(|handle| {
        gpu_helpers::exact_provider_for_handle(handle)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
    }) {
        return Ok(None);
    }

    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(None);
    }
    let host_extension_input = gpu_handles.is_empty()
        && (value_has_integer_class(lhs)
            || value_has_integer_class(rhs)
            || crate::builtins::common::validation::value_has_logical_class(lhs)
            || crate::builtins::common::validation::value_has_logical_class(rhs));
    if host_extension_input
        || (provider.precision() != runmat_accelerate_api::ProviderPrecision::F64
            && [lhs, rhs].iter().any(|value| {
                value_has_integer_class(value)
                    || crate::builtins::common::validation::value_has_logical_class(value)
            }))
    {
        return Ok(None);
    }

    let lhs_operand = match prepare_gpu_operand(lhs, provider, &gpu_handles)? {
        Some(op) => op,
        None => return Ok(None),
    };
    let mut protected = gpu_handles.clone();
    protected.push(lhs_operand.handle());
    let rhs_operand = match prepare_gpu_operand(rhs, provider, &protected)? {
        Some(op) => op,
        None => {
            return Ok(None);
        }
    };

    if is_scalar_handle(lhs_operand.handle()) || is_scalar_handle(rhs_operand.handle()) {
        return Ok(None);
    }

    let mut provider_opts: ProviderLinsolveOptions = options.into();
    let lhs_rows = lhs_operand.handle().shape().first().copied().unwrap_or(1);
    let lhs_cols = lhs_operand.handle().shape().get(1).copied().unwrap_or(1);
    let effective_rows = if options.transposed {
        lhs_cols
    } else {
        lhs_rows
    };
    let effective_cols = if options.transposed {
        lhs_rows
    } else {
        lhs_cols
    };
    let rectangular = effective_rows != effective_cols;
    let wants_second_output = matches!(crate::output_count::current_output_count(), Some(2));
    if rectangular && wants_second_output {
        return Ok(None);
    }
    provider_opts.need_rcond = wants_second_output || options.rcond.is_some();
    let result = match provider
        .linsolve(lhs_operand.handle(), rhs_operand.handle(), &provider_opts)
        .await
    {
        Ok(result) => Some(result),
        Err(error) if error.is_unsupported() => None,
        Err(source) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                builtin_error(source.to_string()),
                source,
            ));
        }
    };

    if let Some(ProviderLinsolveResult {
        solution,
        reciprocal_condition,
    }) = result
    {
        let aliases_lhs = solution.shares_allocation_with(lhs_operand.handle());
        let aliases_rhs = solution.shares_allocation_with(rhs_operand.handle());
        let expected_rows = effective_cols;
        let expected_cols = rhs_operand.handle().shape().get(1).copied().unwrap_or(1);
        let expected_precision = match (
            runmat_accelerate_api::handle_precision(lhs_operand.handle()),
            runmat_accelerate_api::handle_precision(rhs_operand.handle()),
        ) {
            (
                Some(runmat_accelerate_api::ProviderPrecision::F32),
                Some(runmat_accelerate_api::ProviderPrecision::F32),
            ) => Some(runmat_accelerate_api::ProviderPrecision::F32),
            (Some(_), Some(_)) => Some(runmat_accelerate_api::ProviderPrecision::F64),
            _ => None,
        };
        let valid = !aliases_lhs
            && !aliases_rhs
            && solution.shape() == vec![expected_rows, expected_cols]
            && solution.device_id() == provider.device_id()
            && gpu_helpers::exact_provider_for_handle(&solution).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
            && runmat_accelerate_api::handle_storage(&solution)
                == runmat_accelerate_api::GpuTensorStorage::Real
            && expected_precision.is_some()
            && runmat_accelerate_api::handle_precision(&solution) == expected_precision
            && runmat_accelerate_api::handle_integer_type(&solution).is_none()
            && !runmat_accelerate_api::handle_is_logical(&solution);
        if !valid {
            return Err(gpu_helpers::reject_gpu_output(
                solution,
                builtin_error("linsolve: provider returned malformed or aliased output"),
            ));
        }
        let solution =
            gpu_helpers::propagate_owned_output_provenance(solution, gpu_handles.iter().copied())?;
        let eval = LinsolveEval::new(
            gpu_helpers::resident_gpu_value(solution)?,
            Some(reciprocal_condition),
        );
        return Ok(Some(eval));
    }

    Ok(None)
}

fn value_has_integer_class(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
}

fn parse_options(value: &Value) -> BuiltinResult<SolveOptions> {
    let struct_val = match value {
        Value::Struct(s) => s,
        other => {
            return Err(argument_error(format!(
                "linsolve: opts must be a struct, got {other:?}"
            )))
        }
    };
    let mut opts = SolveOptions::default();
    for (key, raw_value) in &struct_val.fields {
        let name = key.to_ascii_uppercase();
        match name.as_str() {
            "LT" => opts.lower = parse_bool_field("LT", raw_value)?,
            "UT" => opts.upper = parse_bool_field("UT", raw_value)?,
            "RECT" => opts.rectangular = parse_bool_field("RECT", raw_value)?,
            "SYM" => opts.symmetric = parse_bool_field("SYM", raw_value)?,
            "POSDEF" => opts.posdef = parse_bool_field("POSDEF", raw_value)?,
            "TRANSA" => {
                if matches!(
                    raw_value,
                    Value::CharArray(_) | Value::String(_) | Value::StringArray(_)
                ) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &LINSOLVE_TEXT_TRANSA_EXTENSION,
                        NAME,
                    )?;
                }
                let transa = parse_transa(raw_value)?;
                opts.transposed = transa != TransposeMode::None;
                opts.conjugate = transa == TransposeMode::Conjugate;
            }
            "RCOND" => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LINSOLVE_RCOND_EXTENSION,
                    NAME,
                )?;
                let threshold = parse_scalar_f64("RCOND", raw_value)?;
                if threshold < 0.0 {
                    return Err(argument_error("linsolve: RCOND must be non-negative"));
                }
                opts.rcond = Some(threshold);
            }
            other => {
                return Err(argument_error(format!(
                    "linsolve: unknown option '{other}'"
                )))
            }
        }
    }
    if opts.lower && opts.upper {
        return Err(argument_error(
            "linsolve: LT and UT are mutually exclusive.",
        ));
    }
    Ok(opts)
}

fn parse_bool_field(name: &str, value: &Value) -> BuiltinResult<bool> {
    if matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LINSOLVE_INTEGER_OPTION_EXTENSION,
            NAME,
        )?;
    }
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(match scalar_tensor_integer(t) {
            Some(value) => !value.is_zero(),
            None => tensor::tensor_value_f64(t, 0) != 0.0,
        }),
        Value::LogicalArray(arr) if arr.len() == 1 => Ok(arr.data[0] != 0),
        other => Err(argument_error(format!(
            "linsolve: option '{name}' must be logical or numeric, got {other:?}"
        ))),
    }
}

fn parse_scalar_f64(name: &str, value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(match scalar_tensor_integer(t) {
            Some(value) => value.to_f64(),
            None => tensor::tensor_value_f64(t, 0),
        }),
        other => Err(argument_error(format!(
            "linsolve: option '{name}' must be a scalar numeric value, got {other:?}"
        ))),
    }
}

fn scalar_tensor_integer(tensor: &Tensor) -> Option<IntValue> {
    tensor
        .integer_storage()
        .and_then(|storage| storage.value_at(0))
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum TransposeMode {
    None,
    Transpose,
    Conjugate,
}

fn parse_transa(value: &Value) -> BuiltinResult<TransposeMode> {
    match value {
        Value::Bool(false) => return Ok(TransposeMode::None),
        Value::Bool(true) => return Ok(TransposeMode::Conjugate),
        Value::LogicalArray(array) if array.len() == 1 && array.data[0] == 0 => {
            return Ok(TransposeMode::None)
        }
        Value::LogicalArray(array) if array.len() == 1 => return Ok(TransposeMode::Conjugate),
        _ => {}
    }
    let text = tensor::value_to_string(value)
        .ok_or_else(|| argument_error("linsolve: TRANSA must be a logical scalar"))?;
    if text.is_empty() {
        return Err(argument_error("linsolve: TRANSA cannot be empty"));
    }
    match text.trim().to_ascii_uppercase().as_str() {
        "N" => Ok(TransposeMode::None),
        "T" => Ok(TransposeMode::Transpose),
        "C" => Ok(TransposeMode::Conjugate),
        other => Err(argument_error(format!(
            "linsolve: extended text TRANSA must be 'N', 'T', or 'C', got '{other}'"
        ))),
    }
}

enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

enum NumericPair {
    Real(Tensor, Tensor),
    Complex(ComplexTensor, ComplexTensor),
}

async fn coerce_numeric_pair(lhs: Value, rhs: Value) -> BuiltinResult<NumericPair> {
    let lhs_num = coerce_numeric(lhs).await?;
    let rhs_num = coerce_numeric(rhs).await?;
    match (lhs_num, rhs_num) {
        (NumericInput::Real(lhs_r), NumericInput::Real(rhs_r)) => {
            Ok(NumericPair::Real(lhs_r, rhs_r))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Complex(rhs_c)) => {
            Ok(NumericPair::Complex(lhs_c, rhs_c))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Real(rhs_r)) => {
            let rhs_c = promote_real_tensor(&rhs_r)?;
            Ok(NumericPair::Complex(lhs_c, rhs_c))
        }
        (NumericInput::Real(lhs_r), NumericInput::Complex(rhs_c)) => {
            let lhs_c = promote_real_tensor(&lhs_r)?;
            Ok(NumericPair::Complex(lhs_c, rhs_c))
        }
    }
}

async fn coerce_numeric(value: Value) -> BuiltinResult<NumericInput> {
    match value {
        Value::Tensor(tensor) => {
            let tensor = tensor::integer_tensor_to_f64(tensor).map_err(builtin_error)?;
            ensure_matrix_shape(NAME, tensor.shape())?;
            Ok(NumericInput::Real(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(builtin_error)?;
            ensure_matrix_shape(NAME, tensor.shape())?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Bool(b) => {
            let tensor =
                Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Real(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            Ok(NumericInput::Complex(tensor))
        }
        Value::ComplexTensor(ct) => {
            ensure_matrix_shape(NAME, ct.shape())?;
            Ok(NumericInput::Complex(ct))
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(map_control_flow)?;
            let tensor = tensor::integer_tensor_to_f64(tensor).map_err(builtin_error)?;
            ensure_matrix_shape(NAME, tensor.shape())?;
            Ok(NumericInput::Real(tensor))
        }
        other => Err(builtin_error(format!(
            "{NAME}: unsupported input type {:?}; convert to numeric values first",
            other
        ))),
    }
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn is_scalar_handle(handle: &GpuTensorHandle) -> bool {
    crate::builtins::common::shape::is_scalar_shape(handle.shape())
}

struct PreparedOperand {
    handle: GpuTensorHandle,
}

#[cfg(test)]
#[path = "linsolve/owning_tests.rs"]
mod owning_tests;

impl PreparedOperand {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
        }
    }

    fn owned(handle: GpuTensorHandle) -> Self {
        Self { handle }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

fn prepare_gpu_operand(
    value: &Value,
    provider: &'static dyn AccelProvider,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Option<PreparedOperand>> {
    match value {
        Value::GpuTensor(handle) => {
            if handle.device_id() != provider.device_id()
                || gpu_helpers::exact_provider_for_handle(handle).is_none_or(|owner| {
                    !runmat_accelerate_api::same_provider_instance(owner, provider)
                })
                || is_scalar_handle(handle)
            {
                Ok(None)
            } else {
                Ok(Some(PreparedOperand::borrowed(handle)))
            }
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(None)
            } else {
                let uploaded = upload_tensor(provider, tensor, protected)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(None)
            } else {
                let tensor = tensor::logical_to_tensor(logical).map_err(builtin_error)?;
                let uploaded = upload_tensor(provider, &tensor, protected)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        _ => Ok(None),
    }
}

fn upload_tensor(
    provider: &'static dyn AccelProvider,
    tensor: &Tensor,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<GpuTensorHandle> {
    if tensor.integer_storage().is_none() {
        return gpu_helpers::upload_tensor_protecting(provider, tensor, protected);
    }
    // The integer extension has an explicit floating boundary. It does not
    // inherit the provider's preferred precision or round through F32.
    let values = tensor::tensor_values_f64_cow(tensor);
    let transfer =
        Tensor::new(values.into_owned(), tensor.shape().to_vec()).map_err(builtin_error)?;
    gpu_helpers::upload_tensor_protecting(provider, &transfer, protected)
}

fn solve_real(lhs: Tensor, rhs: Tensor, options: &SolveOptions) -> BuiltinResult<(Tensor, f64)> {
    let mut lhs_effective = lhs;
    let mut rhs_effective = rhs;
    let mut lower = options.lower;
    let mut upper = options.upper;

    if options.transposed {
        lhs_effective = transpose_tensor(&lhs_effective);
        if options.conjugate {
            conjugate_in_place(&mut lhs_effective);
        }
        if lower || upper {
            std::mem::swap(&mut lower, &mut upper);
        }
    }

    rhs_effective = normalize_rhs_tensor(rhs_effective, lhs_effective.rows())?;

    if lower {
        ensure_square(lhs_effective.rows(), lhs_effective.cols())?;
        let (solution, rcond) = forward_substitution_real(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    if upper {
        ensure_square(lhs_effective.rows(), lhs_effective.cols())?;
        let (solution, rcond) = backward_substitution_real(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    let (solution, rcond, rank) = solve_general_real(&lhs_effective, &rhs_effective)?;
    enforce_rcond(options, rcond)?;
    Ok((
        solution,
        if lhs_effective.rows() == lhs_effective.cols() {
            rcond
        } else {
            rank
        },
    ))
}

fn solve_complex(
    lhs: ComplexTensor,
    rhs: ComplexTensor,
    options: &SolveOptions,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let mut lhs_effective = lhs;
    let mut rhs_effective = rhs;
    let mut lower = options.lower;
    let mut upper = options.upper;

    if options.transposed {
        lhs_effective = transpose_complex(&lhs_effective);
        if options.conjugate {
            conjugate_complex_in_place(&mut lhs_effective);
        }
        if lower || upper {
            std::mem::swap(&mut lower, &mut upper);
        }
    }

    rhs_effective = normalize_rhs_complex(rhs_effective, lhs_effective.rows())?;

    if lower {
        ensure_square(lhs_effective.rows(), lhs_effective.cols())?;
        let (solution, rcond) = forward_substitution_complex(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    if upper {
        ensure_square(lhs_effective.rows(), lhs_effective.cols())?;
        let (solution, rcond) = backward_substitution_complex(&lhs_effective, &rhs_effective)?;
        enforce_rcond(options, rcond)?;
        return Ok((solution, rcond));
    }

    let (solution, rcond, rank) = solve_general_complex(&lhs_effective, &rhs_effective)?;
    enforce_rcond(options, rcond)?;
    Ok((
        solution,
        if lhs_effective.rows() == lhs_effective.cols() {
            rcond
        } else {
            rank
        },
    ))
}

fn forward_substitution_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64)> {
    let n = lhs.rows();
    let lhs_values = tensor::tensor_values_f64_cow(lhs);
    let mut solution = tensor::tensor_values_f64(rhs);
    let nrhs = rhs.cols();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for i in 0..n {
            let diag = lhs_values[i + i * n];
            let diag_abs = diag.abs();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error(
                    "linsolve: matrix is singular to working precision.",
                ));
            }
            let mut accum = 0.0;
            for j in 0..i {
                accum += lhs_values[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = if n == 0 {
        1.0
    } else {
        diagonal_rcond(min_diag, max_diag)
    };
    let tensor = real_solution_tensor(
        solution,
        rhs.shape().to_vec(),
        real_solution_dtype(lhs, rhs),
    )?;
    Ok((tensor, rcond))
}

fn backward_substitution_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<(Tensor, f64)> {
    let n = lhs.rows();
    let lhs_values = tensor::tensor_values_f64_cow(lhs);
    let mut solution = tensor::tensor_values_f64(rhs);
    let nrhs = rhs.cols();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for row_rev in 0..n {
            let i = n - 1 - row_rev;
            let diag = lhs_values[i + i * n];
            let diag_abs = diag.abs();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error(
                    "linsolve: matrix is singular to working precision.",
                ));
            }
            let mut accum = 0.0;
            for j in (i + 1)..n {
                accum += lhs_values[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = if n == 0 {
        1.0
    } else {
        diagonal_rcond(min_diag, max_diag)
    };
    let tensor = real_solution_tensor(
        solution,
        rhs.shape().to_vec(),
        real_solution_dtype(lhs, rhs),
    )?;
    Ok((tensor, rcond))
}

fn forward_substitution_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let n = lhs.rows();
    let nrhs = rhs.cols();
    let lhs_data: Vec<Complex64> = lhs
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut solution: Vec<Complex64> = rhs
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for i in 0..n {
            let diag = lhs_data[i + i * n];
            let diag_abs = diag.norm();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error(
                    "linsolve: matrix is singular to working precision.",
                ));
            }
            let mut accum = Complex64::new(0.0, 0.0);
            for j in 0..i {
                accum += lhs_data[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = if n == 0 {
        1.0
    } else {
        diagonal_rcond(min_diag, max_diag)
    };
    let tensor = ComplexTensor::new(
        solution.iter().map(|c| (c.re, c.im)).collect(),
        rhs.shape().to_vec(),
    )
    .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok((tensor, rcond))
}

fn backward_substitution_complex(
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<(ComplexTensor, f64)> {
    let n = lhs.rows();
    let nrhs = rhs.cols();
    let lhs_data: Vec<Complex64> = lhs
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut solution: Vec<Complex64> = rhs
        .materialize_f64()
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;

    for col in 0..nrhs {
        for row_rev in 0..n {
            let i = n - 1 - row_rev;
            let diag = lhs_data[i + i * n];
            let diag_abs = diag.norm();
            min_diag = min_diag.min(diag_abs);
            max_diag = max_diag.max(diag_abs);
            if diag_abs == 0.0 {
                return Err(builtin_error(
                    "linsolve: matrix is singular to working precision.",
                ));
            }
            let mut accum = Complex64::new(0.0, 0.0);
            for j in (i + 1)..n {
                accum += lhs_data[i + j * n] * solution[j + col * n];
            }
            let rhs_value = solution[i + col * n] - accum;
            solution[i + col * n] = rhs_value / diag;
        }
    }

    let rcond = if n == 0 {
        1.0
    } else {
        diagonal_rcond(min_diag, max_diag)
    };
    let tensor = ComplexTensor::new(
        solution.iter().map(|c| (c.re, c.im)).collect(),
        rhs.shape().to_vec(),
    )
    .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok((tensor, rcond))
}

fn normalize_rhs_tensor(rhs: Tensor, expected_rows: usize) -> BuiltinResult<Tensor> {
    if rhs.rows() == expected_rows {
        return Ok(rhs);
    }
    if rhs.shape().len() == 1 && rhs.shape()[0] == expected_rows {
        return rhs
            .reshape(vec![expected_rows, 1])
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if tensor::tensor_element_len(&rhs) == 0 && expected_rows == 0 {
        return Ok(rhs);
    }
    Err(builtin_error("Matrix dimensions must agree."))
}

fn normalize_rhs_complex(rhs: ComplexTensor, expected_rows: usize) -> BuiltinResult<ComplexTensor> {
    if rhs.rows() == expected_rows {
        return Ok(rhs);
    }
    if rhs.shape().len() == 1 && rhs.shape()[0] == expected_rows {
        return ComplexTensor::new(rhs.materialize_f64(), vec![expected_rows, 1])
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rhs.materialize_f64().is_empty() && expected_rows == 0 {
        return Ok(rhs);
    }
    Err(builtin_error("Matrix dimensions must agree."))
}

fn enforce_rcond(options: &SolveOptions, rcond: f64) -> BuiltinResult<()> {
    if let Some(threshold) = options.rcond {
        if rcond < threshold {
            return Err(builtin_error(
                "linsolve: matrix is singular to working precision.",
            ));
        }
    }
    Ok(())
}

fn compute_svd_tolerance(singular_values: &[f64], rows: usize, cols: usize) -> f64 {
    let max_sv = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_dim = rows.max(cols) as f64;
    f64::EPSILON * max_dim * max_sv.max(1.0)
}

fn matrix_real_to_tensor(matrix: DMatrix<f64>, dtype: NumericDType) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    real_solution_tensor(matrix.as_slice().to_vec(), vec![rows, cols], dtype)
}

fn real_solution_dtype(lhs: &Tensor, rhs: &Tensor) -> NumericDType {
    if lhs.numeric_dtype() == NumericDType::F32 && rhs.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn real_solution_tensor(
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
) -> BuiltinResult<Tensor> {
    let tensor = match dtype {
        NumericDType::F32 => Tensor::from_f32(
            values.into_iter().map(|value| value as f32).collect(),
            shape,
        ),
        NumericDType::F64 => Tensor::new(values, shape),
        _ => Err(format!(
            "linsolve: unsupported real solution class {}",
            dtype.class_name()
        )),
    };
    tensor.map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn complex_solution_tensor(
    matrix: DMatrix<Complex64>,
    lhs: &ComplexTensor,
    rhs: &ComplexTensor,
) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if lhs.numeric_dtype() == NumericDType::F32 && rhs.numeric_dtype() == NumericDType::F32 {
        ComplexTensor::from_f32(
            matrix
                .as_slice()
                .iter()
                .map(|c| (c.re as f32, c.im as f32))
                .collect(),
            vec![rows, cols],
        )
    } else {
        ComplexTensor::new(
            matrix.as_slice().iter().map(|c| (c.re, c.im)).collect(),
            vec![rows, cols],
        )
    }
    .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn promote_real_tensor(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let result = if let Some(values) = tensor.as_f32_slice() {
        ComplexTensor::from_f32(
            values.iter().map(|&real| (real, 0.0)).collect(),
            tensor.shape().to_vec(),
        )
    } else {
        let values = tensor::tensor_values_f64_cow(tensor);
        ComplexTensor::new(
            values.iter().map(|&real| (real, 0.0)).collect(),
            tensor.shape().to_vec(),
        )
    };
    result.map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if is_effectively_matrix(shape) {
        Ok(())
    } else {
        Err(builtin_error(format!(
            "{name}: inputs must be 2-D matrices or vectors"
        )))
    }
}

fn is_effectively_matrix(shape: &[usize]) -> bool {
    match shape.len() {
        0..=2 => true,
        _ => shape.iter().skip(2).all(|&dim| dim == 1),
    }
}

fn ensure_square(rows: usize, cols: usize) -> BuiltinResult<()> {
    if rows == cols {
        Ok(())
    } else {
        Err(builtin_error(
            "linsolve: triangular solves require a square coefficient matrix.",
        ))
    }
}

fn transpose_tensor(tensor: &Tensor) -> Tensor {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut indices = vec![0usize; tensor::tensor_element_len(tensor)];
    for r in 0..rows {
        for c in 0..cols {
            indices[c + r * cols] = r + c * rows;
        }
    }
    let storage = tensor
        .clone()
        .into_numeric_storage()
        .expect("validated tensor storage");
    Tensor::from_numeric_storage(
        storage
            .gather(&indices)
            .expect("transpose indices in bounds"),
        vec![cols, rows],
    )
    .expect("transpose tensor shape matches storage")
}

fn transpose_complex(tensor: &ComplexTensor) -> ComplexTensor {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut indices = vec![0usize; tensor.len()];
    for r in 0..rows {
        for c in 0..cols {
            indices[c + r * cols] = r + c * rows;
        }
    }
    ComplexTensor::from_complex_storage(
        tensor
            .complex_storage()
            .gather(&indices)
            .expect("transpose indices in bounds"),
        vec![cols, rows],
    )
    .expect("transpose complex shape matches storage")
}

fn conjugate_in_place(_tensor: &mut Tensor) {
    // Real-valued matrices are unaffected by conjugation.
}

fn conjugate_complex_in_place(tensor: &mut ComplexTensor) {
    let shape = tensor.shape().to_vec();
    let storage = match tensor.clone().into_complex_storage() {
        ComplexStorage::F64(mut values) => {
            for value in &mut values {
                value.1 = -value.1;
            }
            ComplexStorage::F64(values)
        }
        ComplexStorage::F32(mut values) => {
            for value in &mut values {
                value.1 = -value.1;
            }
            ComplexStorage::F32(values)
        }
        ComplexStorage::Integer(storage) => {
            let (real, imag) = storage.into_unbounded_components();
            ComplexStorage::Integer(
                IntegerComplexStorage::new(real, conjugate_integer_imaginary_storage(imag))
                    .expect("complex integer component classes remain matched"),
            )
        }
    };
    *tensor = ComplexTensor::from_complex_storage(storage, shape)
        .expect("conjugated complex storage retains shape");
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{CharArray, IntegerStorage, NumericStorage};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn approx_eq(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-7);
    }

    #[cfg(feature = "wgpu")]
    fn first_output(outputs: runmat_value::OutputListStorage) -> Value {
        outputs
            .try_into_owned()
            .expect("unique output list")
            .into_iter()
            .next()
            .expect("one output")
    }

    fn evaluate_args(a: Value, b: Value, rest: &[Value]) -> Result<LinsolveEval, RuntimeError> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::evaluate_args(
            a,
            b,
            crate::call::FunctionArgumentView::from_slice(rest),
        ))
    }

    #[test]
    fn linsolve_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = LINSOLVE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"X = linsolve(A, B)"));
        assert!(labels.contains(&"X = linsolve(A, B, opts)"));
        assert!(labels.contains(&"[X, R] = linsolve(A, B)"));
        assert!(labels.contains(&"[X, R] = linsolve(A, B, opts)"));
    }

    #[test]
    fn linsolve_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = LINSOLVE_DESCRIPTOR
            .errors
            .iter()
            .map(|err| err.code)
            .collect();
        assert!(codes.contains(&"RM.LINSOLVE.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.LINSOLVE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.LINSOLVE.INTERNAL"));
    }

    use crate::builtins::common::test_support;
    use runmat_accelerate_api::ProviderTelemetry;

    fn linsolve_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        block_on(super::linsolve_builtin(lhs, rhs, rest.into()))
    }

    #[test]
    fn linsolve_integer_extension_is_rejected_in_matlab_mode() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let lhs = Tensor::new_integer(IntegerStorage::I32(vec![1]), vec![1, 1]).unwrap();
        let error = block_on(super::linsolve_builtin(
            Value::Tensor(lhs),
            Value::Num(1.0),
            crate::execution::RootedValues::new(Vec::new())
                .expect("ordinary source arguments")
                .into(),
        ))
        .expect_err("integer linsolve is a RunMat extension");
        assert_eq!(
            error.identifier(),
            LINSOLVE_INTEGER_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn linsolve_option_extensions_are_independently_gated() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);

        let mut integer_option = runmat_value::UnboundedStructValueBuilder::new();
        integer_option
            .fields
            .insert("LT".to_string(), Value::Int(IntValue::I8(1)));
        let error = match options_from_rest(crate::call::FunctionArgumentView::from_slice(&[
            Value::Struct(integer_option.finish()),
        ])) {
            Err(error) => error,
            Ok(_) => panic!("typed structural option must be gated"),
        };
        assert_eq!(
            error.identifier(),
            LINSOLVE_INTEGER_OPTION_EXTENSION.error_identifier
        );

        let mut text_transa = runmat_value::UnboundedStructValueBuilder::new();
        text_transa
            .fields
            .insert("TRANSA".to_string(), Value::from("T"));
        let error = match options_from_rest(crate::call::FunctionArgumentView::from_slice(&[
            Value::Struct(text_transa.finish()),
        ])) {
            Err(error) => error,
            Ok(_) => panic!("text TRANSA must be gated"),
        };
        assert_eq!(
            error.identifier(),
            LINSOLVE_TEXT_TRANSA_EXTENSION.error_identifier
        );

        let mut rcond = runmat_value::UnboundedStructValueBuilder::new();
        rcond.fields.insert("RCOND".to_string(), Value::Num(0.1));
        let error = match options_from_rest(crate::call::FunctionArgumentView::from_slice(&[
            Value::Struct(rcond.finish()),
        ])) {
            Err(error) => error,
            Ok(_) => panic!("RCOND must be gated"),
        };
        assert_eq!(
            error.identifier(),
            LINSOLVE_RCOND_EXTENSION.error_identifier
        );
    }

    #[test]
    fn linsolve_documented_logical_transa_does_not_require_extension() {
        let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options
            .fields
            .insert("TRANSA".to_string(), Value::Bool(true));
        let parsed = options_from_rest(crate::call::FunctionArgumentView::from_slice(&[
            Value::Struct(options.finish()),
        ]))
        .expect("logical TRANSA");
        assert!(parsed.transposed);
        assert!(parsed.conjugate);
    }

    fn evaluate(lhs: Value, rhs: Value, options: SolveOptions) -> BuiltinResult<LinsolveEval> {
        block_on(super::evaluate(lhs, rhs, options))
    }

    fn fallback_count(telemetry: &ProviderTelemetry, reason: &str) -> u64 {
        telemetry
            .solve_fallbacks
            .iter()
            .find(|entry| entry.reason == reason)
            .map(|entry| entry.count)
            .unwrap_or(0)
    }

    #[cfg(feature = "wgpu")]
    fn kernel_launch_count(telemetry: &ProviderTelemetry, kernel: &str) -> usize {
        telemetry
            .kernel_launches
            .iter()
            .filter(|entry| entry.kernel == kernel)
            .count()
    }

    fn clear_accel_provider_state() {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
    }

    fn host_linsolve_real(
        a: &Tensor,
        b: &Tensor,
        options: ProviderLinsolveOptions,
    ) -> (Tensor, f64) {
        super::linsolve_host_real_for_provider(a, b, &options).expect("host linsolve")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_basic_square() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let result =
            linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("linsolve");
        let t = test_support::gather(result).expect("gather");
        assert_eq!(t.shape(), [2, 1]);
        approx_eq(t.materialize_f64()[0], 1.0);
        approx_eq(t.materialize_f64()[1], 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn rhs_normalization_preserves_real_getters_and_complex_cached_geometry() {
        let real_rank_zero = Tensor::new(vec![7.0], vec![]).expect("rank-zero real rhs");
        let real_rank_zero = normalize_rhs_tensor(real_rank_zero, 1).expect("real scalar rhs");
        assert!(real_rank_zero.shape().is_empty());

        let complex_rank_zero =
            ComplexTensor::new(vec![(7.0, -1.0)], vec![]).expect("rank-zero complex rhs");
        assert!(normalize_rhs_complex(complex_rank_zero, 1).is_err());

        let real_rank_one = Tensor::new(vec![2.0, 3.0], vec![2]).expect("rank-one real rhs");
        let real_rank_one = normalize_rhs_tensor(real_rank_one, 2).expect("real vector rhs");
        assert_eq!(real_rank_one.shape(), [2]);

        let complex_rank_one = ComplexTensor::new(vec![(2.0, 1.0), (3.0, -1.0)], vec![2])
            .expect("rank-one complex rhs");
        let complex_rank_one =
            normalize_rhs_complex(complex_rank_one, 2).expect("complex vector rhs");
        assert_eq!(complex_rank_one.shape(), [2, 1]);
        assert_eq!(
            complex_rank_one.materialize_f64(),
            vec![(2.0, 1.0), (3.0, -1.0)]
        );
    }

    #[test]
    fn linsolve_cpu_preserves_native_single_for_general_and_triangular_solutions() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();

        let a = Tensor::from_f32(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::from_f32(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let result =
            linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(
            tensor.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0])
        );

        let lower = Tensor::from_f32(vec![2.0, 1.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let rhs = Tensor::from_f32(vec![2.0, 7.0], vec![2, 1]).unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.fields.insert("LT".to_string(), Value::Bool(true));
        let result = linsolve_builtin(
            Value::Tensor(lower),
            Value::Tensor(rhs),
            vec![Value::Struct(options.finish())],
        )
        .expect("lower triangular linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(
            tensor.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0])
        );
    }

    #[test]
    fn linsolve_reads_typed_integer_tensor_storage_exactly() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new_integer(IntegerStorage::U64(vec![2, 1, 1, 2]), vec![2, 2])
            .expect("integer lhs");
        let b =
            Tensor::new_integer(IntegerStorage::U64(vec![4, 5]), vec![2, 1]).expect("integer rhs");

        let result =
            linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(tensor.shape(), [2, 1]);
        approx_eq(tensor.materialize_f64()[0], 1.0);
        approx_eq(tensor.materialize_f64()[1], 2.0);
    }

    #[test]
    fn linsolve_complex_promotion_reads_typed_integer_storage_exactly() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();

        let real_lhs = Tensor::new_integer(IntegerStorage::I64(vec![1, 0, 0, 1]), vec![2, 2])
            .expect("integer lhs");
        let complex_rhs = ComplexTensor::new(vec![(3.0, 4.0), (5.0, -6.0)], vec![2, 1]).unwrap();
        let result = linsolve_builtin(
            Value::Tensor(real_lhs),
            Value::ComplexTensor(complex_rhs),
            Vec::new(),
        )
        .expect("linsolve");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor output");
        };
        assert_eq!(out.shape(), [2, 1]);
        approx_eq(out.materialize_f64()[0].0, 3.0);
        approx_eq(out.materialize_f64()[0].1, 4.0);
        approx_eq(out.materialize_f64()[1].0, 5.0);
        approx_eq(out.materialize_f64()[1].1, -6.0);

        let complex_lhs = ComplexTensor::new(
            vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let real_rhs =
            Tensor::new_integer(IntegerStorage::U64(vec![7, 11]), vec![2, 1]).expect("integer rhs");
        let result = linsolve_builtin(
            Value::ComplexTensor(complex_lhs),
            Value::Tensor(real_rhs),
            Vec::new(),
        )
        .expect("linsolve");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor output");
        };
        assert_eq!(out.shape(), [2, 1]);
        approx_eq(out.materialize_f64()[0].0, 7.0);
        approx_eq(out.materialize_f64()[0].1, 0.0);
        approx_eq(out.materialize_f64()[1].0, 11.0);
        approx_eq(out.materialize_f64()[1].1, 0.0);
    }

    #[test]
    fn linsolve_provider_host_helper_reads_typed_integer_storage_exactly() {
        let a = Tensor::new_integer(IntegerStorage::U64(vec![2, 1, 1, 2]), vec![2, 2])
            .expect("integer lhs");
        let b =
            Tensor::new_integer(IntegerStorage::U64(vec![4, 5]), vec![2, 1]).expect("integer rhs");

        let (solution, _rcond) =
            linsolve_host_real_for_provider(&a, &b, &ProviderLinsolveOptions::default())
                .expect("provider helper");
        assert_eq!(solution.shape(), [2, 1]);
        approx_eq(solution.materialize_f64()[0], 1.0);
        approx_eq(solution.materialize_f64()[1], 2.0);
    }

    #[test]
    fn linsolve_general_real_reads_typed_integer_storage_exactly() {
        let a = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 1, 0, 0, 1]), vec![3, 2])
            .expect("integer lhs");
        let b = Tensor::new_integer(IntegerStorage::I16(vec![3, 2, 1]), vec![3, 1]).expect("rhs");

        let (solution, _rcond) =
            linsolve_host_real_for_provider(&a, &b, &ProviderLinsolveOptions::default())
                .expect("provider helper");

        assert_eq!(solution.shape(), [2, 1]);
        approx_eq(solution.materialize_f64()[0], 7.0 / 5.0);
        approx_eq(solution.materialize_f64()[1], -2.0 / 5.0);
        assert!(solution.integer_storage().is_none());
    }

    #[test]
    fn linsolve_transa_reads_typed_integer_storage_exactly() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new_integer(
            IntegerStorage::I16(vec![3, 1, 0, 0, 4, 2, 0, 0, 5]),
            vec![3, 3],
        )
        .expect("integer lhs");
        let b = Tensor::new_integer(IntegerStorage::I16(vec![5, 14, 23]), vec![3, 1])
            .expect("integer rhs");
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("LT".to_string(), Value::Bool(true));
        opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );

        let result = linsolve_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::Struct(opts.finish())],
        )
        .expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");

        assert_eq!(tensor.shape(), [3, 1]);
        let expected_a = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .expect("expected lhs");
        let expected_b = Tensor::new(vec![5.0, 14.0, 23.0], vec![3, 1]).expect("expected rhs");
        let expected_a_transposed = transpose_tensor(&expected_a);
        let (expected_tensor, _) = host_linsolve_real(
            &expected_a_transposed,
            &expected_b,
            ProviderLinsolveOptions::default(),
        );
        for (actual, expected) in tensor
            .materialize_f64()
            .iter()
            .zip(expected_tensor.materialize_f64().iter())
        {
            approx_eq(*actual, *expected);
        }
        assert!(tensor.integer_storage().is_none());
    }

    #[test]
    fn triangular_empty_systems_preserve_rhs_geometry_and_class() {
        for columns in [0, 1, 3] {
            for dtype in [NumericDType::F32, NumericDType::F64] {
                let lhs = real_solution_tensor(Vec::new(), vec![0, 0], dtype).unwrap();
                let rhs = real_solution_tensor(Vec::new(), vec![0, columns], dtype).unwrap();
                for solve in [forward_substitution_real, backward_substitution_real] {
                    let (solution, rcond) = solve(&lhs, &rhs).unwrap();
                    assert_eq!(solution.shape(), [0, columns]);
                    assert_eq!(solution.numeric_dtype(), dtype);
                    assert!(solution.materialize_f64().is_empty());
                    assert_eq!(rcond, 1.0);
                }
            }
            let lhs = ComplexTensor::new(Vec::new(), vec![0, 0]).unwrap();
            let rhs = ComplexTensor::new(Vec::new(), vec![0, columns]).unwrap();
            for solve in [forward_substitution_complex, backward_substitution_complex] {
                let (solution, rcond) = solve(&lhs, &rhs).unwrap();
                assert_eq!(solution.shape(), [0, columns]);
                assert!(solution.materialize_f64().is_empty());
                assert_eq!(rcond, 1.0);
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_lower_triangular_hint() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(
            vec![3.0, -1.0, 4.0, 0.0, 2.0, 1.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::new(vec![9.0, 1.0, 19.0], vec![3, 1]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("LT".to_string(), Value::Bool(true));
        let result = linsolve_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::Struct(opts.finish())],
        )
        .expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(tensor.shape(), [3, 1]);
        approx_eq(tensor.materialize_f64()[0], 3.0);
        approx_eq(tensor.materialize_f64()[1], 2.0);
        approx_eq(tensor.materialize_f64()[2], 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_transposed_triangular_hint() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::new(vec![5.0, 14.0, 23.0], vec![3, 1]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("LT".to_string(), Value::Bool(true));
        opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );

        let result = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(opts.finish())],
        )
        .expect("linsolve");
        let tensor = test_support::gather(result).expect("gather");
        assert_eq!(tensor.shape(), [3, 1]);

        let a_transposed = transpose_tensor(&a);
        let (expected_tensor, _) =
            host_linsolve_real(&a_transposed, &b, ProviderLinsolveOptions::default());

        for (actual, expected) in tensor
            .materialize_f64()
            .iter()
            .zip(expected_tensor.materialize_f64().iter())
        {
            approx_eq(*actual, *expected);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_complex_inputs_match_residual() {
        let a = ComplexTensor::new(
            vec![(2.0, 1.0), (-1.0, 0.0), (1.0, -2.0), (3.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::new(vec![(1.0, 0.0), (4.0, 1.0)], vec![2, 1]).unwrap();
        let result = linsolve_builtin(
            Value::ComplexTensor(a.clone()),
            Value::ComplexTensor(b.clone()),
            Vec::new(),
        )
        .expect("linsolve");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };

        let mat_a: Vec<Complex64> = a
            .materialize_f64()
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();
        let mat_b: Vec<Complex64> = b
            .materialize_f64()
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();
        let mat_x: Vec<Complex64> = out
            .materialize_f64()
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();
        let a_mat = DMatrix::from_column_slice(a.rows(), a.cols(), &mat_a);
        let b_mat = DMatrix::from_column_slice(b.rows(), b.cols(), &mat_b);
        let x_mat = DMatrix::from_column_slice(out.rows(), out.cols(), &mat_x);
        let residual = a_mat * x_mat - b_mat;
        assert!(residual.norm() < 1e-10, "residual={}", residual.norm());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_complex_conjugate_transpose_matches_explicit_reference() {
        let a = ComplexTensor::new(
            vec![(2.0, 1.0), (0.0, -1.0), (1.0, 2.0), (3.0, 0.5)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::new(vec![(1.0, -1.0), (2.0, 0.5)], vec![2, 1]).unwrap();

        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("C")),
        );
        let result = linsolve_builtin(
            Value::ComplexTensor(a.clone()),
            Value::ComplexTensor(b.clone()),
            vec![Value::Struct(opts.finish())],
        )
        .expect("linsolve");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };

        let mut a_conj_t = transpose_complex(&a);
        conjugate_complex_in_place(&mut a_conj_t);
        let reference = evaluate(
            Value::ComplexTensor(a_conj_t),
            Value::ComplexTensor(b.clone()),
            SolveOptions::default(),
        )
        .expect("reference");
        let Value::ComplexTensor(expected) = reference.solution() else {
            panic!("expected complex tensor reference");
        };

        assert_eq!(out.shape(), expected.shape());
        for ((out_re, out_im), (exp_re, exp_im)) in out
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert!(
                (out_re - exp_re).abs() < 1e-10,
                "out_re={out_re} exp_re={exp_re}"
            );
            assert!(
                (out_im - exp_im).abs() < 1e-10,
                "out_im={out_im} exp_im={exp_im}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_rcond_enforced() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![1.0, 1.0, 1.0, 1.0 + 1e-12], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 2.0 + 1e-12], vec![2, 1]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("RCOND".to_string(), Value::Num(1e-3));
        let err = unwrap_error(
            linsolve_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::Struct(opts.finish())],
            )
            .expect_err("singular matrix must fail"),
        );
        assert!(
            err.message().contains("singular to working precision"),
            "unexpected error message: {err}"
        );
        assert_eq!(err.identifier(), LINSOLVE_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_options_read_integer_tensor_storage() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 7.0], vec![2, 1]).unwrap();
        let upper = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        let rcond = Tensor::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("UT".to_string(), Value::Tensor(upper));
        opts.fields
            .insert("RCOND".to_string(), Value::Tensor(rcond));
        let result = linsolve_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::Struct(opts.finish())],
        )
        .expect("linsolve");
        let Value::Tensor(out) = result else {
            panic!("expected tensor output");
        };
        approx_eq(out.materialize_f64()[0], -1.0 / 6.0);
        approx_eq(out.materialize_f64()[1], 7.0 / 3.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_unknown_option_identifier() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("UNKNOWN".to_string(), Value::Bool(true));
        let err = unwrap_error(
            linsolve_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::Struct(opts.finish())],
            )
            .expect_err("unknown option should fail"),
        );
        assert!(err.message().contains("unknown option"));
        assert_eq!(err.identifier(), LINSOLVE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_output_count_limit_identifier() {
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let err = unwrap_error(
            linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new())
                .expect_err("three outputs should fail"),
        );
        assert!(err.message().contains("at most two outputs"));
        assert_eq!(err.identifier(), LINSOLVE_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linsolve_recovers_rcond_output() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let eval = evaluate_args(Value::Tensor(a.clone()), Value::Tensor(b.clone()), &[])
            .expect("evaluate");
        let solution_tensor = match eval.solution() {
            Value::Tensor(sol) => sol.clone(),
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather solution")
            }
            other => panic!("unexpected solution value {other:?}"),
        };
        assert_eq!(solution_tensor.shape(), [2, 1]);
        approx_eq(solution_tensor.materialize_f64()[0], 1.0);
        approx_eq(solution_tensor.materialize_f64()[1], 2.0);

        let rcond_value = match eval.reciprocal_condition() {
            Value::Num(r) => r,
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle.clone())).expect("gather rcond");
                gathered.materialize_f64()[0]
            }
            other => panic!("unexpected rcond value {other:?}"),
        };
        approx_eq(rcond_value, 1.0);
    }

    #[test]
    fn general_square_lu_preserves_forward_solution_near_svd_rank_cutoff() {
        let gap = 1e-12;
        let a = Tensor::new(vec![1.0, 1.0, 1.0, 1.0 + gap], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 2.0 + gap], vec![2, 1]).unwrap();
        let (solution, rcond, rank) = solve_general_real(&a, &b).unwrap();
        let values = solution.materialize_f64();
        assert!((values[0] - 1.0).abs() < 1e-6, "{values:?}");
        assert!((values[1] - 1.0).abs() < 1e-6, "{values:?}");
        assert!(rcond > 0.0 && rcond < 1e-8);
        assert_eq!(rank, 2.0);

        let single_a = ComplexTensor::from_f32(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (3.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let single_b = ComplexTensor::from_f32(vec![(2.0, 1.0), (3.0, -1.0)], vec![2, 1]).unwrap();
        let (single_solution, _, _) = solve_general_complex(&single_a, &single_b).unwrap();
        assert_eq!(single_solution.numeric_dtype(), NumericDType::F32);
        assert!(!binary_scale_preserves_component(1.0, 2.0_f64.powi(1023)));
        assert!(!binary_scale_preserves_component(
            f64::MAX,
            f64::MIN_POSITIVE
        ));
        let mixed_scale = Tensor::new(vec![1e308, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let mixed_rhs = Tensor::new(vec![1e308, 1.0], vec![2, 1]).unwrap();
        let (mixed_solution, _, _) = solve_general_real(&mixed_scale, &mixed_rhs).unwrap();
        assert_eq!(mixed_solution.materialize_f64(), vec![1.0, 1.0]);
        let tiny = Tensor::new(vec![f64::from_bits(1)], vec![1, 1]).unwrap();
        let unit = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let error = solve_general_real(&tiny, &unit)
            .expect_err("an unrepresentable finite-input solution must not publish infinity");
        assert_eq!(error.identifier(), LINSOLVE_ERROR_INVALID_INPUT.identifier);

        let complex_a = ComplexTensor::new(
            vec![(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0 + gap, 1.0)],
            vec![2, 2],
        )
        .unwrap();
        let complex_b = ComplexTensor::new(vec![(2.0, 2.0), (2.0 + gap, 2.0)], vec![2, 1]).unwrap();
        let (solution, rcond, rank) = solve_general_complex(&complex_a, &complex_b).unwrap();
        for (real, imaginary) in solution.materialize_f64() {
            assert!((real - 1.0).abs() < 1e-6);
            assert!(imaginary.abs() < 1e-6);
        }
        assert!(rcond > 0.0 && rcond < 1e-8);
        assert_eq!(rank, 2.0);
    }

    #[test]
    fn linsolve_rectangular_second_output_reports_rank() {
        let _accel_guard = test_support::accel_test_lock();
        clear_accel_provider_state();
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();

        let eval = evaluate_args(Value::Tensor(a), Value::Tensor(b), &[]).expect("evaluate");

        assert_eq!(eval.reciprocal_condition(), Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();

            let cpu = linsolve_builtin(
                Value::Tensor(a.clone()),
                Value::Tensor(b.clone()),
                Vec::new(),
            )
            .expect("cpu linsolve");
            let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

            let view_a = HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let view_b = HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            };
            let ha = provider.upload(&view_a).expect("upload A");
            let hb = provider.upload(&view_b).expect("upload B");

            let gpu_value = linsolve_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu linsolve");
            let gathered = test_support::gather(gpu_value).expect("gather");
            let weak_a = ha.downgrade();
            let weak_b = hb.downgrade();
            ha.clone()
                .release()
                .expect("release temporary caller alias");
            let original = block_on(provider.download(&ha)).expect("caller A survives solve");
            assert_eq!(original.shape.as_slice(), a.shape());
            assert_eq!(original.data, a.materialize_f64());
            ha.release().expect("release caller A");
            hb.release().expect("release caller B");
            assert!(weak_a.acquire().is_err());
            assert!(weak_b.acquire().is_err());

            assert_eq!(gathered.shape(), cpu_tensor.shape());
            assert_eq!(gathered.len(), cpu_tensor.len());
            for (gpu, cpu) in gathered
                .materialize_f64()
                .iter()
                .zip(cpu_tensor.materialize_f64().iter())
            {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn host_inputs_auto_promote_into_provider_solve_path() {
        test_support::with_test_provider(|provider| {
            provider.reset_telemetry();
            let a = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
            let _ = linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new())
                .expect("host linsolve");
            let telemetry = provider.telemetry_snapshot();
            assert!(telemetry.linsolve.count >= 1);
            assert!(fallback_count(&telemetry, "linsolve:host_reupload") >= 1);
            assert!(telemetry.upload_bytes > 0);
            assert!(telemetry.download_bytes > 0);
        });
    }

    #[test]
    fn typed_integer_host_inputs_keep_the_double_solve_boundary_on_host() {
        test_support::with_test_provider(|provider| {
            provider.reset_telemetry();
            let a = Tensor::new_integer(IntegerStorage::U64(vec![2, 1, 1, 2]), vec![2, 2])
                .expect("integer lhs");
            let b = Tensor::new_integer(IntegerStorage::U64(vec![4, 5]), vec![2, 1])
                .expect("integer rhs");

            let result = linsolve_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new())
                .expect("host provider linsolve");
            let tensor = test_support::gather(result).expect("gather");
            assert_eq!(tensor.shape(), [2, 1]);
            approx_eq(tensor.materialize_f64()[0], 1.0);
            approx_eq(tensor.materialize_f64()[1], 2.0);

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.linsolve.count, 0);
            assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        });
    }

    #[test]
    fn provider_telemetry_records_gpu_host_reupload_path() {
        test_support::with_test_provider(|provider| {
            provider.reset_telemetry();
            let a = Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
            let ha = provider
                .upload(&HostTensorView {
                    data: &a.materialize_f64(),
                    shape: a.shape(),
                })
                .expect("upload A");
            let hb = provider
                .upload(&HostTensorView {
                    data: &b.materialize_f64(),
                    shape: b.shape(),
                })
                .expect("upload B");

            let _ = linsolve_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu linsolve");

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.linsolve.count, 1);
            assert!(telemetry.upload_bytes > 0);
            assert!(telemetry.download_bytes > 0);
            assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 1);

            ha.release().expect("release caller A");
            hb.release().expect("release caller B");
        });
    }

    #[test]
    fn scalar_gpu_inputs_fall_back_without_provider_solve_dispatch() {
        test_support::with_test_provider(|provider| {
            provider.reset_telemetry();
            let a = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
            let b = Tensor::new(vec![6.0], vec![1, 1]).unwrap();
            let ha = provider
                .upload(&HostTensorView {
                    data: &a.materialize_f64(),
                    shape: a.shape(),
                })
                .expect("upload A");
            let hb = provider
                .upload(&HostTensorView {
                    data: &b.materialize_f64(),
                    shape: b.shape(),
                })
                .expect("upload B");

            let result = linsolve_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("fallback linsolve");
            let gathered = test_support::gather(result).expect("gather fallback");
            assert_eq!(gathered.materialize_f64(), vec![3.0]);

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.linsolve.count, 0);
            assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
            assert!(telemetry.download_bytes > 0);

            ha.release().expect("release caller A");
            hb.release().expect("release caller B");
        });
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_square_linsolve_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();

        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu square linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4);
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_posdef_chol"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 1);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_square_linsolve_uses_device_path_without_output_count() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();

        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu square linsolve");
        let gathered = test_support::gather(gpu_value).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 1);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_square_linsolve_recovers_rcond_output_on_device() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();

        let (_, cpu_rcond) = host_linsolve_real(&a, &b, ProviderLinsolveOptions::default());
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu square linsolve");
        let outputs = match gpu_value {
            Value::OutputList(outputs) => outputs,
            other => panic!("expected output list, got {other:?}"),
        };
        assert_eq!(outputs.len(), 2);
        let gathered = test_support::gather(outputs[0].clone()).expect("gather");
        let gpu_rcond = match &outputs[1] {
            Value::Num(value) => *value,
            other => panic!("unexpected gpu rcond {other:?}"),
        };
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), [2, 1]);
        assert!(
            (gpu_rcond - cpu_rcond).abs() < 1e-4,
            "gpu={gpu_rcond} cpu={cpu_rcond}"
        );

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 1);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_square_linsolve_with_rcond_option_stays_on_device() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }

        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();
        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts
            .fields
            .insert("RCOND".to_string(), Value::Num(0.05));
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts
            .fields
            .insert("RCOND".to_string(), Value::Num(0.05));
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu square linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 1);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_tall_linsolve_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 2.0], vec![3, 1]).unwrap();

        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu tall linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_posdef_linsolve_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();

        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts
            .fields
            .insert("POSDEF".to_string(), Value::Bool(true));
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        let (_, cpu_rcond) = host_linsolve_real(
            &a,
            &b,
            ProviderLinsolveOptions {
                posdef: true,
                ..Default::default()
            },
        );
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts
            .fields
            .insert("POSDEF".to_string(), Value::Bool(true));
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu posdef linsolve");
        let outputs = match gpu_value {
            Value::OutputList(outputs) => outputs,
            other => panic!("expected output list, got {other:?}"),
        };
        let mut outputs = outputs
            .try_into_owned()
            .expect("unique output list")
            .into_iter();
        let gpu_solution = outputs.next().expect("solution output");
        let gpu_rcond = match outputs.next().expect("condition output") {
            Value::Num(value) => value,
            other => panic!("unexpected rcond value {other:?}"),
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }
        assert!(
            (gpu_rcond - cpu_rcond).abs() < 1e-4,
            "gpu={gpu_rcond} cpu={cpu_rcond}"
        );

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_posdef_chol"), 1);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_transposed_posdef_linsolve_uses_cholesky_path() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![6.0, 2.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![8.0, 9.0], vec![2, 1]).unwrap();

        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts
            .fields
            .insert("POSDEF".to_string(), Value::Bool(true));
        cpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts
            .fields
            .insert("POSDEF".to_string(), Value::Bool(true));
        gpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu transposed posdef linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_posdef_chol"), 1);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_symmetric_linsolve_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![5.0, 2.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![9.0, 8.0], vec![2, 1]).unwrap();

        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts.fields.insert("SYM".to_string(), Value::Bool(true));
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts.fields.insert("SYM".to_string(), Value::Bool(true));
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu symmetric linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_transposed_square_linsolve_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 14.0], vec![2, 1]).unwrap();

        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu transposed square linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_conjugate_square_linsolve_avoids_host_reupload_fallback_for_real_inputs() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }

        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 14.0], vec![2, 1]).unwrap();
        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("C")),
        );
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("C")),
        );
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu conjugate square linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
        assert_eq!(kernel_launch_count(&telemetry, "linsolve_tall_qr"), 1);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_transposed_rectangular_linsolve_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
            return;
        }
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 2.0], vec![3, 1]).unwrap();

        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        cpu_opts
            .fields
            .insert("RECT".to_string(), Value::Bool(true));
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        gpu_opts
            .fields
            .insert("RECT".to_string(), Value::Bool(true));
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu transposed rectangular linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-4, "gpu={gpu} cpu={cpu}");
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_triangular_hint_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let a = Tensor::new(
            vec![3.0, -1.0, 4.0, 0.0, 2.0, 1.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::new(vec![9.0, 1.0, 19.0], vec![3, 1]).unwrap();

        let cpu = linsolve_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()), {
            let mut opts = runmat_value::UnboundedStructValueBuilder::new();
            opts.fields.insert("LT".to_string(), Value::Bool(true));
            vec![Value::Struct(opts.finish())]
        })
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut opts = runmat_value::UnboundedStructValueBuilder::new();
        opts.fields.insert("LT".to_string(), Value::Bool(true));
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(opts.finish())],
        )
        .expect("gpu triangular linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-5);
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_transposed_triangular_hint_avoids_host_reupload_fallback() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let a = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 5.0],
            vec![3, 3],
        )
        .unwrap();
        let b = Tensor::new(vec![5.0, 14.0, 23.0], vec![3, 1]).unwrap();

        let mut cpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        cpu_opts.fields.insert("LT".to_string(), Value::Bool(true));
        cpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            vec![Value::Struct(cpu_opts.finish())],
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");
        provider.reset_telemetry();

        let ha = provider
            .upload(&HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            })
            .expect("upload A");
        let hb = provider
            .upload(&HostTensorView {
                data: &b.materialize_f64(),
                shape: b.shape(),
            })
            .expect("upload B");

        let _output_guard =
            crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let mut gpu_opts = runmat_value::UnboundedStructValueBuilder::new();
        gpu_opts.fields.insert("LT".to_string(), Value::Bool(true));
        gpu_opts.fields.insert(
            "TRANSA".to_string(),
            Value::CharArray(CharArray::new_row("T")),
        );
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            vec![Value::Struct(gpu_opts.finish())],
        )
        .expect("gpu transposed triangular linsolve");
        let gpu_solution = match gpu_value {
            Value::OutputList(outputs) => first_output(outputs),
            other => other,
        };
        let gathered = test_support::gather(gpu_solution).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < 1e-5);
        }

        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.linsolve.count, 1);
        assert_eq!(fallback_count(&telemetry, "linsolve:host_reupload"), 0);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wgpu_round_trip_matches_cpu() {
        let _accel_guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };

        let a = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();

        let cpu = linsolve_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu linsolve");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

        let view_a = HostTensorView {
            data: &a.materialize_f64(),
            shape: a.shape(),
        };
        let view_b = HostTensorView {
            data: &b.materialize_f64(),
            shape: b.shape(),
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let gpu_value = linsolve_builtin(
            Value::GpuTensor(ha.clone()),
            Value::GpuTensor(hb.clone()),
            Vec::new(),
        )
        .expect("gpu linsolve");
        let gathered = test_support::gather(gpu_value).expect("gather");
        ha.release().expect("release caller A");
        hb.release().expect("release caller B");

        assert_eq!(gathered.shape(), cpu_tensor.shape());
        for (gpu, cpu) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu_tensor.materialize_f64().iter())
        {
            assert!((gpu - cpu).abs() < tol);
        }
    }
}
