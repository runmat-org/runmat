//! MATLAB-compatible `lsqcurvefit` builtin for nonlinear curve fitting.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, LSQCURVEFIT_DESCRIPTOR,
    LSQCURVEFIT_ERROR_INVALID_ARGUMENT, LSQCURVEFIT_ERROR_INVALID_INPUT,
    LSQCURVEFIT_ERROR_TOO_MANY_OUTPUTS, LSQCURVEFIT_INTEGER_BOUND_EXTENSION,
    LSQCURVEFIT_INTEGER_CALLBACK_EXTENSION, LSQCURVEFIT_INTEGER_OPTION_EXTENSION,
    LSQCURVEFIT_INTEGER_X0_EXTENSION, LSQCURVEFIT_INTEGER_XDATA_EXTENSION,
    LSQCURVEFIT_INTEGER_YDATA_EXTENSION, LSQCURVEFIT_LOGICAL_NUMERIC_EXTENSION,
    LSQCURVEFIT_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::math::optim::common::{
    call_function_in_context, initial_guess_with_cooperation,
};
use crate::builtins::math::optim::least_squares::{
    solve_least_squares, LeastSquaresBounds, LeastSquaresEvaluator, LeastSquaresOptions,
    LeastSquaresResult, ResidualFuture,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "lsqcurvefit";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;
const DEFAULT_MAX_FUN_EVALS_FACTOR: usize = 100;
const ALGORITHM: &str = "levenberg-marquardt";

fn lsq_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("lsqcurvefit:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = lsq_error_with_detail(fallback, err.message());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.failure_origin = err.failure_origin;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "lsqcurvefit",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::lsqcurvefit"
)]
async fn lsqcurvefit_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    xdata: Value,
    ydata: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    lsqcurvefit_in_runtime(runtime.as_ref(), function, x0, xdata, ydata, rest).await
}

#[cfg(test)]
async fn lsqcurvefit_builtin(
    function: Value,
    x0: Value,
    xdata: Value,
    ydata: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    lsqcurvefit_in_runtime(None, function, x0, xdata, ydata, rest.into()).await
}

async fn lsqcurvefit_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    xdata: Value,
    ydata: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    validate_requested_outputs()?;
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqcurvefit,
        runtime,
    );
    cooperation.completed(0).await?;
    ensure_input_extensions(&x0, &xdata, &ydata, rest.view())?;
    let _option_sources = rest.fork();
    let mut parsed = ParsedArgs::parse(rest)
        .await
        .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_ARGUMENT))?;
    let prepared_options = crate::builtins::math::optim::common::prepare_solver_options(
        runmat_builtins::OptimSolver::Lsqcurvefit,
        parsed.options.take(),
        Some(&LSQCURVEFIT_INTEGER_OPTION_EXTENSION),
        Some(&LSQCURVEFIT_RESIDENT_INPUT_EXTENSION),
        runtime,
        &mut cooperation,
    )
    .await
    .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_ARGUMENT))?;
    crate::builtins::math::optim::common::preflight_real_array(NAME, &x0, "initial point", false)?;
    let x0 = prepare_floating_input(
        "initial point",
        x0,
        &LSQCURVEFIT_INTEGER_X0_EXTENSION,
        &mut cooperation,
    )
    .await
    .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_INPUT))?;
    let guess = initial_guess_with_cooperation(NAME, x0, &mut cooperation)
        .await
        .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_INPUT))?;
    let n = guess.values.len();
    let options =
        LsqOptions::from_struct(prepared_options.as_ref().map(|options| options.structure()))
            .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_ARGUMENT))?;
    let bounds = parsed
        .bounds(n, &mut cooperation)
        .await
        .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_ARGUMENT))?;
    let ydata = real_array_checked(
        "ydata",
        ydata,
        &LSQCURVEFIT_INTEGER_YDATA_EXTENSION,
        &mut cooperation,
    )
    .await
    .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_INPUT))?;
    if ydata.values.is_empty() {
        return Err(lsq_error_with_detail(
            &LSQCURVEFIT_ERROR_INVALID_INPUT,
            "ydata must not be empty",
        ));
    }
    let mut evaluator = CurveFitEvaluator {
        cooperation,
        function,
        x_shape: guess.shape.clone(),
        x_scalar: guess.scalar,
        xdata,
        ydata,
    };
    let result = solve_least_squares(NAME, &mut evaluator, guess.values, &bounds, &options.solver)
        .await
        .map_err(|err| map_error(err, &LSQCURVEFIT_ERROR_INVALID_INPUT))?;
    finalize(
        result,
        &guess.shape,
        guess.scalar,
        &evaluator.ydata.shape,
        &bounds,
        &options.algorithm,
        &mut evaluator.cooperation,
    )
    .await
    .map_err(|error| map_error(error, &LSQCURVEFIT_ERROR_INVALID_INPUT))
}

fn ensure_input_extensions(
    x0: &Value,
    xdata: &Value,
    ydata: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    use crate::builtins::common::validation::{
        value_contains_explicit_gpu, value_contains_native_integer_class,
    };
    if value_contains_native_integer_class(x0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LSQCURVEFIT_INTEGER_X0_EXTENSION,
            NAME,
        )?;
    }
    if value_contains_native_integer_class(xdata) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LSQCURVEFIT_INTEGER_XDATA_EXTENSION,
            NAME,
        )?;
    }
    if value_contains_native_integer_class(ydata) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LSQCURVEFIT_INTEGER_YDATA_EXTENSION,
            NAME,
        )?;
    }
    if rest.len() >= 2 {
        for value in rest.iter().take(2) {
            if value_contains_native_integer_class(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LSQCURVEFIT_INTEGER_BOUND_EXTENSION,
                    NAME,
                )?;
            }
        }
    }
    if is_logical_numeric(x0) || is_logical_numeric(xdata) || is_logical_numeric(ydata) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LSQCURVEFIT_LOGICAL_NUMERIC_EXTENSION,
            NAME,
        )?;
    }
    if value_contains_explicit_gpu(x0)
        || value_contains_explicit_gpu(xdata)
        || value_contains_explicit_gpu(ydata)
        || rest.iter().any(value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LSQCURVEFIT_RESIDENT_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn is_logical_numeric(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn preflight_floating_input(
    role: &str,
    value: &Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_contains_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer_extension, NAME)?;
    }
    if is_logical_numeric(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LSQCURVEFIT_LOGICAL_NUMERIC_EXTENSION,
            NAME,
        )?;
    }
    crate::builtins::math::optim::common::preflight_real_array(NAME, value, role, true)
}

async fn prepare_floating_input(
    role: &str,
    value: Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Value> {
    preflight_floating_input(role, &value, integer_extension)?;
    crate::builtins::math::optim::common::gather_exact_floating_value_with_cooperation(
        NAME,
        value,
        role,
        cooperation,
    )
    .await
}

fn validate_requested_outputs() -> BuiltinResult<()> {
    crate::output_count::validate_bounded_outputs(
        &LSQCURVEFIT_DESCRIPTOR,
        &LSQCURVEFIT_ERROR_TOO_MANY_OUTPUTS,
        NAME,
    )
}

struct ParsedArgs {
    lb: Option<Value>,
    ub: Option<Value>,
    options: Option<StructValue>,
}

impl ParsedArgs {
    async fn parse(rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        match rest.len() {
            0 => Ok(Self {
                lb: None,
                ub: None,
                options: None,
            }),
            1 => match rest.into_iter().next().unwrap() {
                Value::Struct(options) => Ok(Self {
                    lb: None,
                    ub: None,
                    options: Some(options),
                }),
                other => Err(lsq_error_with_detail(
                    &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                    format!("single optional argument must be options struct, got {other:?}"),
                )),
            },
            2 | 3 => {
                let mut values = rest.into_iter();
                let lb = values.next();
                let ub = values.next();
                let options = match values.next() {
                    None => None,
                    Some(Value::Struct(options)) => Some(options),
                    Some(other) => {
                        return Err(lsq_error_with_detail(
                            &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                            format!("options must be a struct, got {other:?}"),
                        ))
                    }
                };
                Ok(Self { lb, ub, options })
            }
            _ => Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                "linear and nonlinear constraint forms are not supported yet",
            )),
        }
    }

    async fn bounds(
        self,
        n: usize,
        cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
    ) -> BuiltinResult<LeastSquaresBounds> {
        let mut bounds = LeastSquaresBounds::unbounded(n);
        if let Some(lb) = self.lb {
            bounds.lower = bound_vector_with_cooperation(
                "lower bounds",
                lb,
                n,
                f64::NEG_INFINITY,
                cooperation,
            )
            .await?;
        }
        if let Some(ub) = self.ub {
            bounds.upper =
                bound_vector_with_cooperation("upper bounds", ub, n, f64::INFINITY, cooperation)
                    .await?;
        }
        bounds.validate(NAME, n)?;
        Ok(bounds)
    }
}

struct LsqOptions {
    solver: LeastSquaresOptions,
    algorithm: String,
}

impl LsqOptions {
    fn from_struct(options: Option<&StructValue>) -> BuiltinResult<Self> {
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(options);
        let display = fields.text(runmat_builtins::OptimOptionField::Display, "off")?;
        if !matches!(
            runmat_builtins::OptimDisplay::parse(&display),
            Some(
                runmat_builtins::OptimDisplay::Off
                    | runmat_builtins::OptimDisplay::None
                    | runmat_builtins::OptimDisplay::Final
                    | runmat_builtins::OptimDisplay::Iter
            )
        ) {
            return Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                "option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let algorithm = fields.text(runmat_builtins::OptimOptionField::Algorithm, ALGORITHM)?;
        let algorithm = algorithm.to_ascii_lowercase();
        if !matches!(
            runmat_builtins::OptimAlgorithm::parse(&algorithm),
            Some(
                runmat_builtins::OptimAlgorithm::LevenbergMarquardt
                    | runmat_builtins::OptimAlgorithm::TrustRegionReflective
            )
        ) {
            return Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                "option Algorithm must be 'levenberg-marquardt' or 'trust-region-reflective'",
            ));
        }
        let tol_x = fields.real(NAME, runmat_builtins::OptimOptionField::TolX, DEFAULT_TOL_X)?;
        let tol_fun = fields.real(
            NAME,
            runmat_builtins::OptimOptionField::TolFun,
            DEFAULT_TOL_FUN,
        )?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter = fields
            .count(
                NAME,
                runmat_builtins::OptimOptionField::MaxIter,
                DEFAULT_MAX_ITER,
            )?
            .max(1);
        let max_fun_evals = fields
            .count_product_default(
                NAME,
                runmat_builtins::OptimOptionField::MaxFunEvals,
                DEFAULT_MAX_FUN_EVALS_FACTOR,
                max_iter,
            )?
            .max(1);
        Ok(Self {
            solver: LeastSquaresOptions {
                criteria: None,
                tol_x,
                tol_fun,
                max_iter,
                max_fun_evals,
                final_gradient: true,
                final_jacobian: true,
                scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields(NAME,&fields,true)?,
            },
            algorithm,
        })
    }
}

struct CurveFitEvaluator {
    cooperation: crate::builtins::math::optim::common::ScalarCooperation,
    function: Value,
    x_shape: Vec<usize>,
    x_scalar: bool,
    xdata: Value,
    ydata: RealArray,
}

impl LeastSquaresEvaluator for CurveFitEvaluator {
    fn cooperation(
        &mut self,
    ) -> Option<&mut crate::builtins::math::optim::common::ScalarCooperation> {
        Some(&mut self.cooperation)
    }
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(async move {
            let arg = x_value(x, &self.x_shape, self.x_scalar)?;
            let value = call_function_in_context(
                self.cooperation.runtime(),
                &self.function,
                [arg, self.xdata.clone()],
                1,
            )
            .await?;
            let model = value
                .with_owned_async(|value| {
                    real_array_checked(
                        "model output",
                        value,
                        &LSQCURVEFIT_INTEGER_CALLBACK_EXTENSION,
                        &mut self.cooperation,
                    )
                })
                .await?;
            if model.shape != self.ydata.shape {
                return Err(lsq_error_with_detail(
                    &LSQCURVEFIT_ERROR_INVALID_INPUT,
                    format!(
                        "model output shape {:?} must match ydata shape {:?}",
                        model.shape, self.ydata.shape
                    ),
                ));
            }
            let mut residual = model.values;
            for (model, observed) in residual
                .chunks_mut(4096)
                .zip(self.ydata.values.chunks(4096))
            {
                for (model, observed) in model.iter_mut().zip(observed) {
                    *model -= observed;
                }
                self.cooperation.completed(model.len() as u64).await?;
            }
            Ok(residual)
        })
    }
}

#[derive(Clone)]
struct RealArray {
    values: Vec<f64>,
    shape: Vec<usize>,
}

#[cfg(test)]
async fn real_array(label: &str, value: Value) -> BuiltinResult<RealArray> {
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqcurvefit,
        None,
    );
    real_array_with_cooperation(label, value, &mut cooperation).await
}

async fn real_array_with_cooperation(
    label: &str,
    value: Value,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<RealArray> {
    crate::builtins::math::optim::common::preflight_real_array(NAME, &value, label, true)?;
    let value = crate::dispatcher::gather_if_needed_async(&value).await?;
    match value {
        Value::Num(n) => finite_array(label, vec![n], vec![1, 1], cooperation).await,
        Value::Int(i) => finite_array(label, vec![i.to_f64()], vec![1, 1], cooperation).await,
        Value::Bool(flag) => {
            finite_array(
                label,
                vec![if flag { 1.0 } else { 0.0 }],
                vec![1, 1],
                cooperation,
            )
            .await
        }
        Value::Tensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let values =
                crate::builtins::math::optim::common::tensor_values(&tensor, cooperation).await?;
            finite_array(label, values, shape, cooperation).await
        }
        Value::LogicalArray(LogicalArray { data, shape }) => {
            finite_array(
                label,
                crate::builtins::math::optim::common::logical_values(&data, cooperation).await?,
                shape.as_slice().to_vec(),
                cooperation,
            )
            .await
        }
        other => Err(lsq_error_with_detail(
            &LSQCURVEFIT_ERROR_INVALID_INPUT,
            format!("{label} must be real numeric, got {other:?}"),
        )),
    }
}

async fn real_array_checked(
    label: &str,
    value: Value,
    integer_extension: &'static BuiltinExtensionDescriptor,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<RealArray> {
    let value = prepare_floating_input(label, value, integer_extension, cooperation).await?;
    real_array_with_cooperation(label, value, cooperation).await
}

async fn finite_array(
    label: &str,
    values: Vec<f64>,
    shape: Vec<usize>,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<RealArray> {
    if let Some(value) =
        crate::builtins::math::optim::common::first_nonfinite(&values, cooperation).await?
    {
        return Err(lsq_error_with_detail(
            &LSQCURVEFIT_ERROR_INVALID_INPUT,
            format!("{label} must be finite, got {value}"),
        ));
    }
    Ok(RealArray { values, shape })
}

fn x_value(x: &[f64], shape: &[usize], scalar: bool) -> BuiltinResult<Value> {
    if scalar {
        Ok(Value::Num(x[0]))
    } else {
        Tensor::new(x.to_vec(), shape.to_vec())
            .map(Value::Tensor)
            .map_err(|err| lsq_error_with_detail(&LSQCURVEFIT_ERROR_INVALID_INPUT, err))
    }
}

#[cfg(test)]
async fn bound_vector(
    label: &str,
    value: Value,
    n: usize,
    default: f64,
) -> BuiltinResult<Vec<f64>> {
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Lsqcurvefit,
        None,
    );
    bound_vector_with_cooperation(label, value, n, default, &mut cooperation).await
}

async fn bound_vector_with_cooperation(
    label: &str,
    value: Value,
    n: usize,
    default: f64,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    preflight_floating_input(label, &value, &LSQCURVEFIT_INTEGER_BOUND_EXTENSION)?;
    if crate::value_fact::value_fact(&value)
        .shape
        .element_count()
        .is_some_and(|count| count != 0 && count != 1 && count != n)
    {
        return Err(lsq_error_with_detail(
            &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
            format!("{label} must be scalar, empty, or match x0 length"),
        ));
    }
    let value = prepare_floating_input(
        label,
        value,
        &LSQCURVEFIT_INTEGER_BOUND_EXTENSION,
        cooperation,
    )
    .await?;
    if is_empty_value(&value) {
        return Ok(vec![default; n]);
    }
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(flag) => vec![if flag { 1.0 } else { 0.0 }],
        Value::Tensor(tensor) => {
            crate::builtins::math::optim::common::tensor_values(&tensor, cooperation).await?
        }
        Value::LogicalArray(LogicalArray { data, .. }) => {
            crate::builtins::math::optim::common::logical_values(&data, cooperation).await?
        }
        other => {
            return Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                format!("{label} must be real numeric or empty, got {other:?}"),
            ))
        }
    };
    let out = if values.len() == 1 && n > 1 {
        vec![values[0]; n]
    } else if values.len() == n {
        values
    } else {
        return Err(lsq_error_with_detail(
            &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
            format!("{label} must be scalar, empty, or match x0 length"),
        ));
    };
    for chunk in out.chunks(4096) {
        if chunk.iter().any(|value| value.is_nan()) {
            return Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_INVALID_ARGUMENT,
                format!("{label} must not contain NaN"),
            ));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(out)
}

fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor::tensor_element_len(tensor) == 0,
        Value::LogicalArray(LogicalArray { data, .. }) => data.is_empty(),
        _ => false,
    }
}

async fn finalize(
    result: LeastSquaresResult,
    x_shape: &[usize],
    x_scalar: bool,
    residual_shape: &[usize],
    bounds: &LeastSquaresBounds,
    algorithm: &str,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Value> {
    let requested = match crate::output_count::current_output_count() {
        None => return x_value(&result.x, x_shape, x_scalar),
        Some(0) => return Ok(Value::output_list(Vec::new())),
        Some(count @ 1..=7) => count,
        Some(_) => {
            return Err(lsq_error_with_detail(
                &LSQCURVEFIT_ERROR_TOO_MANY_OUTPUTS,
                "lsqcurvefit: too many output arguments; maximum is 7",
            ))
        }
    };
    let mut outputs = Vec::with_capacity(requested);
    outputs.push(x_value(&result.x, x_shape, x_scalar)?);
    if requested >= 2 {
        outputs.push(Value::Num(result.resnorm));
    }
    if requested >= 3 {
        outputs.push(tensor_value(
            result.residual.clone(),
            residual_shape.to_vec(),
        )?);
    }
    if requested >= 4 {
        outputs.push(Value::Num(result.exitflag as f64));
    }
    if requested >= 5 {
        outputs.push(Value::Struct(output_struct(&result, algorithm)));
    }
    if requested >= 6 {
        outputs.push(Value::Struct(
            lambda_struct(&result, x_shape, x_scalar, bounds, cooperation).await?,
        ));
    }
    if requested >= 7 {
        outputs.push(
            crate::builtins::math::optim::least_squares::jacobian_value(NAME, &result, cooperation)
                .await?,
        );
    }
    Ok(crate::output_count::output_list_with_padding(
        requested, outputs,
    ))
}

fn tensor_value(values: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|err| lsq_error_with_detail(&LSQCURVEFIT_ERROR_INVALID_INPUT, err))
}

fn output_struct(result: &LeastSquaresResult, algorithm: &str) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("firstorderopt", Value::Num(result.first_order_optimality));
    fields.insert("algorithm", Value::from(algorithm));
    fields.insert("message", Value::from(result.message.clone()));
    fields.finish()
}

async fn lambda_struct(
    result: &LeastSquaresResult,
    x_shape: &[usize],
    x_scalar: bool,
    bounds: &LeastSquaresBounds,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<StructValue> {
    let gradient =
        crate::builtins::math::optim::least_squares::bound_gradient(result, cooperation).await?;
    let mut lower = vec![0.0; result.variable_len];
    let mut upper = vec![0.0; result.variable_len];
    for i in 0..result.variable_len {
        if bounds.lower[i].is_finite()
            && (result.x[i] - bounds.lower[i]).abs() <= 1.0e-8 * (1.0 + result.x[i].abs())
        {
            lower[i] = gradient[i].max(0.0);
        }
        if bounds.upper[i].is_finite()
            && (result.x[i] - bounds.upper[i]).abs() <= 1.0e-8 * (1.0 + result.x[i].abs())
        {
            upper[i] = (-gradient[i]).max(0.0);
        }
    }
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("lower", multiplier_value(lower, x_shape, x_scalar)?);
    fields.insert("upper", multiplier_value(upper, x_shape, x_scalar)?);
    Ok(fields.finish())
}

fn multiplier_value(values: Vec<f64>, x_shape: &[usize], x_scalar: bool) -> BuiltinResult<Value> {
    if x_scalar {
        Ok(Value::Num(values[0]))
    } else {
        tensor_value(values, x_shape.to_vec())
    }
}

#[cfg(test)]
mod tests;
