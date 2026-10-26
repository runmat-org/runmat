//! MATLAB-compatible `fsolve` builtin for nonlinear systems.

use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    FSOLVE_CALLBACK_NUMERIC_EXTENSION, FSOLVE_DESCRIPTOR, FSOLVE_ERROR_INVALID_ARGUMENT,
    FSOLVE_ERROR_INVALID_INPUT, FSOLVE_INPUT_NUMERIC_EXTENSION, FSOLVE_OPTION_EXTENSION,
    FSOLVE_RESIDENT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

use crate::builtins::math::optim::common::{
    call_function_in_context, value_to_real_vector_with_cooperation, vector_to_value,
};
use crate::builtins::math::optim::least_squares::{
    solve_least_squares, LeastSquaresBounds, LeastSquaresEvaluator, LeastSquaresOptions,
    LeastSquaresResult, ResidualFuture,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(in crate::builtins::math::optim) mod compiled;
mod jacobian;
mod sparse_jacobian;

const NAME: &str = "fsolve";
const DEFAULT_TOL_X: f64 = 1.0e-6;
const DEFAULT_TOL_FUN: f64 = 1.0e-6;
const DEFAULT_MAX_ITER: usize = 400;

fn fsolve_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fsolve:") {
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

fn fsolve_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = fsolve_error_with_detail(fallback, err.message());
        mapped.gpu_gather_retry = err.gpu_gather_retry;
        mapped.failure_origin = err.failure_origin;
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "fsolve",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fsolve"
)]
async fn fsolve_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    fsolve_in_runtime(runtime.as_ref(), function, x0, rest).await
}

#[cfg(test)]
async fn fsolve_builtin(function: Value, x0: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    fsolve_in_runtime(None, function, x0, rest.into()).await
}

async fn fsolve_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    function: Value,
    x0: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME, "finite host-memory execution is unsupported; this optimizer has no complete workspace and factorization bound")?;
    crate::output_count::validate_bounded_outputs(
        &FSOLVE_DESCRIPTOR,
        &FSOLVE_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
        runmat_builtins::OptimSolver::Fsolve,
        runtime,
    );
    cooperation.completed(0).await?;
    if rest.len() > 1 {
        return Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }
    let options = parse_options(rest.first())
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let options = crate::builtins::math::optim::common::prepare_solver_options(
        runmat_builtins::OptimSolver::Fsolve,
        options,
        Some(&FSOLVE_OPTION_EXTENSION),
        Some(&FSOLVE_RESIDENT_EXTENSION),
        runtime,
        &mut cooperation,
    )
    .await
    .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let opts = FsolveOptions::from_struct(options.as_ref().map(|options| options.structure()))
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_ARGUMENT))?;
    let guess = crate::builtins::math::optim::common::initial_guess_with_extensions(
        NAME,
        x0,
        &FSOLVE_INPUT_NUMERIC_EXTENSION,
        &FSOLVE_RESIDENT_EXTENSION,
        &mut cooperation,
    )
    .await
    .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))?;
    let outcome = solve(
        &function,
        guess.values,
        &guess.shape,
        guess.scalar,
        &opts,
        &mut cooperation,
    )
    .await
    .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))?;
    finalize(outcome, &guess.shape, guess.scalar, &mut cooperation)
        .await
        .map_err(|err| fsolve_map_error(err, &FSOLVE_ERROR_INVALID_INPUT))
}

fn parse_options(value: Option<&Value>) -> BuiltinResult<Option<StructValue>> {
    match value {
        None => Ok(None),
        Some(Value::Struct(options)) => Ok(Some(options.clone())),
        Some(other) => Err(fsolve_error_with_detail(
            &FSOLVE_ERROR_INVALID_ARGUMENT,
            format!("options must be a struct, got {other:?}"),
        )),
    }
}

#[derive(Clone)]
struct FsolveOptions {
    tol_x: f64,
    tol_fun: f64,
    max_iter: usize,
    max_fun_evals: usize,
    specify_objective_gradient: bool,
    scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
}

impl FsolveOptions {
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
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "option Display must be 'off', 'none', 'final', or 'iter'",
            ));
        }
        let tol_x = fields.real(NAME, runmat_builtins::OptimOptionField::TolX, DEFAULT_TOL_X)?;
        let tol_fun = fields.real(
            NAME,
            runmat_builtins::OptimOptionField::TolFun,
            DEFAULT_TOL_FUN,
        )?;
        if tol_x <= 0.0 || tol_fun <= 0.0 {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "options TolX and TolFun must be positive",
            ));
        }
        let max_iter = fields.count(
            NAME,
            runmat_builtins::OptimOptionField::MaxIter,
            DEFAULT_MAX_ITER,
        )?;
        let max_fun_evals = fields.count_product_default(
            NAME,
            runmat_builtins::OptimOptionField::MaxFunEvals,
            100,
            max_iter.max(1),
        )?;
        if max_fun_evals == 0 {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "MaxFunEvals must be positive",
            ));
        }
        Ok(Self {
            tol_x,
            tol_fun,
            max_iter,
            max_fun_evals,
            scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling::from_fields(NAME,&fields,!fields.switch(NAME,runmat_builtins::OptimOptionField::SpecifyObjectiveGradient)?)?,
            specify_objective_gradient: fields.switch(
                NAME,
                runmat_builtins::OptimOptionField::SpecifyObjectiveGradient,
            )?,
        })
    }
}

async fn solve(
    function: &Value,
    x: Vec<f64>,
    shape: &[usize],
    scalar: bool,
    options: &FsolveOptions,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<FsolveOutcome> {
    let final_jacobian =
        crate::output_count::current_output_count().is_some_and(|count| count >= 5);
    let mut evaluator = FsolveEvaluator {
        cooperation,
        function,
        shape: shape.to_vec(),
        scalar,
        residual_shape: None,
        residual_scalar: false,
        specify_objective_gradient: options.specify_objective_gradient,
        jacobian: None,
        accepted_jacobian: None,
    };
    let variable_len = x.len();
    let mut result = solve_least_squares(
        NAME,
        &mut evaluator,
        x,
        &LeastSquaresBounds::unbounded(variable_len),
        &LeastSquaresOptions {
            criteria: None,
            tol_x: options.tol_x,
            tol_fun: options.tol_fun,
            max_iter: options.max_iter,
            max_fun_evals: options.max_fun_evals,
            final_gradient: final_jacobian,
            final_jacobian,
            scaling: options.scaling.clone(),
        },
    )
    .await?;
    crate::builtins::math::optim::least_squares::equation::require_root(
        &mut result,
        options.tol_fun,
        Some(&mut *evaluator.cooperation),
    )
    .await?;
    Ok(FsolveOutcome {
        result,
        residual_shape: evaluator.residual_shape.unwrap_or_else(|| vec![1, 1]),
        residual_scalar: evaluator.residual_scalar,
    })
}

struct FsolveOutcome {
    result: LeastSquaresResult,
    residual_shape: Vec<usize>,
    residual_scalar: bool,
}

struct FsolveEvaluator<'a> {
    cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    function: &'a Value,
    shape: Vec<usize>,
    scalar: bool,
    residual_shape: Option<Vec<usize>>,
    residual_scalar: bool,
    specify_objective_gradient: bool,
    jacobian: Option<Vec<f64>>,
    accepted_jacobian: Option<Vec<f64>>,
}

impl FsolveEvaluator<'_> {
    fn accept_residual_shape(&mut self, shape: Vec<usize>, scalar: bool) -> BuiltinResult<()> {
        if let Some(expected) = &self.residual_shape {
            if expected != &shape || self.residual_scalar != scalar {
                return Err(fsolve_error_with_detail(
                    &FSOLVE_ERROR_INVALID_INPUT,
                    format!(
                        "function output shape {:?} changed from initial shape {:?}",
                        shape, expected
                    ),
                ));
            }
        } else {
            self.residual_shape = Some(shape);
            self.residual_scalar = scalar;
        }
        Ok(())
    }
}

impl LeastSquaresEvaluator for FsolveEvaluator<'_> {
    fn accept_residual(&mut self) {
        self.accepted_jacobian = self.jacobian.take();
    }
    fn take_analytic_jacobian(&mut self) -> Option<Vec<f64>> {
        self.accepted_jacobian.take()
    }

    fn analytic_jacobian(&self) -> Option<&[f64]> {
        self.accepted_jacobian.as_deref()
    }

    fn cooperation(
        &mut self,
    ) -> Option<&mut crate::builtins::math::optim::common::ScalarCooperation> {
        Some(&mut *self.cooperation)
    }
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(async move {
            if self.specify_objective_gradient {
                return self.residual_with_jacobian(x).await;
            }
            let arg = if self.scalar {
                Value::Num(x[0])
            } else {
                Value::Tensor(
                    runmat_value::Tensor::new(x.to_vec(), self.shape.clone())
                        .map_err(|e| fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_INPUT, e))?,
                )
            };
            let value =
                call_function_in_context(self.cooperation.runtime(), self.function, [arg], 1)
                    .await?;
            let cooperation = &mut *self.cooperation;
            let (shape, scalar, residual) = value
                .with_owned_async(|value| async move {
                    let value = crate::builtins::math::optim::common::prepare_floating_value_with_cooperation(
                        NAME,
                        value,
                        &FSOLVE_CALLBACK_NUMERIC_EXTENSION,
                        &FSOLVE_RESIDENT_EXTENSION,
                        "function residual",
                        cooperation,
                    )
                    .await?;
                    let (shape, scalar) = match &value {
                        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (vec![1, 1], true),
                        Value::Tensor(tensor) => (tensor.shape().to_vec(), false),
                        Value::LogicalArray(array) => (array.shape.to_vec(), false),
                        _ => (vec![1, 1], false),
                    };
                    let residual = value_to_real_vector_with_cooperation(NAME, value, cooperation).await?;
                    Ok::<_, crate::RuntimeError>((shape, scalar, residual))
                })
                .await?;
            if residual.is_empty() {
                return Err(fsolve_error_with_detail(
                    &FSOLVE_ERROR_INVALID_INPUT,
                    "function value must not be empty",
                ));
            }
            self.accept_residual_shape(shape, scalar)?;
            Ok(residual)
        })
    }
}

async fn finalize(
    outcome: FsolveOutcome,
    x_shape: &[usize],
    x_scalar: bool,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Value> {
    let result = outcome.result;
    let requested = match crate::output_count::current_output_count() {
        None => return vector_to_value(NAME, result.x, x_shape, x_scalar),
        Some(0) => return Ok(Value::output_list(Vec::new())),
        Some(count @ 1..=5) => count,
        Some(_) => {
            return Err(fsolve_error_with_detail(
                &FSOLVE_ERROR_INVALID_ARGUMENT,
                "too many output arguments; maximum is 5",
            ))
        }
    };
    let mut outputs = Vec::with_capacity(requested);
    outputs.push(vector_to_value(NAME, result.x.clone(), x_shape, x_scalar)?);
    if requested >= 2 {
        outputs.push(vector_to_value(
            NAME,
            result.residual.clone(),
            &outcome.residual_shape,
            outcome.residual_scalar,
        )?);
    }
    if requested >= 3 {
        outputs.push(Value::Num(result.exitflag as f64));
    }
    if requested >= 4 {
        outputs.push(information(&result));
    }
    if requested >= 5 {
        outputs.push(
            crate::builtins::math::optim::least_squares::jacobian_value(NAME, &result, cooperation)
                .await?,
        );
    }
    Ok(crate::output_count::output_list_with_padding(
        requested, outputs,
    ))
}

fn information(result: &LeastSquaresResult) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("iterations", Value::Num(result.iterations as f64));
    fields.insert("funcCount", Value::Num(result.func_count as f64));
    fields.insert("algorithm", Value::from("levenberg-marquardt"));
    fields.insert("firstorderopt", Value::Num(result.first_order_optimality));
    fields.insert("stepsize", Value::Num(result.step_size));
    fields.insert("message", Value::from(result.message.clone()));
    Value::Struct(fields.finish())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "fsolve/jacobian_tests.rs"]
mod jacobian_tests;
