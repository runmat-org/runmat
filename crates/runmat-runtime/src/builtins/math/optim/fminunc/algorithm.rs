//! Existing BFGS, line-search, and Hessian algorithms with invocation cooperation.

use super::*;

#[derive(Debug, Clone)]
struct ObjectiveState {
    x: Vec<f64>,
    f: f64,
    grad: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(super) struct MinimizeOutcome {
    x: Vec<f64>,
    fval: f64,
    grad: Vec<f64>,
    inverse_hessian: Vec<f64>,
    iterations: usize,
    func_count: usize,
    exitflag: i32,
    firstorderopt: f64,
    stepsize: f64,
    message: String,
}

struct ObjectiveEvaluator<'a> {
    cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    function: &'a Value,
    shape: Vec<usize>,
    scalar: bool,
    specify_gradient: bool,
    max_fun_evals: usize,
    func_count: usize,
}

impl<'a> ObjectiveEvaluator<'a> {
    fn new(
        function: &'a Value,
        shape: &[usize],
        scalar: bool,
        specify_gradient: bool,
        max_fun_evals: usize,
        cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    ) -> Self {
        Self {
            cooperation,
            function,
            shape: shape.to_vec(),
            scalar,
            specify_gradient,
            max_fun_evals,
            func_count: 0,
        }
    }

    async fn evaluate(&mut self, x: &[f64]) -> BuiltinResult<ObjectiveState> {
        self.ensure_state_budget(x.len())?;
        if self.specify_gradient {
            self.evaluate_with_gradient(x).await
        } else {
            let f = self.evaluate_value(x).await?;
            let grad = self.forward_difference_gradient(x, f).await?;
            Ok(ObjectiveState {
                x: x.to_vec(),
                f,
                grad,
            })
        }
    }

    async fn evaluate_value(&mut self, x: &[f64]) -> BuiltinResult<f64> {
        self.cooperation.completed(x.len() as u64).await?;
        self.ensure_value_budget()?;
        let arg = vector_to_value(NAME, x.to_vec(), &self.shape, self.scalar)?;
        let value = call_function(self.function, vec![arg]).await?;
        self.func_count += 1;
        value
            .with_owned_async(|value| {
                crate::builtins::math::optim::common::value_to_scalar_with_extensions(
                    NAME,
                    value,
                    &FMINUNC_CALLBACK_NUMERIC_EXTENSION,
                    &FMINUNC_RESIDENT_EXTENSION,
                    "objective value",
                )
            })
            .await
    }

    async fn evaluate_with_gradient(&mut self, x: &[f64]) -> BuiltinResult<ObjectiveState> {
        self.cooperation.completed(x.len() as u64).await?;
        let callback =
            crate::canonicalize_callback_handle_for_semantic_resolution(self.function.clone());
        let arg = vector_to_value(NAME, x.to_vec(), &self.shape, self.scalar)?;
        let value = crate::call_feval_async_with_outputs(callback, &[arg], 2).await?;
        self.func_count += 1;
        let cooperation = &mut *self.cooperation;
        let (f, grad) = value
            .with_owned_async(|value| async move {
                let Value::OutputList(outputs) = value else {
                    return Err(fminunc_error_with_detail(
                        &FMINUNC_ERROR_INVALID_INPUT,
                        "objective must return [f, g] when SpecifyObjectiveGradient is true",
                    ));
                };
                if outputs.len() < 2 {
                    return Err(fminunc_error_with_detail(
                        &FMINUNC_ERROR_INVALID_INPUT,
                        "objective must return both objective value and gradient",
                    ));
                }
                let outputs = outputs.try_into_owned().map_err(|_| {
                    crate::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?;
                let mut outputs = outputs.into_iter();
                let objective = outputs.next().expect("validated objective output");
                let gradient = outputs.next().expect("validated gradient output");
                let f = crate::builtins::math::optim::common::value_to_scalar_with_extensions(
                    NAME,
                    objective,
                    &FMINUNC_CALLBACK_NUMERIC_EXTENSION,
                    &FMINUNC_RESIDENT_EXTENSION,
                    "objective value",
                )
                .await?;
                let grad =
                    crate::builtins::math::optim::common::value_to_real_vector_with_extensions(
                        NAME,
                        gradient,
                        &FMINUNC_CALLBACK_NUMERIC_EXTENSION,
                        &FMINUNC_RESIDENT_EXTENSION,
                        "objective gradient",
                        cooperation,
                    )
                    .await?;
                Ok::<_, crate::RuntimeError>((f, grad))
            })
            .await?;
        if grad.len() != x.len() {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_INPUT,
                format!(
                    "objective gradient length {} does not match x length {}",
                    grad.len(),
                    x.len()
                ),
            ));
        }
        Ok(ObjectiveState {
            x: x.to_vec(),
            f,
            grad,
        })
    }

    async fn forward_difference_gradient(&mut self, x: &[f64], f0: f64) -> BuiltinResult<Vec<f64>> {
        let mut grad = vec![0.0; x.len()];
        for i in 0..x.len() {
            let step = finite_difference_step(x[i]);
            let mut trial = x.to_vec();
            trial[i] += step;
            let f_step = self.evaluate_value(&trial).await?;
            grad[i] = (f_step - f0) / step;
        }
        Ok(grad)
    }

    fn state_evaluation_cost(&self, variables: usize) -> Option<usize> {
        if self.specify_gradient {
            Some(1)
        } else {
            variables.checked_add(1)
        }
    }

    fn remaining_func_evals(&self) -> usize {
        self.max_fun_evals.saturating_sub(self.func_count)
    }

    fn can_evaluate_state(&self, variables: usize) -> bool {
        self.state_evaluation_cost(variables)
            .is_some_and(|cost| cost <= self.remaining_func_evals())
    }

    fn can_evaluate_value(&self) -> bool {
        self.remaining_func_evals() >= 1
    }

    fn ensure_state_budget(&self, variables: usize) -> BuiltinResult<()> {
        if self.can_evaluate_state(variables) {
            return Ok(());
        }
        Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            format!(
                "MaxFunEvals exceeded before completing objective/gradient evaluation (FuncCount {}, MaxFunEvals {})",
                self.func_count, self.max_fun_evals
            ),
        ))
    }

    fn ensure_value_budget(&self) -> BuiltinResult<()> {
        if self.can_evaluate_value() {
            return Ok(());
        }
        Err(fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            format!(
                "MaxFunEvals exceeded before objective evaluation (FuncCount {}, MaxFunEvals {})",
                self.func_count, self.max_fun_evals
            ),
        ))
    }
}

fn finite_difference_step(x: f64) -> f64 {
    f64::EPSILON.sqrt() * x.abs().max(1.0)
}

pub(super) async fn minimize(
    function: &Value,
    x0: Vec<f64>,
    shape: &[usize],
    scalar: bool,
    options: &FminuncOptions,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<MinimizeOutcome> {
    let n = x0.len();
    validate_problem_size(n)?;
    let mut evaluator = ObjectiveEvaluator::new(
        function,
        shape,
        scalar,
        options.specify_objective_gradient,
        options.max_fun_evals,
        cooperation,
    );
    let mut state = evaluator.evaluate(&x0).await?;
    let mut inverse_hessian = identity(n)?;
    let mut iterations = 0usize;
    let mut stepsize = 0.0f64;
    let mut exitflag = 0i32;
    let mut line_search_failed = false;

    if norm_inf(&state.grad) <= options.tol_fun {
        exitflag = 1;
    }

    while exitflag == 0
        && iterations < options.max_iter
        && evaluator.func_count < options.max_fun_evals
    {
        let mut direction =
            mat_vec(&inverse_hessian, &state.grad, n, evaluator.cooperation).await?;
        for value in &mut direction {
            *value = -*value;
        }
        if dot(&direction, &state.grad) >= -1.0e-14 || !all_finite(&direction) {
            inverse_hessian = identity(n)?;
            direction = state.grad.iter().map(|value| -value).collect();
        }

        let next = match line_search(&mut evaluator, &state, &direction).await? {
            LineSearchResult::Step(next) => next,
            LineSearchResult::BudgetExhausted => {
                exitflag = 0;
                break;
            }
            LineSearchResult::Failed => {
                line_search_failed = true;
                exitflag = -3;
                break;
            }
        };

        let s = subtract(&next.x, &state.x);
        stepsize = norm_inf(&s);
        let y = subtract(&next.grad, &state.grad);
        update_inverse_hessian(&mut inverse_hessian, &s, &y, n, evaluator.cooperation).await?;
        let f_delta = (state.f - next.f).abs();
        let x_scale = 1.0 + norm_inf(&state.x);
        let f_scale = 1.0 + state.f.abs();
        state = next;
        iterations += 1;

        if norm_inf(&state.grad) <= options.tol_fun {
            exitflag = 1;
        } else if stepsize <= options.tol_x * x_scale {
            exitflag = 2;
        } else if f_delta <= options.tol_fun * f_scale {
            exitflag = 3;
        }
    }

    let firstorderopt = norm_inf(&state.grad);
    let message = build_message(
        exitflag,
        iterations,
        evaluator.func_count,
        line_search_failed,
    );
    emit_summary(
        &state,
        exitflag,
        &message,
        options,
        iterations,
        evaluator.func_count,
    );

    Ok(MinimizeOutcome {
        x: state.x,
        fval: state.f,
        grad: state.grad,
        inverse_hessian,
        iterations,
        func_count: evaluator.func_count,
        exitflag,
        firstorderopt,
        stepsize,
        message,
    })
}

async fn line_search(
    evaluator: &mut ObjectiveEvaluator<'_>,
    state: &ObjectiveState,
    direction: &[f64],
) -> BuiltinResult<LineSearchResult> {
    let dphi0 = dot(&state.grad, direction);
    if dphi0 >= 0.0 || !dphi0.is_finite() {
        return Ok(LineSearchResult::Failed);
    }

    let mut alpha_prev = 0.0;
    let mut f_prev = state.f;
    let mut alpha = 1.0;
    for iter in 0..MAX_LINE_SEARCH_ITERS {
        if !evaluator.can_evaluate_state(state.x.len()) {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let trial_x = add_scaled(&state.x, direction, alpha);
        let trial = evaluator.evaluate(&trial_x).await?;
        if trial.f > state.f + C1 * alpha * dphi0 || (iter > 0 && trial.f >= f_prev) {
            return zoom(evaluator, state, direction, alpha_prev, alpha).await;
        }
        let dphi = dot(&trial.grad, direction);
        if dphi.abs() <= -C2 * dphi0 {
            return Ok(LineSearchResult::Step(trial));
        }
        if dphi >= 0.0 {
            return zoom(evaluator, state, direction, alpha, alpha_prev).await;
        }
        alpha_prev = alpha;
        f_prev = trial.f;
        alpha = (alpha * 2.0).min(64.0);
    }
    Ok(LineSearchResult::Failed)
}

enum LineSearchResult {
    Step(ObjectiveState),
    BudgetExhausted,
    Failed,
}

async fn zoom(
    evaluator: &mut ObjectiveEvaluator<'_>,
    state: &ObjectiveState,
    direction: &[f64],
    mut lo: f64,
    mut hi: f64,
) -> BuiltinResult<LineSearchResult> {
    let dphi0 = dot(&state.grad, direction);
    let mut f_lo = if lo == 0.0 {
        state.f
    } else {
        if !evaluator.can_evaluate_value() {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let x_lo = add_scaled(&state.x, direction, lo);
        evaluator.evaluate_value(&x_lo).await?
    };
    for _ in 0..MAX_LINE_SEARCH_ITERS {
        if !evaluator.can_evaluate_state(state.x.len()) {
            return Ok(LineSearchResult::BudgetExhausted);
        }
        let alpha = 0.5 * (lo + hi);
        let trial_x = add_scaled(&state.x, direction, alpha);
        let trial = evaluator.evaluate(&trial_x).await?;
        if trial.f > state.f + C1 * alpha * dphi0 || trial.f >= f_lo {
            hi = alpha;
        } else {
            let dphi = dot(&trial.grad, direction);
            if dphi.abs() <= -C2 * dphi0 {
                return Ok(LineSearchResult::Step(trial));
            }
            if dphi * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
            f_lo = trial.f;
        }
        if (hi - lo).abs() <= 1.0e-12 * (1.0 + lo.abs() + hi.abs()) {
            if trial.f <= state.f + C1 * alpha * dphi0 {
                return Ok(LineSearchResult::Step(trial));
            }
            return Ok(LineSearchResult::Failed);
        }
    }
    Ok(LineSearchResult::Failed)
}

async fn update_inverse_hessian(
    h: &mut [f64],
    s: &[f64],
    y: &[f64],
    n: usize,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<()> {
    let ys = dot(y, s);
    if ys <= 1.0e-14 || !ys.is_finite() {
        return Ok(());
    }
    let rho = 1.0 / ys;
    let hy = mat_vec(h, y, n, cooperation).await?;
    let yhy = dot(y, &hy);
    let coeff = (1.0 + yhy * rho) * rho;
    for row in 0..n {
        cooperation.completed(n as u64).await?;
        for col in 0..n {
            h[row * n + col] +=
                coeff * s[row] * s[col] - rho * (s[row] * hy[col] + hy[row] * s[col]);
        }
    }
    Ok(())
}

pub(super) async fn finalize(
    outcome: MinimizeOutcome,
    shape: &[usize],
    scalar: bool,
    _options: &FminuncOptions,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Value> {
    let requested = match crate::output_count::current_output_count() {
        None => return vector_to_value(NAME, outcome.x, shape, scalar),
        Some(0) => return Ok(Value::output_list(Vec::new())),
        Some(count @ 1..=6) => count,
        Some(_) => {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "too many output arguments; maximum is 6",
            ))
        }
    };
    let mut outputs = Vec::with_capacity(requested);
    outputs.push(vector_to_value(NAME, outcome.x.clone(), shape, scalar)?);
    if requested >= 2 {
        outputs.push(Value::Num(outcome.fval));
    }
    if requested >= 3 {
        outputs.push(Value::Num(outcome.exitflag as f64));
    }
    if requested >= 4 {
        outputs.push(Value::Struct(build_output_struct(&outcome)));
    }
    if requested >= 5 {
        outputs.push(vector_to_value(NAME, outcome.grad.clone(), shape, scalar)?);
    }
    if requested >= 6 {
        outputs.push(
            hessian_value(
                &outcome.inverse_hessian,
                outcome.grad.len(),
                scalar,
                cooperation,
            )
            .await?,
        );
    }
    Ok(crate::output_count::output_list_with_padding(
        requested, outputs,
    ))
}

async fn hessian_value(
    inverse_hessian: &[f64],
    n: usize,
    scalar: bool,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Value> {
    let hessian = match invert_matrix(inverse_hessian, n, cooperation).await? {
        Some(hessian) => hessian,
        None => identity(n)?,
    };
    if scalar {
        Ok(Value::Num(hessian[0]))
    } else {
        Tensor::new(hessian, vec![n, n])
            .map(Value::Tensor)
            .map_err(|e| fminunc_error_with_detail(&FMINUNC_ERROR_INVALID_INPUT, e))
    }
}

fn build_output_struct(outcome: &MinimizeOutcome) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("funcCount", Value::Num(outcome.func_count as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("firstorderopt", Value::Num(outcome.firstorderopt));
    fields.insert("stepsize", Value::Num(outcome.stepsize));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields
}

fn build_message(
    exitflag: i32,
    iterations: usize,
    func_count: usize,
    line_search_failed: bool,
) -> String {
    match exitflag {
        1 => format!(
            "Optimization terminated: first-order optimality is below OPTIONS.TolFun. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        2 => format!(
            "Optimization terminated: step size is below OPTIONS.TolX. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        3 => format!(
            "Optimization terminated: objective change is below OPTIONS.TolFun. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        -3 if line_search_failed => format!(
            "Exiting: line search could not find a point satisfying the strong Wolfe conditions. Iterations: {iterations}, FuncCount: {func_count}."
        ),
        _ => format!(
            "Exiting: Maximum number of function evaluations or iterations has been exceeded. Iterations: {iterations}, FuncCount: {func_count}."
        ),
    }
}

fn emit_summary(
    state: &ObjectiveState,
    exitflag: i32,
    message: &str,
    options: &FminuncOptions,
    iterations: usize,
    func_count: usize,
) {
    let should_emit = match options.display {
        DisplayMode::Off | DisplayMode::None => false,
        DisplayMode::Final | DisplayMode::Iter => true,
        DisplayMode::Notify => exitflag <= 0,
    };
    if !should_emit {
        return;
    }
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!(
            "fminunc: fval = {fval:.6e}, firstorderopt = {opt:.6e}, exitflag = {exitflag}. {message}",
            fval = state.f,
            opt = norm_inf(&state.grad),
        ),
    );
    if matches!(options.display, DisplayMode::Iter) {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            format!("fminunc: iterations = {iterations}, funcCount = {func_count}"),
        );
    }
}

pub(super) fn dense_len(n: usize) -> BuiltinResult<usize> {
    n.checked_mul(n).ok_or_else(|| {
        fminunc_error_with_detail(
            &FMINUNC_ERROR_INVALID_INPUT,
            "dense BFGS matrix size overflows usize",
        )
    })
}

fn identity(n: usize) -> BuiltinResult<Vec<f64>> {
    let len = dense_len(n)?;
    let mut out = vec![0.0; len];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    Ok(out)
}

async fn invert_matrix(
    matrix: &[f64],
    n: usize,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut a = matrix.to_vec();
    let mut inv = identity(n)?;
    for col in 0..n {
        cooperation.completed(n as u64).await?;
        let mut pivot = col;
        let mut pivot_abs = a[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = a[row * n + col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if pivot_abs <= 1.0e-14 || !pivot_abs.is_finite() {
            return Ok(None);
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
                inv.swap(col * n + j, pivot * n + j);
            }
        }
        let diag = a[col * n + col];
        for j in 0..n {
            a[col * n + j] /= diag;
            inv[col * n + j] /= diag;
        }
        for row in 0..n {
            cooperation.completed(n as u64).await?;
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                a[row * n + j] -= factor * a[col * n + j];
                inv[row * n + j] -= factor * inv[col * n + j];
            }
        }
    }
    Ok(Some(inv))
}

async fn mat_vec(
    matrix: &[f64],
    vector: &[f64],
    n: usize,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut out = vec![0.0; n];
    for row in 0..n {
        cooperation.completed(n as u64).await?;
        for col in 0..n {
            out[row] += matrix[row * n + col] * vector[col];
        }
    }
    Ok(out)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_inf(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn add_scaled(x: &[f64], direction: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .zip(direction.iter())
        .map(|(xi, di)| xi + alpha * di)
        .collect()
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn one_output_does_not_invert_unused_hessian() {
        let _outputs = crate::output_count::push_output_count(Some(1));
        let outcome = MinimizeOutcome {
            x: vec![2.0],
            fval: 0.0,
            grad: vec![0.0],
            inverse_hessian: vec![],
            iterations: 0,
            func_count: 1,
            exitflag: 1,
            firstorderopt: 0.0,
            stepsize: 0.0,
            message: String::new(),
        };
        let options = FminuncOptions {
            tol_x: 1.0e-6,
            tol_fun: 1.0e-6,
            max_iter: 1,
            max_fun_evals: 1,
            display: DisplayMode::Off,
            specify_objective_gradient: false,
        };
        let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
            runmat_builtins::OptimSolver::Fminunc,
            None,
        );
        let value = block_on(finalize(outcome, &[1, 1], true, &options, &mut cooperation)).unwrap();
        assert!(
            matches!(value, Value::OutputList(values) if matches!(values.as_slice(), [Value::Num(number)] if *number == 2.0))
        );
    }
}
