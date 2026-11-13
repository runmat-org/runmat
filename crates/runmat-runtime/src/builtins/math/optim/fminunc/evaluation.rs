//! One original objective/gradient callback and evaluation-budget owner.
use super::algorithm::finite_difference::DerivativeMode;
use super::*;

#[derive(Debug, Clone)]
pub(super) struct ObjectiveState {
    pub(super) x: Vec<f64>,
    pub(super) f: f64,
    pub(super) grad: Vec<f64>,
}

mod internal;
pub(super) enum ObjectiveTarget<'a> {
    Callable(&'a Value),
    Internal(&'a mut dyn crate::builtins::math::optim::fmincon::compiled::Oracle),
}
pub(super) struct ObjectiveEvaluator<'a> {
    pub(super) cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    target: ObjectiveTarget<'a>,
    pub(super) shape: Vec<usize>,
    pub(super) scalar: bool,
    pub(super) derivatives: DerivativeMode,
    pub(super) max_fun_evals: usize,
    pub(super) func_count: usize,
    pub(super) parallel: Option<crate::execution::CallbackBatch>,
    pub(super) scaling: crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling,
}

impl<'a> ObjectiveEvaluator<'a> {
    pub(super) fn new(
        function: &'a Value,
        shape: &[usize],
        scalar: bool,
        specify_gradient: bool,
        max_fun_evals: usize,
        cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    ) -> Self {
        Self::with_target(
            ObjectiveTarget::Callable(function),
            shape,
            scalar,
            specify_gradient,
            max_fun_evals,
            cooperation,
        )
    }

    fn with_target(
        target: ObjectiveTarget<'a>,
        shape: &[usize],
        scalar: bool,
        specify_gradient: bool,
        max_fun_evals: usize,
        cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    ) -> Self {
        Self {
            cooperation,
            target,
            shape: shape.to_vec(),
            scalar,
            derivatives: if specify_gradient {
                DerivativeMode::Analytic
            } else {
                DerivativeMode::Forward
            },
            max_fun_evals,
            func_count: 0,
            parallel: None,
            scaling: Default::default(),
        }
    }

    pub(super) async fn evaluate(&mut self, x: &[f64]) -> BuiltinResult<ObjectiveState> {
        self.ensure_state_budget(x.len())?;
        if self.derivatives == DerivativeMode::Analytic {
            self.evaluate_with_gradient(x).await
        } else {
            let f = self.evaluate_value(x).await?;
            let grad = self.difference_gradient(x, f).await?;
            Ok(ObjectiveState {
                x: super::algorithm::vectors::copied(x, self.cooperation).await?,
                f,
                grad,
            })
        }
    }

    pub(super) async fn evaluate_value(&mut self, x: &[f64]) -> BuiltinResult<f64> {
        self.cooperation.completed(x.len() as u64).await?;
        self.ensure_value_budget()?;
        if matches!(&self.target, ObjectiveTarget::Internal(_)) {
            return self.internal_value(x).await;
        }
        let ObjectiveTarget::Callable(function) = &self.target else {
            unreachable!("selected scalar target");
        };
        let arg = vector_to_value(
            NAME,
            super::algorithm::vectors::copied(x, self.cooperation).await?,
            &self.shape,
            self.scalar,
        )?;
        let value =
            call_function_in_context(self.cooperation.runtime(), function, [arg], 1).await?;
        self.func_count += 1;
        self.scalar_result(value).await
    }

    pub(in crate::builtins::math::optim::fminunc) async fn scalar_result(
        &mut self,
        value: crate::execution::RootedValue,
    ) -> BuiltinResult<f64> {
        let cooperation = &mut *self.cooperation;
        value
            .with_owned_async(|value| {
                crate::builtins::math::optim::common::value_to_scalar_with_extensions(
                    NAME,
                    value,
                    &FMINUNC_CALLBACK_NUMERIC_EXTENSION,
                    &FMINUNC_RESIDENT_EXTENSION,
                    "objective value",
                    cooperation,
                )
            })
            .await
    }

    pub(super) async fn evaluate_with_gradient(
        &mut self,
        x: &[f64],
    ) -> BuiltinResult<ObjectiveState> {
        self.evaluate_analytic(x, false)
            .await
            .map(|(state, _)| state)
    }

    pub(super) async fn evaluate_analytic(
        &mut self,
        x: &[f64],
        third: bool,
    ) -> BuiltinResult<(ObjectiveState, Option<crate::execution::RootedValue>)> {
        self.ensure_value_budget()?;
        self.cooperation.completed(x.len() as u64).await?;
        let requested = if third { 3 } else { 2 };
        let ObjectiveTarget::Callable(function) = &self.target else {
            return Err(internal::invalid(
                "modeled analytic objective derivatives are unavailable",
            ));
        };
        let arg = vector_to_value(
            NAME,
            super::algorithm::vectors::copied(x, self.cooperation).await?,
            &self.shape,
            self.scalar,
        )?;
        let value =
            call_function_in_context(self.cooperation.runtime(), function, [arg], requested)
                .await?;
        self.func_count += 1;
        let cooperation = &mut *self.cooperation;
        let (f, grad, model) = value
            .with_owned_async(|value| async move {
                let Value::OutputList(outputs) = value else {
                    return Err(fminunc_error_with_detail(
                        &FMINUNC_ERROR_INVALID_INPUT,
                        "objective must return [f, g] when SpecifyObjectiveGradient is true",
                    ));
                };
                if outputs.len() < requested {
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
                    cooperation,
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
                let model = if third {
                    Some(crate::execution::RootedValue::in_optional_context(
                        outputs.next().expect("validated model output"),
                        cooperation.runtime(),
                    )?)
                } else {
                    None
                };
                Ok::<_, crate::RuntimeError>((f, grad, model))
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
        Ok((
            ObjectiveState {
                x: super::algorithm::vectors::copied(x, self.cooperation).await?,
                f,
                grad,
            },
            model,
        ))
    }

    pub(super) fn state_evaluation_cost(&self, variables: usize) -> Option<usize> {
        self.derivatives.state_cost(variables)
    }

    pub(super) fn remaining_func_evals(&self) -> usize {
        self.max_fun_evals.saturating_sub(self.func_count)
    }

    pub(super) fn can_evaluate_state(&self, variables: usize) -> bool {
        self.state_evaluation_cost(variables)
            .is_some_and(|cost| cost <= self.remaining_func_evals())
    }

    pub(super) fn can_evaluate_value(&self) -> bool {
        self.remaining_func_evals() >= 1
    }

    pub(super) fn ensure_state_budget(&self, variables: usize) -> BuiltinResult<()> {
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

    pub(super) fn ensure_value_budget(&self) -> BuiltinResult<()> {
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
