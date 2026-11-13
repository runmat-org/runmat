//! Derivative precision for the existing invocation-owned objective evaluator.
use super::*;
mod worker;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::math::optim::fminunc) enum DerivativeMode {
    Analytic,
    Forward,
    Central,
}
impl DerivativeMode {
    pub(in crate::builtins::math::optim::fminunc) fn state_cost(
        self,
        variables: usize,
    ) -> Option<usize> {
        match self {
            Self::Analytic => Some(1),
            Self::Forward => variables.checked_add(1),
            Self::Central => variables.checked_mul(2)?.checked_add(1),
        }
    }
}

pub(in crate::builtins::math::optim::fminunc) enum Refinement {
    Updated,
    BudgetLimit,
    Unavailable,
}

impl ObjectiveEvaluator<'_> {
    pub(in crate::builtins::math::optim::fminunc) async fn difference_gradient(
        &mut self,
        x: &[f64],
        f0: f64,
    ) -> BuiltinResult<Vec<f64>> {
        if self.parallel.is_some() {
            return self
                .worker_gradient(x, f0, self.derivatives == DerivativeMode::Central)
                .await;
        }
        match self.derivatives {
            DerivativeMode::Forward => {
                let mut grad = vectors::zeroed_like(x, self.cooperation).await?;
                for i in 0..x.len() {
                    let point = self.scaling.unbounded(NAME, x[i], i, false)?;
                    let step = point.first - x[i];
                    let mut trial = vectors::copied(x, self.cooperation).await?;
                    trial[i] = point.first;
                    let f_step = self.evaluate_value(&trial).await?;
                    grad[i] = (f_step - f0) / step;
                }
                Ok(grad)
            }
            DerivativeMode::Central => self.central_gradient(x).await,
            DerivativeMode::Analytic => unreachable!("analytic derivative owner"),
        }
    }

    // A one-way precision change after numerical line-search failure only.
    // This never catches objective/cancellation errors or retries their calls.
    // The current accepted objective value remains owned and is not reevaluated.
    pub(in crate::builtins::math::optim::fminunc) async fn refine_gradient(
        &mut self,
        state: &mut ObjectiveState,
    ) -> BuiltinResult<Refinement> {
        if self.derivatives != DerivativeMode::Forward {
            return Ok(Refinement::Unavailable);
        }
        if state
            .x
            .len()
            .checked_mul(2)
            .is_none_or(|cost| cost > self.remaining_func_evals())
        {
            return Ok(Refinement::BudgetLimit);
        }
        for (index, &value) in state.x.iter().enumerate() {
            if self.scaling.unbounded(NAME, value, index, true).is_err() {
                return Ok(Refinement::Unavailable);
            }
            self.cooperation.completed(1).await?;
        }
        let gradient = self.central_gradient(&state.x).await?;
        state.grad = gradient;
        self.derivatives = DerivativeMode::Central;
        Ok(Refinement::Updated)
    }

    async fn central_gradient(&mut self, x: &[f64]) -> BuiltinResult<Vec<f64>> {
        if self.parallel.is_some() {
            // Refinement has reserved 2n callbacks but changes mode only after
            // the completed gradient replaces the accepted state's gradient.
            return self.worker_gradient(x, 0.0, true).await;
        }
        let mut grad = vectors::zeroed_like(x, self.cooperation).await?;
        let mut trial = vectors::copied(x, self.cooperation).await?;
        for (index, &value) in x.iter().enumerate() {
            let points = self.scaling.unbounded(NAME, value, index, true)?;
            let plus = points.first;
            let minus = points.second.expect("central pair");
            trial[index] = plus;
            let f_plus = self.evaluate_value(&trial).await?;
            trial[index] = minus;
            let f_minus = self.evaluate_value(&trial).await?;
            trial[index] = value;
            let delta = f_plus - f_minus;
            let denominator = plus - minus;
            grad[index] = if delta.is_infinite() && f_plus.is_finite() && f_minus.is_finite() {
                f_plus / denominator - f_minus / denominator
            } else {
                delta / denominator
            };
            if !grad[index].is_finite() {
                return Err(fminunc_error_with_detail(
                    &FMINUNC_ERROR_INVALID_INPUT,
                    "central finite-difference gradient is not finite",
                ));
            }
        }
        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[test]
    fn refinement_reserves_the_actual_difference_calls_and_never_reevaluates_f0() {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 974);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let function =
            crate::user_functions::test_support::exact_local_value("derivative_fixture", binding);
        let calls = Arc::new(AtomicUsize::new(0));
        let base_calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let counted_base = base_calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                assert_eq!(id, 974);
                assert_eq!(requested, 1);
                counted.fetch_add(1, Ordering::SeqCst);
                let Value::Tensor(x) = &args[0] else {
                    panic!("vector")
                };
                let x = x.materialize_f64();
                if x == [1.0, 1.0] {
                    counted_base.fetch_add(1, Ordering::SeqCst);
                }
                let f = 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(f))
                })
            },
        )))
        .unwrap();
        for budget in [6, 7] {
            calls.store(0, Ordering::SeqCst);
            base_calls.store(0, Ordering::SeqCst);
            let mut cooperation = crate::builtins::math::optim::common::ScalarCooperation::new(
                runmat_builtins::OptimSolver::Fminunc,
                Some(&runtime),
            );
            let mut evaluator =
                ObjectiveEvaluator::new(&function, &[2, 1], false, false, budget, &mut cooperation);
            let mut state = block_on(evaluator.evaluate(&[1.0, 1.0])).unwrap();
            assert!(
                block_on(super::super::vectors::norm_inf(
                    &state.grad,
                    evaluator.cooperation,
                ))
                .unwrap()
                    > DEFAULT_TOL_FUN,
                "forward bias is not a stationary gradient"
            );
            let before = state.grad.clone();
            let refinement = block_on(evaluator.refine_gradient(&mut state)).unwrap();
            if budget == 6 {
                assert!(matches!(refinement, Refinement::BudgetLimit));
                assert_eq!(evaluator.derivatives, DerivativeMode::Forward);
                assert_eq!(state.grad, before);
                assert_eq!(calls.load(Ordering::SeqCst), 3);
            } else {
                assert!(matches!(refinement, Refinement::Updated));
                assert_eq!(evaluator.derivatives, DerivativeMode::Central);
                assert_eq!(evaluator.state_evaluation_cost(2), Some(5));
                assert!(
                    block_on(super::super::vectors::norm_inf(
                        &state.grad,
                        evaluator.cooperation,
                    ))
                    .unwrap()
                        < DEFAULT_TOL_FUN
                );
                assert_eq!(calls.load(Ordering::SeqCst), 7);
            }
            assert_eq!(state.x, [1.0, 1.0]);
            assert_eq!(state.f, 0.0);
            assert_eq!(
                base_calls.load(Ordering::SeqCst),
                1,
                "original f0 stays owned"
            );
            assert_eq!(evaluator.func_count, calls.load(Ordering::SeqCst));
        }
    }
}
