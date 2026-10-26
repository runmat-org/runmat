//! Borrowed original model scalar probes share the existing fminunc counter.
use super::*;
use crate::builtins::math::optim::fmincon::compiled::{Oracle, Probe, ProbeRole};
pub(super) fn invalid(message: &str) -> RuntimeError {
    let mut error = fminunc_error_with_detail(&FMINUNC_ERROR_INVALID_INPUT, message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
impl<'a> ObjectiveEvaluator<'a> {
    pub(in crate::builtins::math::optim::fminunc) fn internal(
        oracle: &'a mut dyn Oracle,
        variables: usize,
        max_fun_evals: usize,
        cooperation: &'a mut crate::builtins::math::optim::common::ScalarCooperation,
    ) -> Self {
        Self::with_target(
            ObjectiveTarget::Internal(oracle),
            &[variables, 1],
            false,
            false,
            max_fun_evals,
            cooperation,
        )
    }
    pub(super) async fn internal_value(&mut self, x: &[f64]) -> BuiltinResult<f64> {
        let ObjectiveTarget::Internal(oracle) = &mut self.target else {
            return Err(invalid("internal objective target is unavailable"));
        };
        let returned = oracle
            .evaluate(x, ProbeRole::Full, self.cooperation)
            .await?;
        self.func_count += 1;
        match returned {
            Ok(Probe::Full {
                objective,
                inequality,
                equality,
            }) => {
                if !inequality.is_empty() || !equality.is_empty() {
                    return Err(invalid(
                        "unconstrained objective cannot discard constraint rows",
                    ));
                }
                if !objective.is_finite() {
                    return Err(invalid("original modeled objective is not finite"));
                }
                Ok(objective)
            }
            Ok(Probe::ConstraintsOnly { .. }) => Err(invalid(
                "unconstrained objective requires its full scalar probe",
            )),
            Err(_) => Err(invalid(
                "original modeled objective did not return a defined scalar value",
            )),
        }
    }
}
