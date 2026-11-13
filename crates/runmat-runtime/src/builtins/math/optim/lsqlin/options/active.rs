//! Actually active working-set controls; no callback or device authority is invented.
use super::*;
pub(in crate::builtins::math::optim::lsqlin) struct Active {
    pub step: f64,
    pub objective_limit: f64,
}
impl Active {
    pub(super) fn parse(
        fields: &crate::builtins::math::optim::common::ScalarOptionFields<'_>,
    ) -> BuiltinResult<Self> {
        let step = fields.real(NAME, Field::TolX, 1e-8)?;
        let objective_limit = fields
            .exact(Field::ObjectiveLimit)
            .map(|value| {
                crate::builtins::math::optim::common::quasi_newton_controls::objective_limit(
                    NAME, value,
                )
            })
            .transpose()?
            .unwrap_or(-1e20);
        if !(step > 0. && step.is_finite()) {
            return Err(invalid("active-set StepTolerance must be positive finite"));
        }
        Ok(Self {
            step,
            objective_limit,
        })
    }
}
