//! Numerical result selection under the original callback root lease.
use super::*;
use numeric::{Geometry, Numeric};
impl Evaluator<'_> {
    pub(super) async fn objective_result(
        &mut self,
        returned: crate::execution::RootedValueSequence,
    ) -> BuiltinResult<Option<f64>> {
        let returned = select(returned, 1, self.cooperation.runtime())?;
        let returned = {
            let _entry = self
                .cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            returned.try_into_single().map_err(consumer_refusal)?
        };
        let cooperation = &mut *self.cooperation;
        let policy = self.context.policy;
        let value = returned
            .with_owned_async(|value| async move {
                numeric::read(value, "objective", Geometry::Scalar, &policy, cooperation).await
            })
            .await?;
        let Numeric::Real { values, .. } = value else {
            return Ok(None);
        };
        Ok(Some(values[0]))
    }
    pub(super) async fn constraint_result(
        &mut self,
        returned: crate::execution::RootedValueSequence,
    ) -> BuiltinResult<Option<(Vec<f64>, Vec<f64>)>> {
        let returned = select(returned, 2, self.cooperation.runtime())?;
        let mut extents = [0; 2];
        // Admit every required output before the first provider transfer. An
        // opaque second output must not cause inspection of its descendants.
        for (index, value) in returned.as_slice().iter().enumerate() {
            if !numeric::preflight(
                value,
                "nonlinear constraint",
                Geometry::Vector,
                &self.context.policy,
                self.cooperation.runtime(),
            )? {
                return Ok(None);
            }
            extents[index] = numeric::shape(value)
                .expect("preflight numeric carrier")
                .iter()
                .try_fold(1usize, |size, &axis| size.checked_mul(axis))
                .ok_or_else(|| invalid("nonlinear constraint extent overflow"))?;
        }
        if self
            .state
            .extents
            .is_some_and(|known| known != (extents[0], extents[1]))
        {
            return Err(invalid("nonlinear constraint output extent changed"));
        }
        let mut returned = {
            let _entry = self
                .cooperation
                .runtime()
                .map(crate::context::RuntimeContext::enter);
            returned.try_into_values().map_err(consumer_refusal)?
        };
        // The iterator retains the original result root forest while BOTH raw
        // carriers are numerically decoded, including across provider awaits.
        let Numeric::Real {
            values: inequality, ..
        } = numeric::read(
            returned.next().expect("selected inequality"),
            "nonlinear inequality",
            Geometry::Vector,
            &self.context.policy,
            self.cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        let Numeric::Real {
            values: equality, ..
        } = numeric::read(
            returned.next().expect("selected equality"),
            "nonlinear equality",
            Geometry::Vector,
            &self.context.policy,
            self.cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        drop(returned);
        self.state.extents = Some((extents[0], extents[1]));
        Ok(Some((inequality, equality)))
    }
}
