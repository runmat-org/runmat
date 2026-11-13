//! Actual equality targets and active progress share the caller's original controls.
use super::*;
pub(super) struct Target<'a, V> {
    pub model: &'a Model,
    pub controls: &'a mut options::Controls,
    pub validation: &'a mut V,
    pub scale: optimality::Scale,
    pub metrics: Option<optimality::Metrics>,
    pub retained_active: Option<Vec<bool>>,
}
impl<V: compiled::Validation> Target<'_, V> {
    fn accuracy(&self) -> optimality::Accuracy {
        optimality::Accuracy {
            feasibility: self.controls.feasibility,
            optimality: self.controls.optimality,
        }
    }
}
impl<V: compiled::Validation> working_set::Target for Target<'_, V> {
    async fn initial_active(
        &mut self,
        _rows: working_set::Rows<'_>,
        _x: &[f64],
        activity: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<bool>>> {
        match self.validation.saved_working_set() {
            Some(saved) => {
                working_set::warm::initial(
                    working_set::warm::View {
                        rows: self.model.view(),
                        roles: &self.model.rows.roles,
                    },
                    saved,
                    activity,
                    self.controls.feasibility,
                    super::warm_policy(),
                    c,
                )
                .await
            }
            None => Ok(None),
        }
    }
    fn retain_active(&mut self, active: Vec<bool>) {
        if self.validation.wants_working_set() {
            self.retained_active = Some(active);
        }
    }
    fn begin(&mut self) -> BuiltinResult<()> {
        self.controls.advance()
    }
    async fn equality(
        &mut self,
        a: &Matrix,
        b: &[f64],
        _x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<working_set::EqualityTarget>> {
        let Some(step) = saddle::solve(self.model, a, b, c).await? else {
            return Ok(None);
        };
        let rows = working_set::Rows {
            a,
            b,
            equalities: b.len(),
        };
        if self
            .scale
            .checked(
                self.model,
                rows,
                &step.direction,
                &step.multipliers,
                self.accuracy(),
                c,
            )
            .await?
            .is_some()
        {
            Ok(Some(working_set::EqualityTarget::Point(step)))
        } else {
            Ok(None)
        }
    }
    async fn checked(
        &mut self,
        rows: working_set::Rows<'_>,
        x: &[f64],
        z: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.metrics = self
            .scale
            .checked(self.model, rows, x, z, self.accuracy(), c)
            .await?;
        Ok(self.metrics.is_some())
    }
    async fn advanced(
        &mut self,
        x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<working_set::Stop>> {
        let Some(point) = self.model.evaluate(x, c).await? else {
            return Ok(Some(working_set::Stop::Unverified));
        };
        self.controls.record_active(point.value, c.runtime())?;
        let limit = self
            .controls
            .active
            .as_ref()
            .expect("selected active controls")
            .objective_limit;
        let cost = (0.5 * self.model.units) * point.value;
        if !cost.is_finite() {
            return Ok(Some(working_set::Stop::Unverified));
        }
        Ok(
            match self
                .validation
                .objective_limit(x, cost, limit, self.controls.feasibility, c)
                .await?
            {
                Some(true) => Some(working_set::Stop::ObjectiveLimit),
                Some(false) => None,
                None => Some(working_set::Stop::Unverified),
            },
        )
    }
    fn small_step(&self, change: working_set::Change) -> bool {
        let step = self
            .controls
            .active
            .as_ref()
            .expect("selected active controls")
            .step;
        change.length.is_finite()
            && change.reference.is_finite()
            && change.length / change.reference <= step
    }
}
