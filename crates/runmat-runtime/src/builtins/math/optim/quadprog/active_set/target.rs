//! Numerical targets and original checked terminal KKT share one invocation.
use super::*;
pub(super) struct Target<'a, V> {
    pub model: &'a Model,
    pub controls: &'a mut options::Controls,
    pub validation: &'a mut V,
    pub checked: Option<validation::Checked>,
    pub active: Option<Vec<bool>>,
}
impl<V: compiled::Validation> working_set::Target for Target<'_, V> {
    fn begin(&mut self) -> BuiltinResult<()> {
        self.controls.advance()
    }
    fn retain_active(&mut self, active: Vec<bool>) {
        if self.validation.wants_working_set() {
            self.active = Some(active);
        }
    }
    async fn initial_active(
        &mut self,
        _: working_set::Rows<'_>,
        _: &[f64],
        activity: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<bool>>> {
        match self.validation.saved_working_set() {
            Some(saved) => {
                working_set::warm::initial(
                    working_set::warm::View {
                        rows: view(self.model),
                        roles: &self.model.roles,
                    },
                    saved,
                    activity,
                    self.controls.feasibility,
                    curvature::POLICY,
                    c,
                )
                .await
            }
            None => Ok(None),
        }
    }
    async fn equality(
        &mut self,
        a: &Matrix,
        b: &[f64],
        x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<working_set::EqualityTarget>> {
        let Some(proposal) = saddle::solve(self.model, a, b, c).await? else {
            return Ok(None);
        };
        let geometry = geometry::Equality {
            model: self.model,
            rows: a,
            rhs: b,
        };
        if geometry
            .point(
                &proposal.point,
                self.controls.optimality.min(self.controls.feasibility),
                c,
            )
            .await?
        {
            return Ok(Some(working_set::EqualityTarget::Point(proposal.point)));
        }
        if geometry.recession(x, &proposal.residual, c).await? {
            return Ok(Some(working_set::EqualityTarget::Recession(
                proposal.residual,
            )));
        }
        Ok(None)
    }
    async fn checked(
        &mut self,
        _: working_set::Rows<'_>,
        x: &[f64],
        z: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.checked = validation::optimum(self.model, x, z, self.controls, c).await?;
        Ok(self.checked.as_ref().is_some_and(|checked| checked.valid))
    }
    async fn advanced(
        &mut self,
        x: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<working_set::Stop>> {
        if self.controls.time_reached() {
            return Ok(Some(working_set::Stop::TimeLimit));
        }
        let Some(point) = validation::primal(self.model, x, c).await? else {
            return Ok(Some(working_set::Stop::Unverified));
        };
        self.controls.record_active(point.objective, c.runtime())?;
        if point.violation > self.controls.feasibility {
            return Ok(None);
        }
        let limit = self
            .controls
            .active
            .as_ref()
            .expect("selected active controls")
            .objective_limit;
        Ok(
            match self
                .validation
                .objective_limit(x, point.objective, limit, self.controls.feasibility, c)
                .await?
            {
                Some(true) => Some(working_set::Stop::ObjectiveLimit),
                Some(false) => None,
                None => Some(working_set::Stop::Unverified),
            },
        )
    }
    fn small_step(&self, change: working_set::Change) -> bool {
        change.length.is_finite()
            && change.reference.is_finite()
            && change.length / change.reference
                <= self.controls.active.as_ref().expect("active controls").step
    }
}
