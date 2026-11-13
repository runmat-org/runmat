//! Original equality-nullspace curvature, using one source reverse plan.
use super::*;
use crate::builtins::math::optim::common::conic::convexity::certify_restricted;
use std::collections::BTreeMap;
impl Authority<'_> {
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn restricted_convexity(
        &mut self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        c.completed(0).await?;
        if let Some(checked) = self.restricted_curvature {
            return Ok(checked);
        }
        let checked = self.compute_restricted_convexity(c).await?;
        self.restricted_curvature = Some(checked);
        Ok(checked)
    }
    async fn compute_restricted_convexity(
        &mut self,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(evidence) = self.program.objective.quadratic() else {
            return Ok(false);
        };
        if evidence.curvature == super::super::quadratic::Curvature::Convex {
            return Ok(true);
        }
        let sense = self
            .formulation
            .objective
            .as_ref()
            .map_or(Sense::Minimize, |objective| objective.sense);
        let Some(hessian) = evidence
            .exact_hessian(self.program.layout.columns, sense, self.policy, c)
            .await?
        else {
            return Ok(false);
        };
        self.prepare_origin(c).await?;
        let origin = self.origin.as_ref().expect("original affine operands");
        let mut reverse = exact::SparsePlan::new(self.program, self.policy, c).await?;
        let mut equations = Vec::new();
        for range in &self.equalities {
            let original = &self.formulation.constraints[range.constraint];
            let geometry = &self.geometries[range.constraint];
            for (_, left, right) in geometry.plan.iter() {
                let mut row: BTreeMap<usize, Rational> = BTreeMap::new();
                for (root, index, negative) in
                    [(original.left, left, false), (original.right, right, true)]
                {
                    let terms = reverse
                        .row(self.program, origin, root, index, self.policy, c)
                        .await?;
                    for (column, value) in terms {
                        let value = if negative { value.negated() } else { value };
                        let value = match row.remove(&column) {
                            Some(previous) => value.add(&previous, self.policy, c).await?,
                            None => value,
                        };
                        if !value.is_zero() {
                            row.insert(column, value);
                        }
                        c.completed(1).await?;
                    }
                }
                equations
                    .try_reserve(1)
                    .map_err(|e| self.policy.reserve(e))?;
                equations.push(row);
                c.completed(1).await?;
            }
        }
        certify_restricted(hessian, equations, self.policy, c).await
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn quadratic_recession(
        &mut self,
        point: &[Candidate],
        direction: &[Candidate],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(point) = imported(point, self.policy, c).await? else {
            return Ok(false);
        };
        let Some(direction) = imported(direction, self.policy, c).await? else {
            return Ok(false);
        };
        if !self.feasible(&point, c).await?
            || self.direction_geometry(&direction, c).await?.is_none()
            || !self.restricted_convexity(c).await?
        {
            return Ok(false);
        }
        let Some(evidence) = self.program.objective.quadratic() else {
            return Ok(false);
        };
        let sense = self
            .formulation
            .objective
            .as_ref()
            .map_or(Sense::Minimize, |value| value.sense);
        let gradient = evidence
            .gradient(
                self.program,
                exact::Coordinates::Point(&point),
                sense,
                self.policy,
                c,
            )
            .await?;
        let at_zero = evidence
            .gradient(
                self.program,
                exact::Coordinates::Origin,
                sense,
                self.policy,
                c,
            )
            .await?;
        let displaced = evidence
            .gradient(
                self.program,
                exact::Coordinates::Point(&direction),
                sense,
                self.policy,
                c,
            )
            .await?;
        if gradient.len() != direction.len()
            || at_zero.len() != direction.len()
            || displaced.len() != direction.len()
        {
            return Ok(false);
        }
        let mut curvature = Rational::zero();
        let mut slope = Rational::zero();
        for (((step, gradient), zero), displaced) in
            direction.iter().zip(gradient).zip(at_zero).zip(displaced)
        {
            let hd = displaced.sub(&zero, self.policy, c).await?;
            curvature = curvature
                .add(&step.mul(&hd, self.policy, c).await?, self.policy, c)
                .await?;
            slope = slope
                .add(&step.mul(&gradient, self.policy, c).await?, self.policy, c)
                .await?;
            c.completed(1).await?;
        }
        Ok(curvature.is_zero() && slope.is_negative())
    }
}
