//! Original affine Lagrangian bounds for the existing integer tree. Numerical
//! duals and current node boxes are proposals, never source coefficient facts.
use super::*;
pub(super) struct Objective {
    gradient: Vec<Rational>,
    constant: Rational,
}
impl Authority<'_> {
    async fn prepare_integer_objective(&mut self, c: &mut ScalarCooperation) -> BuiltinResult<()> {
        if self.integer_objective.is_some() {
            return Ok(());
        }
        let policy = self.policy;
        self.prepare_origin(c).await?;
        let origin = self.origin.as_ref().expect("original affine intercepts");
        let sense = self
            .formulation
            .objective
            .as_ref()
            .map_or(Sense::Minimize, |objective| objective.sense);
        let mut seeds = Vec::new();
        let mut constant = match &self.program.objective {
            objective::ModelObjective::Affine => {
                if let Some(objective) = &self.formulation.objective {
                    seeds.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    seeds.push(exact::Seed {
                        root: objective.root,
                        row: 0,
                        value: if sense == Sense::Minimize {
                            Rational::from_usize(1)
                        } else {
                            -Rational::from_usize(1)
                        },
                    });
                    origin
                        .value(objective.root, 0, policy)?
                        .ok_or_else(|| policy.invalid("original integer objective is unassigned"))?
                } else {
                    Rational::zero()
                }
            }
            objective::ModelObjective::ConstantSquares(squares) => {
                squares.value(origin, self.program, policy, c).await?
            }
            _ => {
                return Err(
                    policy.invalid("integer affine proof cannot replace a nonlinear objective")
                )
            }
        };
        if sense == Sense::Maximize {
            constant = -constant;
        }
        let gradient = exact::accumulate(self.program, origin, seeds, policy, c).await?;
        self.integer_objective = Some(Objective { gradient, constant });
        Ok(())
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn integer_objective(
        &mut self,
        point: &[Rational],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rational> {
        self.prepare_integer_objective(c).await?;
        let policy = self.policy;
        let objective = self
            .integer_objective
            .as_ref()
            .expect("original objective evidence");
        if point.len() != objective.gradient.len() {
            return Err(policy.invalid("integer objective point geometry changed"));
        }
        let mut value = objective.constant.clone();
        for (entry, coefficient) in point.iter().zip(&objective.gradient) {
            value = value
                .add(&coefficient.mul(entry, policy, c).await?, policy, c)
                .await?;
            c.completed(1).await?;
        }
        Ok(value)
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn integer_lower_bound(
        &mut self,
        roles: &[RowRole],
        multipliers: &[Rational],
        bounds: BoundValues<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Rational>> {
        self.prepare_integer_objective(c).await?;
        let policy = self.policy;
        if bounds.lower.len() != self.program.layout.columns
            || bounds.upper.len() != self.program.layout.columns
            || roles.len() != multipliers.len()
        {
            return Ok(None);
        }
        for (role, value) in roles.iter().zip(multipliers) {
            if !matches!(role, RowRole::Equality(_)) && value.is_negative() {
                return Ok(None);
            }
            c.completed(1).await?;
        }
        let Some(weighted) = self
            .weighted_rows_at(roles, multipliers, Some(bounds), c)
            .await?
        else {
            return Ok(None);
        };
        let objective = self
            .integer_objective
            .as_ref()
            .expect("original objective evidence");
        let mut value = objective
            .constant
            .add(&weighted.intercept, policy, c)
            .await?;
        for (column, (cost, row)) in objective
            .gradient
            .iter()
            .zip(&weighted.gradient)
            .enumerate()
        {
            let residual = cost.add(row, policy, c).await?;
            if !residual.is_zero() {
                let bound = if residual.is_positive() {
                    bounds.lower[column]
                } else {
                    bounds.upper[column]
                };
                let Some(bound) = Rational::from_float(bound) else {
                    return Ok(None);
                };
                value = value
                    .add(&residual.mul(&bound, policy, c).await?, policy, c)
                    .await?;
            }
            c.completed(1).await?;
        }
        Ok(Some(value))
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) async fn integer_farkas(
        &mut self,
        roles: &[RowRole],
        multipliers: &[Rational],
        bounds: BoundValues<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if bounds.lower.len() != self.program.layout.columns
            || bounds.upper.len() != self.program.layout.columns
            || roles.len() != multipliers.len()
        {
            return Ok(false);
        }
        for (role, value) in roles.iter().zip(multipliers) {
            if !matches!(role, RowRole::Equality(_)) && value.is_negative() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        let Some(weighted) = self
            .weighted_rows_at(roles, multipliers, Some(bounds), c)
            .await?
        else {
            return Ok(false);
        };
        if !weighted.intercept.is_positive() {
            return Ok(false);
        }
        for value in weighted.gradient {
            if !value.is_zero() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
}
