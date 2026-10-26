//! Original-row stationarity/gap/complementarity at the supplied floating
//! candidate. Exact arithmetic prevents rounded coefficient erasure from
//! authorizing optimality; ordinary primal tolerance remains a separate check.
use super::super::quadratic::directional_cost;
use super::*;
mod phases;
impl<'source> Authority<'source> {
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn kkt<'a>(
        &'a mut self,
        point: &'a [f64],
        roles: &'a [RowRole],
        duals: &'a [f64],
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a + use<'a, 'source> {
        self.kkt_at_tolerance(
            point,
            roles,
            duals,
            super::super::super::super::linprog::compiled::validation_tolerance(),
            cooperation,
        )
    }
    pub(in crate::builtins::math::optim::optimmodel::lowering) fn kkt_at_tolerance<'a>(
        &'a mut self,
        point: &'a [f64],
        roles: &'a [RowRole],
        duals: &'a [f64],
        tolerance: f64,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<bool>> + 'a + use<'a, 'source> {
        crate::execution::after_result(
            phases::prepare(self, point, roles, duals, tolerance, cooperation),
            phases::select,
        )
    }
    async fn certify(
        &mut self,
        data: phases::Data,
        roles: &[RowRole],
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let phases::Data {
            candidate,
            weights,
            weighted,
        } = data;
        let policy = self.policy;
        if self.objective_scale.is_none() {
            if let Some(objective) = self.program.objective.squares() {
                self.objective_scale = Some(
                    objective
                        .scale(
                            self.program,
                            self.origin.as_ref().expect("original affine intercepts"),
                            policy,
                            cooperation,
                        )
                        .await?,
                );
            }
        }
        let data_scale = self
            .program
            .objective
            .quadratic()
            .map(|evidence| &evidence.scale)
            .or(self.objective_scale.as_ref());
        let mut x_norm = Rational::zero();
        if data_scale.is_some() {
            for value in &candidate {
                let magnitude = value.abs();
                if magnitude
                    .compare(&x_norm, policy, cooperation)
                    .await?
                    .is_gt()
                {
                    x_norm = magnitude;
                }
                cooperation.completed(1).await?;
            }
        }
        let origin = self.origin.as_ref().expect("original affine intercepts");
        let mut evaluated = if self.program.objective.squares().is_some() {
            Some(
                exact::evaluate(
                    self.program,
                    exact::Coordinates::Point(&candidate),
                    exact::Retain::Roots,
                    policy,
                    cooperation,
                )
                .await?,
            )
        } else {
            None
        };
        let mut differential_cost = None;
        let mut seeds = Vec::new();
        if let Some(objective) = self.program.objective.squares() {
            let (weights, cost) = objective
                .seeds(
                    evaluated.as_ref().expect("original residual point"),
                    origin,
                    self.program,
                    policy,
                    cooperation,
                )
                .await?;
            seeds = weights;
            differential_cost = Some(cost);
        } else if let Some(objective) = self
            .formulation
            .objective
            .as_ref()
            .filter(|_| self.program.objective.quadratic().is_none())
        {
            seeds.try_reserve(1).map_err(|e| policy.reserve(e))?;
            seeds.push(exact::Seed {
                root: objective.root,
                row: 0,
                value: rational(
                    if objective.sense == Sense::Minimize {
                        1.
                    } else {
                        -1.
                    },
                    policy,
                )?,
            });
        }
        let objective_gradient = if let Some(evidence) = self.program.objective.quadratic() {
            let sense = self
                .formulation
                .objective
                .as_ref()
                .map_or(Sense::Minimize, |objective| objective.sense);
            let gradient = evidence
                .gradient(
                    self.program,
                    exact::Coordinates::Point(&candidate),
                    sense,
                    policy,
                    cooperation,
                )
                .await?;
            differential_cost =
                Some(directional_cost(&gradient, &candidate, policy, cooperation).await?);
            gradient
        } else {
            exact::accumulate(self.program, origin, seeds, policy, cooperation).await?
        };
        let objective_scale = if let Some(scale) = data_scale {
            scale
                .magnitude(&candidate, &x_norm, policy, cooperation)
                .await?
        } else {
            let mut objective_scale = Rational::zero();
            for value in &objective_gradient {
                if value
                    .abs()
                    .compare(&objective_scale, policy, cooperation)
                    .await?
                    .is_gt()
                {
                    objective_scale = value.abs();
                }
                cooperation.completed(1).await?;
            }
            if objective_scale.is_zero() {
                objective_scale = rational(1., policy)?;
            }
            objective_scale
        };
        let tolerance = rational(tolerance, policy)?;
        for (column, (objective, row)) in objective_gradient
            .into_iter()
            .zip(&weighted.gradient)
            .enumerate()
        {
            let column_scale = if let Some(scale) = data_scale {
                let value = scale.at(column, &x_norm, policy, cooperation).await?;
                if value.is_zero() {
                    Rational::from_usize(1)
                } else {
                    value
                }
            } else {
                objective_scale.clone()
            };
            let magnitude = column_scale
                .add(&objective.abs(), policy, cooperation)
                .await?
                .add(&row.abs(), policy, cooperation)
                .await?;
            if objective
                .add(row, policy, cooperation)
                .await?
                .abs()
                .compare(
                    &tolerance.mul(&magnitude, policy, cooperation).await?,
                    policy,
                    cooperation,
                )
                .await?
                .is_gt()
            {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        let evaluated = match evaluated.take() {
            Some(value) => value,
            None => {
                exact::evaluate(
                    self.program,
                    exact::Coordinates::Point(&candidate),
                    exact::Retain::Roots,
                    policy,
                    cooperation,
                )
                .await?
            }
        };
        let mut cost = differential_cost.unwrap_or_else(Rational::zero);
        if self.program.objective.squares().is_none()
            && self.program.objective.quadratic().is_none()
        {
            if let Some(objective) = &self.formulation.objective {
                let Some(value) =
                    difference(&evaluated, origin, objective.root, 0, policy, cooperation).await?
                else {
                    return Ok(false);
                };
                cost = if objective.sense == Sense::Minimize {
                    value
                } else {
                    -value
                };
            }
        }
        let scale = objective_scale
            .add(&cost.abs(), policy, cooperation)
            .await?
            .add(&weighted.intercept.abs(), policy, cooperation)
            .await?;
        let threshold = tolerance.mul(&scale, policy, cooperation).await?;
        if cost
            .sub(&weighted.intercept, policy, cooperation)
            .await?
            .abs()
            .compare(&threshold, policy, cooperation)
            .await?
            .is_gt()
        {
            return Ok(false);
        }
        for (role, weight) in roles.iter().zip(&weights) {
            if matches!(role, RowRole::Equality(_)) {
                continue;
            }
            if weight.is_negative()
                && weight
                    .abs()
                    .compare(
                        &tolerance
                            .mul(
                                &objective_scale
                                    .add(&weight.abs(), policy, cooperation)
                                    .await?,
                                policy,
                                cooperation,
                            )
                            .await?,
                        policy,
                        cooperation,
                    )
                    .await?
                    .is_gt()
            {
                return Ok(false);
            }
            let Some(residual) = self
                .residual(role, &candidate, &evaluated, cooperation)
                .await?
            else {
                return Ok(false);
            };
            if weight
                .mul(&residual, policy, cooperation)
                .await?
                .abs()
                .compare(&threshold, policy, cooperation)
                .await?
                .is_gt()
            {
                return Ok(false);
            }
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
    async fn residual(
        &self,
        role: &RowRole,
        point: &[Rational],
        evaluated: &exact::Evaluated,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Rational>> {
        let policy = self.policy;
        match *role {
            RowRole::Equality(row) | RowRole::Inequality(row) => {
                let ranges = if matches!(role, RowRole::Equality(_)) {
                    &self.equalities
                } else {
                    &self.inequalities
                };
                let index = ranges.partition_point(|range| range.end <= row);
                let Some(range) = ranges.get(index).filter(|range| row >= range.start) else {
                    return Ok(None);
                };
                let constraint = &self.formulation.constraints[range.constraint];
                let local = row - range.start;
                let Some((a, b)) = self.geometries[range.constraint].indices(local) else {
                    return Ok(None);
                };
                let Some(left) = evaluated.value(constraint.left, a, policy)? else {
                    return Ok(None);
                };
                let Some(right) = evaluated.value(constraint.right, b, policy)? else {
                    return Ok(None);
                };
                Ok(Some(if constraint.relation == Relation::GreaterEqual {
                    right.sub(&left, policy, cooperation).await?
                } else {
                    left.sub(&right, policy, cooperation).await?
                }))
            }
            RowRole::Lower(column) | RowRole::Upper(column) => {
                let Some((variable, row)) = self.program.layout.coordinate(column) else {
                    return Ok(None);
                };
                let lower = matches!(role, RowRole::Lower(_));
                let Some(bound) = variable
                    .view
                    .definition()
                    .bound_value(if lower { Bound::Lower } else { Bound::Upper }, row)
                    .filter(|value| value.is_finite())
                else {
                    return Ok(None);
                };
                let bound = rational(bound, policy)?;
                Ok(Some(if lower {
                    bound.sub(&point[column], policy, cooperation).await?
                } else {
                    point[column].sub(&bound, policy, cooperation).await?
                }))
            }
        }
    }
}
