//! Equation residual optimality and hard original bounds are separate evidence.
use super::*;
use crate::builtins::math::optim::{
    common::{conic::LinearRowRole as Role, proof_rational::Rational},
    lsqlin::compiled as solver,
};
use runmat_value::optimization_variable::OptimizationBoundRole;
pub(super) struct Validator<'a> {
    pub program: &'a Program,
    pub point: Option<numerical::Point>,
    pub policy: Policy,
    origin: exact::Evaluated,
    scale: proof::ObjectiveScale,
}
impl<'a> Validator<'a> {
    pub(super) async fn new(
        program: &'a Program,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let objective::ModelObjective::Equations(evidence) = &program.objective else {
            return Err(policy.invalid("equation validation requires original residual evidence"));
        };
        let origin = exact::evaluate(
            program,
            exact::Coordinates::Origin,
            exact::Retain::ConstantOperands,
            policy,
            c,
        )
        .await?;
        let scale = evidence.scale(program, &origin, policy, c).await?;
        Ok(Self {
            program,
            point: None,
            policy,
            origin,
            scale,
        })
    }
    fn evidence(&self) -> &Evidence {
        let objective::ModelObjective::Equations(evidence) = &self.program.objective else {
            unreachable!("validated equation evidence")
        };
        evidence
    }
    async fn prepare(&mut self, point: &[f64], c: &mut ScalarCooperation) -> BuiltinResult<bool> {
        if let Some(existing) = &self.point {
            if existing
                .matches(self.program, point, self.policy, c)
                .await?
            {
                return Ok(true);
            }
        }
        self.point = numerical::Point::evaluate(
            self.program,
            &self.evidence().formulation.roots,
            point,
            self.policy,
            c,
        )
        .await?;
        Ok(self.point.is_some())
    }
    fn bound(&self, role: Role) -> Option<(usize, f64, bool)> {
        self.original_bound(role)
            .filter(|(_, value, _)| value.is_finite())
    }
    fn original_bound(&self, role: Role) -> Option<(usize, f64, bool)> {
        let (column, lower) = match role {
            Role::Lower(column) => (column, true),
            Role::Upper(column) => (column, false),
            Role::Equality(_) | Role::Inequality(_) => return None,
        };
        let (variable, row) = self.program.layout.coordinate(column)?;
        let value = variable.view.definition().bound_value(
            if lower {
                OptimizationBoundRole::Lower
            } else {
                OptimizationBoundRole::Upper
            },
            row,
        )?;
        Some((column, value, lower))
    }
    async fn primal(
        &self,
        point: &[f64],
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if point.len() != self.program.layout.columns || !tolerance.is_finite() || tolerance < 0. {
            return Ok(false);
        }
        for (column, &value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Ok(false);
            }
            for role in [Role::Lower(column), Role::Upper(column)] {
                if let Some((_, bound, lower)) = self.bound(role) {
                    let scale = 1.0f64.max(value.abs()).max(bound.abs());
                    let residual = if lower {
                        bound / scale - value / scale
                    } else {
                        value / scale - bound / scale
                    };
                    if residual > tolerance {
                        return Ok(false);
                    }
                }
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
    async fn kkt(
        &self,
        point: &[f64],
        rows: solver::Rows<'_>,
        duals: &[f64],
        tolerance: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(tolerance) = Rational::from_float(tolerance).filter(|x| !x.is_negative()) else {
            return Ok(false);
        };
        let Some(duals) = rows.external(duals) else {
            return Ok(false);
        };
        let mut candidate =
            runmat_value::exact_capacity_vec(point.len()).map_err(|e| self.policy.storage(e))?;
        let mut norm = Rational::zero();
        for &value in point {
            let Some(value) = Rational::from_float(value) else {
                return Ok(false);
            };
            if value.abs().compare(&norm, self.policy, c).await?.is_gt() {
                norm = value.abs();
            }
            candidate.push(value);
            c.completed(1).await?;
        }
        let evaluated = exact::evaluate(
            self.program,
            exact::Coordinates::Point(&candidate),
            exact::Retain::Roots,
            self.policy,
            c,
        )
        .await?;
        let (seeds, cost) = self
            .evidence()
            .seeds(&evaluated, &self.origin, self.policy, c)
            .await?;
        let gradient = exact::accumulate(self.program, &self.origin, seeds, self.policy, c).await?;
        let scale = self
            .scale
            .magnitude(&candidate, &norm, self.policy, c)
            .await?;
        let mut row_gradient =
            runmat_value::exact_capacity_vec(point.len()).map_err(|e| self.policy.storage(e))?;
        for _ in point {
            row_gradient.push(Rational::zero());
            c.completed(1).await?;
        }
        let mut intercept = Rational::zero();
        let mut complement = Vec::new();
        for (role, &weight) in rows.iter().zip(duals) {
            let Some((column, bound, lower)) = self.bound(role) else {
                return Ok(false);
            };
            let Some(weight) = Rational::from_float(weight) else {
                return Ok(false);
            };
            if weight.is_negative()
                && weight
                    .abs()
                    .compare(
                        &tolerance
                            .mul(
                                &scale.add(&weight.abs(), self.policy, c).await?,
                                self.policy,
                                c,
                            )
                            .await?,
                        self.policy,
                        c,
                    )
                    .await?
                    .is_gt()
            {
                return Ok(false);
            }
            let bound = Rational::from_float(bound).expect("finite original bound");
            let residual = if lower {
                bound.sub(&candidate[column], self.policy, c).await?
            } else {
                candidate[column].sub(&bound, self.policy, c).await?
            };
            let signed = if lower {
                -weight.clone()
            } else {
                weight.clone()
            };
            row_gradient[column] = row_gradient[column].add(&signed, self.policy, c).await?;
            intercept = intercept
                .sub(&signed.mul(&bound, self.policy, c).await?, self.policy, c)
                .await?;
            complement
                .try_reserve(1)
                .map_err(|e| self.policy.reserve(e))?;
            complement.push(weight.mul(&residual, self.policy, c).await?.abs());
            c.completed(1).await?;
        }
        for (column, (objective, row)) in gradient.iter().zip(&row_gradient).enumerate() {
            let mut magnitude = self.scale.at(column, &norm, self.policy, c).await?;
            if magnitude.is_zero() {
                magnitude = Rational::from_usize(1);
            }
            magnitude = magnitude
                .add(&objective.abs(), self.policy, c)
                .await?
                .add(&row.abs(), self.policy, c)
                .await?;
            if objective
                .add(row, self.policy, c)
                .await?
                .abs()
                .compare(
                    &tolerance.mul(&magnitude, self.policy, c).await?,
                    self.policy,
                    c,
                )
                .await?
                .is_gt()
            {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        let threshold = tolerance
            .mul(
                &scale
                    .add(&cost.abs(), self.policy, c)
                    .await?
                    .add(&intercept.abs(), self.policy, c)
                    .await?,
                self.policy,
                c,
            )
            .await?;
        if cost
            .sub(&intercept, self.policy, c)
            .await?
            .abs()
            .compare(&threshold, self.policy, c)
            .await?
            .is_gt()
        {
            return Ok(false);
        }
        for value in complement {
            if value.compare(&threshold, self.policy, c).await?.is_gt() {
                return Ok(false);
            }
            c.completed(1).await?;
        }
        Ok(true)
    }
}
impl solver::Validation for Validator<'_> {
    async fn stopped_candidate(
        &mut self,
        point: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.prepare(point, c).await
    }
    async fn objective_limit(
        &mut self,
        point: &[f64],
        _cost: f64,
        limit: f64,
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        if !self.prepare(point, c).await? {
            return Ok(None);
        }
        if !self.primal(point, feasibility, c).await? {
            return Ok(Some(false));
        }
        let residual = self
            .evidence()
            .residual(
                self.point.as_ref().expect("prepared original point"),
                self.policy,
                c,
            )
            .await?;
        let mut cost = 0.;
        for value in residual {
            cost += value * value;
            c.completed(1).await?;
        }
        Ok(Some(cost < limit))
    }
    async fn solution(
        &mut self,
        point: &[f64],
        rows: solver::Rows<'_>,
        duals: &[f64],
        feasibility: f64,
        optimality: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, c).await? || !self.primal(point, feasibility, c).await? {
            return Ok(false);
        }
        self.kkt(point, rows, duals, optimality, c).await
    }
    async fn incumbent(
        &mut self,
        point: &[f64],
        feasibility: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if !self.prepare(point, c).await? {
            return Ok(false);
        }
        self.primal(point, feasibility, c).await
    }
    async fn certificate(
        &mut self,
        certificate: solver::Certificate<'_>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        match certificate {
            solver::Certificate::InconsistentBounds {
                column,
                lower,
                upper,
            } => {
                let Some((_, original_lower, _)) = self.original_bound(Role::Lower(column)) else {
                    return Ok(false);
                };
                let Some((_, original_upper, _)) = self.original_bound(Role::Upper(column)) else {
                    return Ok(false);
                };
                Ok(lower.to_bits() == original_lower.to_bits()
                    && upper.to_bits() == original_upper.to_bits()
                    && original_lower > original_upper)
            }
            solver::Certificate::Farkas { rows, multipliers } => {
                let Some(multipliers) = rows.external(multipliers) else {
                    return Ok(false);
                };
                let mut gradient = runmat_value::exact_capacity_vec(self.program.layout.columns)
                    .map_err(|e| self.policy.storage(e))?;
                for _ in 0..self.program.layout.columns {
                    gradient.push(Rational::zero());
                    c.completed(1).await?;
                }
                let mut intercept = Rational::zero();
                for (role, multiplier) in rows.iter().zip(multipliers) {
                    let Some((column, bound, lower)) = self.bound(role) else {
                        return Ok(false);
                    };
                    let Some(weight) = Rational::import(multiplier, self.policy, c).await? else {
                        return Ok(false);
                    };
                    if weight.is_negative() {
                        return Ok(false);
                    }
                    let signed = if lower { -weight } else { weight };
                    gradient[column] = gradient[column].add(&signed, self.policy, c).await?;
                    let bound = Rational::from_float(bound).expect("finite original bound");
                    intercept = intercept
                        .sub(&signed.mul(&bound, self.policy, c).await?, self.policy, c)
                        .await?;
                    c.completed(1).await?;
                }
                if !intercept.is_positive() {
                    return Ok(false);
                }
                for coefficient in gradient {
                    if !coefficient.is_zero() {
                        return Ok(false);
                    }
                    c.completed(1).await?;
                }
                Ok(true)
            }
        }
    }
}
