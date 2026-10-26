//! Original-model terminal authority supplements the assembled LP checks.
//! A rejected candidate is numerical unverified, never the opposite terminal.
use super::super::super::linprog::compiled::{CertificateCandidate, RowRole};
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
use num_rational::BigRational as Candidate;

use runmat_value::optimization_model::OptimizationConstraintRelation as Relation;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;

use runmat_value::optimization_variable::OptimizationBoundRole as Bound;
mod farkas;
mod integer;
mod quadratic;
#[derive(Clone, Copy)]
pub(super) struct BoundValues<'a> {
    pub lower: &'a [f64],
    pub upper: &'a [f64],
}
mod geometry;
mod kkt;
mod objective_scale;
pub(super) use objective_scale::ObjectiveScale;
struct Range {
    start: usize,
    end: usize,
    constraint: usize,
}
pub(super) struct Authority<'a> {
    program: &'a Program,
    formulation: &'a formulation::Formulation,
    geometries: Vec<geometry::Geometry>,
    equalities: Vec<Range>,
    inequalities: Vec<Range>,
    origin: Option<exact::Evaluated>,
    objective_scale: Option<ObjectiveScale>,
    restricted_curvature: Option<bool>,
    integer_objective: Option<integer::Objective>,
    policy: Policy,
}
impl<'a> Authority<'a> {
    pub(super) async fn new(
        program: &'a Program,
        formulation: &'a formulation::Formulation,
        rows: &[assembly::Rows],
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if rows.len() != formulation.constraints.len() {
            return Err(policy.invalid("assembled row roles differ from original constraints"));
        }
        let mut out = Self {
            program,
            formulation,
            geometries: runmat_value::exact_capacity_vec(rows.len())
                .map_err(|e| policy.storage(e))?,
            equalities: Vec::new(),
            inequalities: Vec::new(),
            origin: None,
            objective_scale: None,
            restricted_curvature: None,
            integer_objective: None,
            policy,
        };
        for (index, row) in rows.iter().enumerate() {
            let original = &formulation.constraints[index];
            out.geometries
                .push(geometry::Geometry::new(program, original, policy)?);
            if row.count != evaluation::element_count(&original.shape, policy)?
                || row.equality != (original.relation == Relation::Equal)
            {
                return Err(
                    policy.invalid("assembled row role differs from its original constraint")
                );
            }
            let ranges = if row.equality {
                &mut out.equalities
            } else {
                &mut out.inequalities
            };
            let expected = ranges.last().map(|range: &Range| range.end).unwrap_or(0);
            if row.start != expected {
                return Err(policy.invalid("assembled original row order is inconsistent"));
            }
            ranges.try_reserve(1).map_err(|e| policy.reserve(e))?;
            ranges.push(Range {
                start: row.start,
                end: row
                    .start
                    .checked_add(row.count)
                    .ok_or_else(|| policy.invalid("original constraint range overflow"))?,
                constraint: index,
            });
            cooperation.completed(1).await?;
        }
        Ok(out)
    }
    async fn prepare_origin(&mut self, cooperation: &mut ScalarCooperation) -> BuiltinResult<()> {
        if self.origin.is_none() {
            self.origin = Some(
                exact::evaluate(
                    self.program,
                    exact::Coordinates::Origin,
                    exact::Retain::ConstantOperands,
                    self.policy,
                    cooperation,
                )
                .await?,
            );
        }
        Ok(())
    }
    pub(super) async fn certificate(
        &mut self,
        candidate: CertificateCandidate<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        match candidate {
            CertificateCandidate::InconsistentBounds {
                column,
                lower,
                upper,
            } => {
                let Some((variable, row)) = self.program.layout.coordinate(column) else {
                    return Ok(false);
                };
                let data = variable.view.definition();
                Ok(data.bound_value(Bound::Lower, row) == Some(lower)
                    && data.bound_value(Bound::Upper, row) == Some(upper)
                    && (lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY))
            }
            CertificateCandidate::Feasible(point) => {
                match imported(point, self.policy, cooperation).await? {
                    Some(point) => self.feasible(&point, cooperation).await,
                    None => Ok(false),
                }
            }
            CertificateCandidate::Descent(direction) => {
                match imported(direction, self.policy, cooperation).await? {
                    Some(direction) => self.descent(&direction, cooperation).await,
                    None => Ok(false),
                }
            }
            CertificateCandidate::Farkas { rows, multipliers } => {
                match imported(multipliers, self.policy, cooperation).await? {
                    Some(weights) => self.farkas(rows, &weights, cooperation).await,
                    None => Ok(false),
                }
            }
        }
    }
    async fn feasible(
        &self,
        point: &[Rational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.feasible_at(point, None, cooperation).await
    }
    pub(super) async fn feasible_at(
        &self,
        point: &[Rational],
        bounds: Option<BoundValues<'_>>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let policy = self.policy;
        if point.len() != self.program.layout.columns {
            return Ok(false);
        }
        for variable in &self.program.layout.variables {
            for row in 0..variable.count {
                let value = &point[variable.start + row];
                if bounds.is_none() && variable.domain.is_semi() && value.is_zero() {
                    cooperation.completed(1).await?;
                    continue;
                }
                let data = variable.view.definition();
                let lower = bounds
                    .map_or_else(
                        || data.bound_value(Bound::Lower, row),
                        |bounds| bounds.lower.get(variable.start + row).copied(),
                    )
                    .ok_or_else(|| policy.invalid("original lower bound is unavailable"))?;
                let upper = bounds
                    .map_or_else(
                        || data.bound_value(Bound::Upper, row),
                        |bounds| bounds.upper.get(variable.start + row).copied(),
                    )
                    .ok_or_else(|| policy.invalid("original upper bound is unavailable"))?;
                if lower == f64::INFINITY
                    || upper == f64::NEG_INFINITY
                    || (lower.is_finite()
                        && value
                            .compare(&rational(lower, policy)?, policy, cooperation)
                            .await?
                            == std::cmp::Ordering::Less)
                    || (upper.is_finite()
                        && value
                            .compare(&rational(upper, policy)?, policy, cooperation)
                            .await?
                            == std::cmp::Ordering::Greater)
                {
                    return Ok(false);
                }
                cooperation.completed(1).await?;
            }
        }
        let evaluated = exact::evaluate(
            self.program,
            exact::Coordinates::Point(point),
            exact::Retain::Roots,
            policy,
            cooperation,
        )
        .await?;
        for (constraint, geometry) in self.formulation.constraints.iter().zip(&self.geometries) {
            for (_, a, b) in geometry.plan.iter() {
                let Some(left) = evaluated.value(constraint.left, a, policy)? else {
                    return Ok(false);
                };
                let Some(right) = evaluated.value(constraint.right, b, policy)? else {
                    return Ok(false);
                };
                let valid = match constraint.relation {
                    Relation::Equal => left.compare(&right, policy, cooperation).await?.is_eq(),
                    Relation::LessEqual => {
                        !left.compare(&right, policy, cooperation).await?.is_gt()
                    }
                    Relation::GreaterEqual => {
                        !left.compare(&right, policy, cooperation).await?.is_lt()
                    }
                };
                if !valid {
                    return Ok(false);
                }
                cooperation.completed(1).await?;
            }
        }
        Ok(true)
    }
    async fn direction_geometry(
        &mut self,
        direction: &[Rational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<exact::Evaluated>> {
        let policy = self.policy;
        if direction.len() != self.program.layout.columns {
            return Ok(None);
        }
        for variable in &self.program.layout.variables {
            for row in 0..variable.count {
                let value = &direction[variable.start + row];
                let data = variable.view.definition();
                let lower = data
                    .bound_value(Bound::Lower, row)
                    .ok_or_else(|| policy.invalid("original lower bound is unavailable"))?;
                let upper = data
                    .bound_value(Bound::Upper, row)
                    .ok_or_else(|| policy.invalid("original upper bound is unavailable"))?;
                if (lower.is_finite() && value.is_negative())
                    || (upper.is_finite() && value.is_positive())
                {
                    return Ok(None);
                }
                cooperation.completed(1).await?;
            }
        }
        self.prepare_origin(cooperation).await?;
        let origin = self.origin.as_ref().expect("original affine intercepts");
        let evaluated = exact::evaluate(
            self.program,
            exact::Coordinates::Point(direction),
            exact::Retain::Roots,
            policy,
            cooperation,
        )
        .await?;
        for (constraint, geometry) in self.formulation.constraints.iter().zip(&self.geometries) {
            for (_, a, b) in geometry.plan.iter() {
                let Some(left) =
                    difference(&evaluated, origin, constraint.left, a, policy, cooperation).await?
                else {
                    return Ok(None);
                };
                let Some(right) =
                    difference(&evaluated, origin, constraint.right, b, policy, cooperation)
                        .await?
                else {
                    return Ok(None);
                };
                let valid = match constraint.relation {
                    Relation::Equal => left.compare(&right, policy, cooperation).await?.is_eq(),
                    Relation::LessEqual => {
                        !left.compare(&right, policy, cooperation).await?.is_gt()
                    }
                    Relation::GreaterEqual => {
                        !left.compare(&right, policy, cooperation).await?.is_lt()
                    }
                };
                if !valid {
                    return Ok(None);
                }
                cooperation.completed(1).await?;
            }
        }
        Ok(Some(evaluated))
    }
    async fn descent(
        &mut self,
        direction: &[Rational],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        let Some(evaluated) = self.direction_geometry(direction, cooperation).await? else {
            return Ok(false);
        };
        let policy = self.policy;
        let origin = self.origin.as_ref().expect("original affine intercepts");
        let Some(objective) = &self.formulation.objective else {
            return Ok(false);
        };
        if let Some(evidence) = self.program.objective.quadratic() {
            return evidence
                .recession(
                    self.program,
                    direction,
                    objective.sense,
                    policy,
                    cooperation,
                )
                .await;
        }
        let Some(value) =
            difference(&evaluated, origin, objective.root, 0, policy, cooperation).await?
        else {
            return Ok(false);
        };
        Ok(if objective.sense == Sense::Minimize {
            value.is_negative()
        } else {
            value.is_positive()
        })
    }
}
fn rational(value: f64, policy: Policy) -> BuiltinResult<Rational> {
    Rational::from_float(value)
        .ok_or_else(|| policy.invalid("exact original evidence requires a finite constant"))
}
async fn difference(
    evaluated: &exact::Evaluated,
    origin: &exact::Evaluated,
    root: runmat_gc::GcHandle,
    row: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Rational>> {
    Ok(
        match (
            evaluated.value(root, row, policy)?,
            origin.value(root, row, policy)?,
        ) {
            (Some(value), Some(offset)) => Some(value.sub(&offset, policy, cooperation).await?),
            _ => None,
        },
    )
}

pub(super) async fn imported(
    values: &[Candidate],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<Rational>>> {
    cooperation.completed(0).await?;
    let mut result =
        runmat_value::exact_capacity_vec(values.len()).map_err(|e| policy.storage(e))?;
    for value in values {
        let Some(value) = Rational::import(value, policy, cooperation).await? else {
            return Ok(None);
        };
        result.push(value);
        cooperation.completed(1).await?;
    }
    Ok(Some(result))
}
