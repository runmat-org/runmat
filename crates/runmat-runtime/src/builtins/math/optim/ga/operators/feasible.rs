//! Adaptive feasible and positive-basis mutation use original polyhedral
//! geometry. They are not aliases of coordinate power mutation or LP rounding.
use super::super::{common, invalid, linear, random, ScalarCooperation, IDENTITY, POLICY};
use crate::BuiltinResult;
use common::{
    feasible_cone::{self, Basis, Cone, Diagonal, Normal},
    global_time::TimeLimit,
    solver_timing::SolverTiming,
};
use num_rational::BigRational;
use num_traits::{One, Zero};
use runmat_builtins::OptimGeneticMutation;

const CONE_POLICY: feasible_cone::Policy = feasible_cone::Policy {
    identity: IDENTITY,
    invalid: &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
};
pub(in super::super) struct Scaling(Vec<BigRational>);
impl Scaling {
    pub async fn new(
        ranges: &[(f64, f64)],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut values = POLICY.capacity(ranges.len(), cooperation.runtime())?;
        for &(lower, upper) in ranges {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(invalid(
                    "feasible mutation requires finite ordered coordinate ranges",
                ));
            }
            let width = BigRational::from_float(upper).expect("finite range")
                - BigRational::from_float(lower).expect("finite range");
            values.push(if width.is_zero() {
                BigRational::one()
            } else {
                width
            });
            cooperation.completed(1).await?;
        }
        Ok(Self(values))
    }
}
impl Diagonal for Scaling {
    fn dimension(&self) -> usize {
        self.0.len()
    }
    fn factor(&self, column: usize) -> BigRational {
        self.0[column].clone()
    }
}

async fn generators(
    domain: &impl linear::Geometry,
    parent: &[f64],
    scaling: &Scaling,
    tolerance: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Cone> {
    if parent.len() != scaling.dimension()
        || parent.len() != domain.bounds().0.len()
        || !tolerance.is_finite()
        || tolerance < 0.
        || parent.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid feasible-mutation parent or scaling"));
    }
    let mut cone = Cone::full(CONE_POLICY, parent.len(), cooperation).await?;
    let (lower, upper) = domain.bounds();
    for (column, &value) in parent.iter().enumerate() {
        if value <= lower[column] {
            let normal = Normal::new(&[(column, -1.)], scaling, CONE_POLICY, cooperation).await?;
            cone.intersect_normal(&normal, false, cooperation).await?;
        }
        if value >= upper[column] {
            let normal = Normal::new(&[(column, 1.)], scaling, CONE_POLICY, cooperation).await?;
            cone.intersect_normal(&normal, false, cooperation).await?;
        }
        cooperation.completed(1).await?;
    }
    let tolerance = BigRational::from_float(tolerance).expect("finite tolerance");
    let mut row = domain.row_scratch(cooperation)?;
    for index in 0..domain.rhs().len() {
        domain.original_row(index, &mut row, cooperation).await?;
        let mut value = BigRational::zero();
        for &(column, coefficient) in &row {
            value += BigRational::from_float(coefficient).expect("original finite coefficient")
                * BigRational::from_float(parent[column]).expect("finite parent");
            cooperation.completed(1).await?;
        }
        let slack =
            BigRational::from_float(domain.rhs()[index]).expect("finite original rhs") - value;
        if index < domain.equalities() || slack <= tolerance {
            let normal = Normal::new(&row, scaling, CONE_POLICY, cooperation).await?;
            cone.intersect_normal(&normal, false, cooperation).await?;
            if index < domain.equalities() {
                cone.intersect_normal(&normal, true, cooperation).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(cone)
}

pub(in super::super) struct Request<'a> {
    pub kind: OptimGeneticMutation,
    pub parent: &'a [f64],
    pub scaling: &'a Scaling,
    pub tolerance: f64,
    pub timing: &'a SolverTiming,
    pub deadline: TimeLimit,
}
pub(in super::super) struct Adaptive {
    magnitude: f64,
    cursor: usize,
    reverse: bool,
}
impl Adaptive {
    pub fn new() -> Self {
        Self {
            magnitude: 1.,
            cursor: 0,
            reverse: false,
        }
    }
    pub fn completed_generation(&mut self, improved: bool) {
        self.magnitude = if improved {
            (2. * self.magnitude).min(f64::MAX)
        } else {
            0.5 * self.magnitude
        };
        if !improved {
            self.reverse = !self.reverse;
        }
        self.cursor = self.cursor.wrapping_add(1);
    }
    pub async fn propose(
        &self,
        domain: &mut impl linear::Geometry,
        request: Request<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<linear::step::Step> {
        use linear::step::{Controls, Step};
        if request.deadline.reached(request.timing.elapsed()) {
            return Ok(Step::TimeLimit);
        }
        let cone = generators(
            domain,
            request.parent,
            request.scaling,
            request.tolerance,
            cooperation,
        )
        .await?;
        let basis =
            match request.kind {
                OptimGeneticMutation::AdaptiveFeasible => Basis::Maximal,
                OptimGeneticMutation::PositiveBasis => Basis::Minimal,
                _ => return Err(invalid(
                    "this mutation owner requires adaptive-feasible or positive-basis selection",
                )),
            };
        let count = cone.len(basis)?;
        if count == 0 || self.magnitude == 0. {
            return Ok(Step::Blocked);
        }
        let (index, draw) = if request.kind == OptimGeneticMutation::AdaptiveFeasible {
            let draws = random::uniform(2, cooperation).await?;
            (
                ((draws[0] * count as f64) as usize).min(count - 1),
                draws[1],
            )
        } else {
            (
                self.cursor % count,
                random::uniform(1, cooperation).await?[0],
            )
        };
        let Some(direction) = cone
            .oriented_direction(index, basis, self.reverse, cooperation)
            .await?
        else {
            return Ok(Step::Unverified);
        };
        let mut physical = POLICY.capacity(direction.len(), cooperation.runtime())?;
        for (column, &value) in direction.iter().enumerate() {
            physical.push(
                request.scaling.factor(column)
                    * BigRational::from_float(value)
                        .ok_or_else(|| invalid("feasible cone returned a nonfinite direction"))?,
            );
            cooperation.completed(1).await?;
        }
        domain
            .step(
                request.parent,
                &physical,
                Controls {
                    maximum: self.magnitude,
                    draw,
                    tolerance: request.tolerance,
                    timing: request.timing,
                    deadline: request.deadline,
                },
                cooperation,
            )
            .await
    }
}
