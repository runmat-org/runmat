//! GSS sampling over borrowed original rows and the shared exact cone.
//! Continuous proposals are checked before integer repair or any evaluation.
use super::{domain::BoxView, invalid, linear_view::OriginalRows, IDENTITY, INVALID_ARGUMENT};
use crate::builtins::math::optim::common::{
    feasible_cone::{Cone, Diagonal, Normal, Policy},
    ScalarCooperation,
};
use crate::BuiltinResult;
use num_rational::BigRational;
use num_traits::Signed;

pub(super) struct Geometry {
    rows: OriginalRows,
}
pub(super) enum Proposal {
    Point { point: Vec<f64> },
    Blocked,
    Unrepresentable,
}
fn policy() -> Policy {
    Policy {
        identity: IDENTITY,
        invalid: &INVALID_ARGUMENT,
    }
}
fn buffer<T>(count: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(count).map_err(|source| {
        crate::build_runtime_error("surrogateopt: GSS sampling workspace allocation failed")
            .with_builtin(super::NAME)
            .with_source(source)
            .build()
    })
}
impl Geometry {
    pub(super) async fn new(
        domain: &BoxView<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            rows: OriginalRows::new(domain, domain.linear().rows(), cooperation).await?,
        })
    }
    #[cfg(test)]
    pub(super) async fn tangent(
        &self,
        domain: &BoxView<'_>,
        anchor: &[f64],
        radius: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Cone> {
        self.tangent_in(domain, anchor, radius, domain, cooperation)
            .await
    }
    pub(super) async fn tangent_in(
        &self,
        domain: &BoxView<'_>,
        anchor: &[f64],
        radius: f64,
        diagonal: &impl Diagonal,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Cone> {
        if !radius.is_finite() || radius <= 0.0 || anchor.len() != domain.variables() {
            return Err(invalid("invalid GSS anchor or radius"));
        }
        let slack = self.rows.exact_slack(domain, anchor, cooperation).await?;
        let mut cone = Cone::full(policy(), domain.dimensions(), cooperation).await?;
        // Internal normalized binding policy. Actual candidate feasibility is
        // checked against every original row, including inactive constraints.
        let epsilon = radius.min(1e-6);
        for (row, slack) in slack.iter().enumerate() {
            let coefficients = self.rows.coefficients(row, domain, cooperation).await?;
            let normal = Normal::new(&coefficients, diagonal, policy(), cooperation).await?;
            let equality = row < domain.linear().equalities();
            if equality || normal.distance_exact(slack) <= epsilon {
                cone.intersect_normal(&normal, false, cooperation).await?;
                if equality {
                    cone.intersect_normal(&normal, true, cooperation).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        for axis in 0..domain.dimensions() {
            let value = domain.unit_coordinate(anchor, axis);
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(invalid("GSS anchor is outside effective original bounds"));
            }
            let column = domain.free_columns()[axis];
            let center = BigRational::from_float(anchor[column]).expect("finite anchor");
            let (lower, upper) = domain.effective_bounds(column);
            let lower_slack = &center - BigRational::from_float(lower).expect("finite lower");
            let upper_slack = BigRational::from_float(upper).expect("finite upper") - center;
            for (coefficient, slack) in [(-1.0, lower_slack), (1.0, upper_slack)] {
                let normal =
                    Normal::new(&[(axis, coefficient)], diagonal, policy(), cooperation).await?;
                if normal.distance_exact(&slack) <= epsilon {
                    cone.intersect_normal(&normal, false, cooperation).await?;
                }
            }
            cooperation.completed(1).await?;
        }
        Ok(cone)
    }

    pub(super) async fn candidate(
        &self,
        domain: &BoxView<'_>,
        anchor: &[f64],
        ray: &[f64],
        radius: f64,
        tolerance: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Proposal> {
        if anchor.len() != domain.variables()
            || ray.len() != domain.dimensions()
            || !radius.is_finite()
            || radius <= 0.0
            || !tolerance.is_finite()
            || tolerance < 0.0
        {
            return Err(invalid("invalid GSS proposal geometry"));
        }
        let mut origin = buffer(ray.len())?;
        let mut length = radius;
        for (axis, &direction) in ray.iter().enumerate() {
            let value = domain.unit_coordinate(anchor, axis);
            if !direction.is_finite() || !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(invalid("nonfinite or out-of-bounds GSS anchor/ray"));
            }
            length = length.min(if direction > 0.0 {
                (1.0 - value) / direction
            } else if direction < 0.0 {
                -value / direction
            } else {
                f64::INFINITY
            });
            origin.push(value);
            cooperation.completed(1).await?;
        }
        if length <= 0.0 {
            return Ok(Proposal::Blocked);
        }
        let mut unit = buffer(ray.len())?;
        unit.resize(ray.len(), 0.0);
        let mut point = buffer(anchor.len())?;
        point.resize(anchor.len(), 0.0);
        let tolerance = BigRational::from_float(tolerance).expect("finite tolerance");
        loop {
            let mut in_box = true;
            for ((coordinate, &base), &direction) in unit.iter_mut().zip(&origin).zip(ray) {
                *coordinate = length.mul_add(direction, base);
                in_box &= coordinate.is_finite() && (0.0..=1.0).contains(coordinate);
                cooperation.completed(1).await?;
            }
            if in_box {
                domain
                    .continuous_denormalize_into(&unit, &mut point, cooperation)
                    .await?;
                let mut changed = false;
                for (&value, &base) in point.iter().zip(anchor) {
                    changed |= value != base;
                    cooperation.completed(1).await?;
                }
                if !changed {
                    return Ok(Proposal::Unrepresentable);
                }
                let slack = self.rows.exact_slack(domain, &point, cooperation).await?;
                let mut feasible = true;
                for (row, value) in slack.iter().enumerate() {
                    feasible &= if row < domain.linear().equalities() {
                        value.abs() <= tolerance
                    } else {
                        value >= &(-&tolerance)
                    };
                    cooperation.completed(1).await?;
                }
                if feasible {
                    return Ok(Proposal::Point { point });
                }
            }
            // No false convergence or finite-grid proof: reduce only this
            // ray, ending when no representable positive step remains.
            length *= 0.5;
            if length == 0.0 {
                return Ok(Proposal::Unrepresentable);
            }
            cooperation.completed(1).await?;
        }
    }
}
