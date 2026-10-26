//! Finite-box coordinate view over the one original feasibility domain.
//! Actual trial coordinates are retained; normalization is computed on reads.
use super::invalid;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::builtins::math::optim::intlinprog::feasibility::{Domain as Feasible, OriginalLinear};
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;

pub(super) enum Geometry {
    Inconsistent,
    Unique,
    Variable,
}
pub(super) struct BoxGeometry {
    free: Vec<usize>,
    geometry: Geometry,
    variables: usize,
}
pub(super) struct BoxView<'a> {
    domain: &'a Feasible,
    metadata: &'a BoxGeometry,
}
impl crate::builtins::math::optim::common::feasible_cone::Diagonal for BoxView<'_> {
    fn dimension(&self) -> usize {
        self.dimensions()
    }
    fn factor(&self, column: usize) -> BigRational {
        self.exact_width(column)
    }
}
impl BoxGeometry {
    pub(super) async fn new(
        domain: &Feasible,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let (lower, upper) = domain.bounds();
        if lower.is_empty() || lower.len() != upper.len() {
            return Err(invalid("bounds must have matching nonempty vector extents"));
        }
        let mut free = Vec::new();
        free.try_reserve_exact(lower.len()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: free-coordinate allocation failed")
                .with_builtin(super::NAME)
                .with_source(source)
                .build()
        })?;
        let mut inconsistent = false;
        for (column, (&lower, &upper)) in lower.iter().zip(upper).enumerate() {
            if !lower.is_finite() || !upper.is_finite() {
                return Err(invalid("all bound coordinates must be finite"));
            }
            inconsistent |= lower > upper;
            let (lower, upper) = if domain.integers().binary_search(&column).is_ok() {
                (lower.ceil(), upper.floor())
            } else {
                (lower, upper)
            };
            inconsistent |= lower > upper;
            if lower < upper {
                free.push(column);
            }
            cooperation.completed(1).await?;
        }
        let geometry = if inconsistent {
            Geometry::Inconsistent
        } else if free.is_empty() {
            Geometry::Unique
        } else {
            Geometry::Variable
        };
        Ok(Self {
            free,
            geometry,
            variables: lower.len(),
        })
    }
    /// Borrow only for one numerical turn. The controller drops this view
    /// before mutably repairing a proposal in the original shared domain.
    pub(super) fn view<'a>(&'a self, domain: &'a Feasible) -> BuiltinResult<BoxView<'a>> {
        if domain.bounds().0.len() != self.variables {
            return Err(invalid(
                "original sampling domain changed its variable extent",
            ));
        }
        Ok(BoxView {
            domain,
            metadata: self,
        })
    }
}
impl BoxView<'_> {
    pub(super) fn same_domain(&self, domain: &Feasible) -> bool {
        std::ptr::eq(self.domain, domain)
    }
    pub(super) fn variables(&self) -> usize {
        self.domain.bounds().0.len()
    }
    pub(super) fn dimensions(&self) -> usize {
        self.metadata.free.len()
    }
    pub(super) fn free_columns(&self) -> &[usize] {
        &self.metadata.free
    }
    pub(super) fn integers(&self) -> &[usize] {
        self.domain.integers()
    }
    /// The shared domain retains original supplied bounds for membership
    /// proof. Sampling computes integer-rounded bounds on reads, not in a
    /// second bound vector; singleton integer coordinates are fixed axes.
    pub(super) fn effective_bounds(&self, column: usize) -> (f64, f64) {
        let (lower, upper) = self.domain.bounds();
        if self.domain.integers().binary_search(&column).is_ok() {
            (lower[column].ceil(), upper[column].floor())
        } else {
            (lower[column], upper[column])
        }
    }
    pub(super) fn linear(&self) -> OriginalLinear<'_> {
        self.domain.original_linear()
    }
    pub(super) fn exact_width(&self, axis: usize) -> BigRational {
        let column = self.metadata.free[axis];
        let (lower, upper) = self.effective_bounds(column);
        BigRational::from_float(upper).expect("finite upper")
            - BigRational::from_float(lower).expect("finite lower")
    }
    pub(super) fn unit_coordinate(&self, point: &[f64], axis: usize) -> f64 {
        let column = self.metadata.free[axis];
        let (lower, upper) = self.effective_bounds(column);
        normalize(point[column], lower, upper)
    }
    pub(super) fn normalize_coordinate(&self, value: f64, axis: usize) -> f64 {
        let column = self.metadata.free[axis];
        let (lower, upper) = self.effective_bounds(column);
        normalize(value, lower, upper)
    }
    pub(super) fn interpolate_coordinate(&self, axis: usize, unit: f64) -> BuiltinResult<f64> {
        if axis >= self.dimensions() || !unit.is_finite() || !(0.0..=1.0).contains(&unit) {
            return Err(invalid("invalid original free-coordinate interpolation"));
        }
        let (lower, upper) = self.effective_bounds(self.metadata.free[axis]);
        Ok(interpolate(unit, lower, upper))
    }
    pub(super) fn integer_interval(
        &self,
        axis: usize,
        center: f64,
        scale: f64,
    ) -> BuiltinResult<(BigInt, BigInt)> {
        let column = self
            .free_columns()
            .get(axis)
            .copied()
            .ok_or_else(|| invalid("invalid integer sampling axis"))?;
        let (lower, upper) = self.effective_bounds(column);
        if self.integers().binary_search(&column).is_err()
            || !scale.is_finite()
            || scale <= 0.0
            || scale > 0.8
            || !center.is_finite()
            || center.fract() != 0.0
            || center < lower
            || center > upper
        {
            return Err(invalid("invalid original integer sampling interval"));
        }
        let ratio = BigRational::from_float(scale * 2.5).expect("finite scale");
        let radius = (self.exact_width(axis) * ratio).max(BigRational::one());
        let center = BigRational::from_float(center).expect("finite anchor");
        let lower = (center.clone() - &radius).ceil().to_integer().max(
            BigRational::from_float(lower)
                .expect("finite lower")
                .to_integer(),
        );
        let upper = (center + radius).floor().to_integer().min(
            BigRational::from_float(upper)
                .expect("finite upper")
                .to_integer(),
        );
        Ok((lower, upper))
    }
    pub(super) async fn denormalize_into(
        &self,
        unit: &[f64],
        point: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.convert_into(unit, point, true, cooperation).await
    }
    pub(super) async fn continuous_denormalize_into(
        &self,
        unit: &[f64],
        point: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.convert_into(unit, point, false, cooperation).await
    }
    async fn convert_into(
        &self,
        unit: &[f64],
        point: &mut [f64],
        round_integers: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if !matches!(
            self.metadata.geometry,
            Geometry::Variable | Geometry::Unique
        ) || unit.len() != self.dimensions()
            || point.len() != self.variables()
            || unit
                .iter()
                .any(|t| !t.is_finite() || !(0.0..=1.0).contains(t))
        {
            return Err(invalid("invalid finite-box sampling geometry"));
        }
        let mut axis = 0;
        for column in 0..point.len() {
            let (lower, upper) = self.effective_bounds(column);
            let value = if self.metadata.free.get(axis) == Some(&column) {
                let value = interpolate(unit[axis], lower, upper);
                axis += 1;
                value
            } else {
                lower
            };
            point[column] = if round_integers && self.integers().binary_search(&column).is_ok() {
                value.round().max(lower).min(upper)
            } else {
                value
            };
            cooperation.completed(1).await?;
        }
        Ok(())
    }
}

/// Endpoint branches preserve the actual original bounds. The half-span form
/// handles opposite finite extremes without forming an infinite full width.
fn normalize(value: f64, lower: f64, upper: f64) -> f64 {
    if value == lower {
        return 0.0;
    }
    if value == upper {
        return 1.0;
    }
    let width = upper - lower;
    if width.is_finite() {
        (value - lower) / width
    } else {
        (value * 0.5 - lower * 0.5) / (upper * 0.5 - lower * 0.5)
    }
}
fn interpolate(unit: f64, lower: f64, upper: f64) -> f64 {
    if unit == 0.0 {
        return lower;
    }
    if unit == 1.0 {
        return upper;
    }
    let width = upper - lower;
    let value = if width.is_finite() {
        unit.mul_add(width, lower)
    } else {
        (1.0 - unit) * lower + unit * upper
    };
    // Only rounding at the finite convex-combination boundary is clamped;
    // no objective measurement is reused at this newly constructed point.
    value.max(lower).min(upper)
}

#[cfg(test)]
mod tests {
    #[test]
    fn finite_extreme_box_has_real_interior_and_exact_endpoints() {
        let (lower, upper) = (-f64::MAX, f64::MAX);
        assert_eq!(super::interpolate(0.0, lower, upper), lower);
        assert_eq!(super::interpolate(1.0, lower, upper), upper);
        assert_eq!(super::interpolate(0.5, lower, upper), 0.0);
        assert_eq!(super::normalize(0.0, lower, upper), 0.5);
        for unit in [0.125, 0.25, 0.75, 0.875] {
            let value = super::interpolate(unit, lower, upper);
            assert!(value.is_finite() && value > lower && value < upper);
            assert!((super::normalize(value, lower, upper) - unit).abs() < 1e-15);
        }
        let tiny = f64::from_bits(1);
        assert_eq!(super::interpolate(0.0, 0.0, tiny), 0.0);
        assert_eq!(super::interpolate(1.0, 0.0, tiny), tiny);
        assert_eq!(super::normalize(tiny, 0.0, tiny), 1.0);
    }
}
