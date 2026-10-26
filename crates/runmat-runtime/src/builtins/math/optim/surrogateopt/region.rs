//! One derived local/tree bound workspace over borrowed original geometry.
//! Restoring a branch changes current limits in place, never cloning the
//! original CSC model or retaining a box/point payload for every tree node.
use super::{domain::BoxView, invalid, NAME};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};

#[derive(Clone, Copy)]
pub(super) enum Scope {
    Local(f64),
    Original,
}
#[derive(Clone, Copy)]
pub(super) enum Neighbor {
    Lower,
    Upper,
}
pub(super) struct Region<'a> {
    domain: BoxView<'a>,
    anchor: &'a [f64],
    scope: Scope,
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl<'a> Region<'a> {
    pub(super) async fn new(
        domain: BoxView<'a>,
        anchor: &'a [f64],
        scope: Scope,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        cooperation.completed(0).await?;
        if anchor.len() != domain.variables()
            || matches!(scope, Scope::Local(scale) if !scale.is_finite() || scale <= 0.0 || scale > 0.8)
        {
            return Err(invalid("invalid local refinement region"));
        }
        for (column, &value) in anchor.iter().enumerate() {
            let (lower, upper) = domain.effective_bounds(column);
            if !value.is_finite()
                || value < lower
                || value > upper
                || domain.integers().binary_search(&column).is_ok() && value.fract() != 0.0
            {
                return Err(invalid(
                    "refinement anchor is outside the original effective domain",
                ));
            }
            cooperation.completed(1).await?;
        }
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            NAME,
            "finite host-memory execution has no complete local refinement bound",
        )?;
        let allocate = || {
            runmat_value::exact_capacity_vec(domain.variables()).map_err(|source| {
                crate::build_runtime_error("surrogateopt: local bound allocation failed")
                    .with_builtin(NAME)
                    .with_source(source)
                    .build()
            })
        };
        let mut region = Self {
            lower: allocate()?,
            upper: allocate()?,
            domain,
            anchor,
            scope,
        };
        region.restore(cooperation).await?;
        Ok(region)
    }
    pub(super) fn bounds(&self) -> (&[f64], &[f64]) {
        (&self.lower, &self.upper)
    }
    pub(super) fn same_domain(
        &self,
        domain: &crate::builtins::math::optim::intlinprog::feasibility::Domain,
    ) -> bool {
        self.domain.same_domain(domain)
    }
    pub(super) fn variables(&self) -> usize {
        self.domain.variables()
    }
    pub(super) fn dimensions(&self) -> usize {
        self.domain.dimensions()
    }
    pub(super) fn integers(&self) -> &[usize] {
        self.domain.integers()
    }
    pub(super) async fn restore(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        self.lower.clear();
        self.upper.clear();
        let mut axis = 0usize;
        for (column, &center) in self.anchor.iter().enumerate() {
            let (original_lower, original_upper) = self.domain.effective_bounds(column);
            let (lower, upper) = match self.scope {
                Scope::Original => (original_lower, original_upper),
                Scope::Local(_) if original_lower == original_upper => {
                    (original_lower, original_upper)
                }
                Scope::Local(scale) if self.domain.integers().binary_search(&column).is_ok() => {
                    let (lower, upper) = self.domain.integer_interval(axis, center, scale)?;
                    (finite_integer(&lower)?, finite_integer(&upper)?)
                }
                Scope::Local(scale) => {
                    let center = self.domain.unit_coordinate(self.anchor, axis);
                    (
                        self.domain
                            .interpolate_coordinate(axis, (center - scale).max(0.0))?,
                        self.domain
                            .interpolate_coordinate(axis, (center + scale).min(1.0))?,
                    )
                }
            };
            self.lower.push(lower);
            self.upper.push(upper);
            axis += usize::from(original_lower != original_upper);
            cooperation.completed(1).await?;
        }
        Ok(())
    }
    /// Fix a genuine integer assignment for a continuous SQP subproblem.
    /// Failure is only an empty branch in the current derived region.
    pub(super) fn pin_integer(&mut self, column: usize, value: f64) -> BuiltinResult<bool> {
        if column >= self.variables()
            || self.integers().binary_search(&column).is_err()
            || !value.is_finite()
            || value.fract() != 0.0
        {
            return Err(invalid("invalid integer refinement assignment"));
        }
        if value < self.lower[column] || value > self.upper[column] {
            return Ok(false);
        }
        self.lower[column] = value;
        self.upper[column] = value;
        Ok(true)
    }
    /// An integer neighbor means an actual changed representable coordinate,
    /// not a floating x+1 operation which may round back to the same x.
    pub(super) fn neighbor(
        &self,
        column: usize,
        center: f64,
        direction: Neighbor,
    ) -> BuiltinResult<Option<f64>> {
        if column >= self.variables()
            || self.integers().binary_search(&column).is_err()
            || !center.is_finite()
            || center.fract() != 0.0
            || center < self.lower[column]
            || center > self.upper[column]
        {
            return Err(invalid("invalid integer refinement neighbor"));
        }
        let integer = BigRational::from_float(center)
            .expect("finite integer center")
            .to_integer();
        let target = match direction {
            Neighbor::Lower => integer - BigInt::one(),
            Neighbor::Upper => integer + BigInt::one(),
        };
        let Some(mut value) = target.to_f64().filter(|value| value.is_finite()) else {
            return Ok(None);
        };
        let represented = BigRational::from_float(value)
            .expect("finite neighbor")
            .to_integer();
        match direction {
            Neighbor::Lower if represented > target => value = value.next_down(),
            Neighbor::Upper if represented < target => value = value.next_up(),
            _ => {}
        }
        if !value.is_finite()
            || value < self.lower[column]
            || value > self.upper[column]
            || value == center
        {
            return Ok(None);
        }
        Ok(Some(value))
    }
}
fn finite_integer(value: &BigInt) -> BuiltinResult<f64> {
    value
        .to_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid("derived integer bound is unrepresentable"))
}
