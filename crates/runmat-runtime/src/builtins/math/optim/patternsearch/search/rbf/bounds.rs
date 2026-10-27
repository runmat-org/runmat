//! Original box endpoints, with absent endpoints derived from current samples.
use super::super::super::buffer;
use super::super::Bounds;
use super::history::History;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

pub(in crate::builtins::math::optim::patternsearch) struct Window {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl Bounds {
    pub async fn axes(
        &self,
        geometry: &super::super::super::linear::Linear,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<crate::builtins::math::optim::common::feasible_cone::IndependentAxes> {
        geometry
            .rbf_axes(&self.lower, &self.upper, cooperation)
            .await
    }
    pub async fn window(
        &self,
        history: &History,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Window>> {
        if history.len() == 0 {
            return Ok(None);
        }
        let mut lower = buffer(self.lower.len())?;
        let mut upper = buffer(self.upper.len())?;
        for axis in 0..self.lower.len() {
            let mut l = self.lower[axis];
            let mut u = self.upper[axis];
            if !l.is_finite() || !u.is_finite() {
                let mut minimum = f64::INFINITY;
                let mut maximum = f64::NEG_INFINITY;
                for row in 0..history.len() {
                    let value = history.coordinate(row, axis);
                    minimum = minimum.min(value);
                    maximum = maximum.max(value);
                    cooperation.completed(1).await?;
                }
                if !l.is_finite() {
                    l = minimum;
                }
                if !u.is_finite() {
                    u = maximum;
                }
            }
            if !l.is_finite() || !u.is_finite() || l > u {
                return Ok(None);
            }
            lower.push(l);
            upper.push(u);
            cooperation.completed(1).await?;
        }
        Ok(Some(Window { lower, upper }))
    }
}
impl Window {
    pub fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.lower.len()
            && point
                .iter()
                .enumerate()
                .all(|(axis, &x)| x.is_finite() && x >= self.lower[axis] && x <= self.upper[axis])
    }
    /// A stable finite bound on distance along a normalized direction. Dividing
    /// endpoints first avoids overflowing u-x for opposite finite extremes.
    pub async fn length(
        &self,
        point: &[f64],
        direction: &[f64],
        maximum: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        if !self.contains(point)
            || direction.len() != point.len()
            || !maximum.is_finite()
            || maximum <= 0.0
        {
            return Ok(None);
        }
        let mut length = maximum;
        for (axis, (&x, &d)) in point.iter().zip(direction).enumerate() {
            if !d.is_finite() {
                return Ok(None);
            }
            if d != 0.0 {
                let edge = if d > 0.0 {
                    self.upper[axis]
                } else {
                    self.lower[axis]
                };
                let distance = edge - x;
                let ratio = if distance.is_finite() {
                    distance / d
                } else {
                    edge / d - x / d
                };
                // An infinite positive ratio does not restrict a finite step.
                if ratio.is_nan() || ratio < 0.0 {
                    return Ok(None);
                }
                length = length.min(ratio);
            }
            cooperation.completed(1).await?;
        }
        Ok((length > 0.0 && length.is_finite()).then_some(length))
    }
}
