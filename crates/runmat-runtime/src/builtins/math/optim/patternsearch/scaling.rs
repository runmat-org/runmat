//! Exact power-of-two coordinate scaling, selected from logarithmic magnitude.
//! The original model is not overwritten. Equality-constrained callers select
//! unit scaling as specified by the classic algorithm's applicability rule.
use super::{buffer, invalid};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

pub(super) struct Scaling {
    dimension: usize,
    diagonal: Option<Vec<f64>>,
}
impl Scaling {
    pub fn unit(dimension: usize) -> Self {
        Self {
            dimension,
            diagonal: None,
        }
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn factor(&self, coordinate: usize) -> f64 {
        self.diagonal
            .as_ref()
            .map_or(1.0, |values| values[coordinate])
    }
    pub async fn logarithmic(
        x: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut diagonal = buffer(x.len())?;
        for &value in x {
            if !value.is_finite() {
                return Err(invalid("mesh scale point must be finite"));
            }
            let absolute = value.abs();
            let bits = absolute.to_bits();
            let exponent = (bits >> 52) & 0x7ff;
            let power = if absolute == 0.0 {
                1.0
            } else if exponent != 0 {
                f64::from_bits(exponent << 52)
            } else {
                let fraction = bits & ((1u64 << 52) - 1);
                f64::from_bits(1u64 << (63 - fraction.leading_zeros()))
            };
            diagonal.push(power);
            cooperation.completed(1).await?;
        }
        Ok(Self {
            dimension: x.len(),
            diagonal: Some(diagonal),
        })
    }
    pub async fn apply(
        &self,
        direction: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if direction.len() != self.dimension {
            return Err(invalid("mesh scale dimension mismatch"));
        }
        for (coordinate, value) in direction.iter_mut().enumerate() {
            let scaled = *value * self.factor(coordinate);
            if !scaled.is_finite() || *value != 0.0 && scaled == 0.0 {
                return Ok(false);
            }
            *value = scaled;
            cooperation.completed(1).await?;
        }
        Ok(true)
    }
}
