//! Componentwise estimate algebra for one adaptive quadrature traversal.

use std::collections::TryReserveError;

use super::geometry::{simpson, ScaledEstimate};

#[derive(Debug)]
pub(crate) enum QuadratureArithmeticError {
    Resource(TryReserveError),
    InconsistentComponents,
}

pub(crate) trait AdaptiveEstimate: Clone {
    fn add(&self, other: &Self) -> Result<Self, QuadratureArithmeticError>;
    fn subtract(&self, other: &Self) -> Result<Self, QuadratureArithmeticError>;
    fn multiply_f64(&self, factor: f64) -> Result<Self, QuadratureArithmeticError>;
    fn tolerance(&self, absolute: f64, relative: f64) -> Result<Self, QuadratureArithmeticError>;
    fn abs_le(&self, other: &Self) -> bool;
}

pub(crate) trait QuadratureSample: Clone {
    type Estimate: AdaptiveEstimate;

    fn simpson(
        a: f64,
        b: f64,
        fa: &Self,
        fm: &Self,
        fb: &Self,
    ) -> Result<Self::Estimate, QuadratureArithmeticError>;
}

impl AdaptiveEstimate for ScaledEstimate {
    fn add(&self, other: &Self) -> Result<Self, QuadratureArithmeticError> {
        Ok((*self).add(*other))
    }

    fn subtract(&self, other: &Self) -> Result<Self, QuadratureArithmeticError> {
        Ok((*self).subtract(*other))
    }

    fn multiply_f64(&self, factor: f64) -> Result<Self, QuadratureArithmeticError> {
        Ok((*self).multiply_f64(factor))
    }

    fn tolerance(&self, absolute: f64, relative: f64) -> Result<Self, QuadratureArithmeticError> {
        Ok(ScaledEstimate::from_f64(absolute).max(self.abs().multiply_f64(relative)))
    }

    fn abs_le(&self, other: &Self) -> bool {
        (*self).abs_le(*other)
    }
}

impl QuadratureSample for f64 {
    type Estimate = ScaledEstimate;

    fn simpson(
        a: f64,
        b: f64,
        fa: &Self,
        fm: &Self,
        fb: &Self,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        Ok(simpson(a, b, *fa, *fm, *fb))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ArrayEstimate(pub(crate) Vec<ScaledEstimate>);

impl ArrayEstimate {
    fn zip_map(
        &self,
        other: &Self,
        map: impl Fn(ScaledEstimate, ScaledEstimate) -> ScaledEstimate,
    ) -> Result<Self, QuadratureArithmeticError> {
        if self.0.len() != other.0.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.0.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        for (&left, &right) in self.0.iter().zip(&other.0) {
            values.push(map(left, right));
        }
        Ok(Self(values))
    }

    fn map(
        &self,
        map: impl Fn(ScaledEstimate) -> ScaledEstimate,
    ) -> Result<Self, QuadratureArithmeticError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.0.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        values.extend(self.0.iter().copied().map(map));
        Ok(Self(values))
    }
}

impl AdaptiveEstimate for ArrayEstimate {
    fn add(&self, other: &Self) -> Result<Self, QuadratureArithmeticError> {
        self.zip_map(other, ScaledEstimate::add)
    }

    fn subtract(&self, other: &Self) -> Result<Self, QuadratureArithmeticError> {
        self.zip_map(other, ScaledEstimate::subtract)
    }

    fn multiply_f64(&self, factor: f64) -> Result<Self, QuadratureArithmeticError> {
        self.map(|value| value.multiply_f64(factor))
    }

    fn tolerance(&self, absolute: f64, relative: f64) -> Result<Self, QuadratureArithmeticError> {
        self.map(|value| ScaledEstimate::from_f64(absolute).max(value.abs().multiply_f64(relative)))
    }

    fn abs_le(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(left, right)| (*left).abs_le(*right))
    }
}

impl QuadratureSample for Vec<f64> {
    type Estimate = ArrayEstimate;

    fn simpson(
        a: f64,
        b: f64,
        fa: &Self,
        fm: &Self,
        fb: &Self,
    ) -> Result<Self::Estimate, QuadratureArithmeticError> {
        if fa.len() != fm.len() || fa.len() != fb.len() {
            return Err(QuadratureArithmeticError::InconsistentComponents);
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(fa.len())
            .map_err(QuadratureArithmeticError::Resource)?;
        for ((&left, &middle), &right) in fa.iter().zip(fm).zip(fb) {
            components.push(simpson(a, b, left, middle, right));
        }
        Ok(ArrayEstimate(components))
    }
}
