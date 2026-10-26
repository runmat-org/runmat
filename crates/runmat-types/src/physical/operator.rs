//! Linear operators retain both physical coordinate layouts. A heterogeneous
//! stiffness matrix is never assigned one invented scalar quantity.
use super::{
    DynamicDimension, Quantity, QuantityField, QuantityFieldError, QuantityKind, QuantityLayout,
    QuantityMarker, QuantitySpec,
};

/// Backend-neutral numerical matrix storage. Physical producers bind it to
/// input/output layouts once; kernels receive raw numbers only inside that view.
pub trait QuantityOperatorStorage {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    fn all_finite(&self) -> bool;
    fn coefficient(&self, row: usize, column: usize) -> f64;
    fn apply_canonical(&self, input: &[f64]) -> Result<Vec<f64>, &'static str>;
}
impl<S: QuantityOperatorStorage + ?Sized> QuantityOperatorStorage for &S {
    fn rows(&self) -> usize {
        (**self).rows()
    }
    fn columns(&self) -> usize {
        (**self).columns()
    }
    fn all_finite(&self) -> bool {
        (**self).all_finite()
    }
    fn coefficient(&self, row: usize, column: usize) -> f64 {
        (**self).coefficient(row, column)
    }
    fn apply_canonical(&self, input: &[f64]) -> Result<Vec<f64>, &'static str> {
        (**self).apply_canonical(input)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformQuantityLayout<K: QuantitySpec> {
    count: usize,
    _marker: K,
}
impl<K: QuantitySpec> UniformQuantityLayout<K> {
    pub const fn from_descriptor(count: usize, descriptor: K) -> Self {
        Self {
            count,
            _marker: descriptor,
        }
    }
}
impl<K: QuantityMarker> UniformQuantityLayout<K> {
    pub const fn new(count: usize) -> Self {
        Self {
            count,
            _marker: K::MARKER,
        }
    }
    pub const fn kind(&self) -> QuantityKind {
        K::KIND
    }
}
impl<K: QuantitySpec> QuantityLayout for UniformQuantityLayout<K> {
    fn component_count(&self) -> usize {
        self.count
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        if index >= self.count {
            return None;
        }
        DynamicDimension::try_with_angle(
            self._marker.physical_dimension(),
            self._marker.nominal_angle_power(),
            self._marker.semantic_kind(),
        )
        .ok()
    }
}

/// Layout kinds are checked by the compiler as well as runtime descriptors.
///
/// ```compile_fail
/// use runmat_types::physical::*;
/// fn wrong_input<S: QuantityOperatorStorage>(
///     operator: &QuantityLinearOperator<UniformQuantityLayout<kind::TemperatureDifference>, UniformQuantityLayout<kind::Power>, S>,
///     pressure: &QuantityField<UniformQuantityLayout<kind::Pressure>, Vec<f64>>,
/// ) { let _ = operator.apply(pressure); }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct QuantityLinearOperator<I, O, S> {
    input: I,
    output: O,
    storage: S,
}
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QuantityOperatorError {
    #[error("physical operator storage shape differs from its coordinate layouts")]
    ShapeMismatch,
    #[error("physical operator coordinate descriptor is incomplete")]
    MissingDescriptor,
    #[error("physical operator input belongs to a different coordinate layout")]
    InputLayoutMismatch,
    #[error("physical operator storage or result contains a nonfinite value")]
    NonFinite,
    #[error("physical operator numerical application failed: {0}")]
    Numerical(&'static str),
}
impl From<QuantityFieldError> for QuantityOperatorError {
    fn from(error: QuantityFieldError) -> Self {
        match error {
            QuantityFieldError::ShapeMismatch => Self::ShapeMismatch,
            QuantityFieldError::MissingComponentDescriptor => Self::MissingDescriptor,
            QuantityFieldError::InvalidValue(_) => Self::NonFinite,
        }
    }
}
impl<I: QuantityLayout, O: QuantityLayout, S: QuantityOperatorStorage>
    QuantityLinearOperator<I, O, S>
{
    pub fn try_new(input: I, output: O, storage: S) -> Result<Self, QuantityOperatorError> {
        if storage.columns() != input.component_count()
            || storage.rows() != output.component_count()
        {
            return Err(QuantityOperatorError::ShapeMismatch);
        }
        if !storage.all_finite() {
            return Err(QuantityOperatorError::NonFinite);
        }
        for i in 0..input.component_count() {
            if input.component_descriptor(i).is_none() {
                return Err(QuantityOperatorError::MissingDescriptor);
            }
        }
        for i in 0..output.component_count() {
            if output.component_descriptor(i).is_none() {
                return Err(QuantityOperatorError::MissingDescriptor);
            }
        }
        Ok(Self {
            input,
            output,
            storage,
        })
    }
    pub fn input_layout(&self) -> &I {
        &self.input
    }
    pub fn output_layout(&self) -> &O {
        &self.output
    }
    pub fn coefficient(&self, row: usize, column: usize) -> Option<Quantity<DynamicDimension>> {
        let descriptor = self
            .output
            .component_descriptor(row)?
            .divide(self.input.component_descriptor(column)?)?;
        Quantity::try_from_descriptor(descriptor, self.storage.coefficient(row, column)).ok()
    }
    /// The layout stays attached while backend-specific preparation/application
    /// accesses its canonical numerical storage.
    pub fn with_canonical_storage<T>(&self, numerical: impl FnOnce(&S) -> T) -> T {
        numerical(&self.storage)
    }
    pub fn apply<J: QuantityLayout, T: AsRef<[f64]>>(
        &self,
        input: &QuantityField<J, T>,
    ) -> Result<QuantityField<O, Vec<f64>>, QuantityOperatorError>
    where
        I: PartialEq<J>,
        O: Clone,
    {
        if &self.input != input.layout() {
            return Err(QuantityOperatorError::InputLayoutMismatch);
        }
        let output = input
            .with_canonical_storage(|values| self.storage.apply_canonical(values))
            .map_err(QuantityOperatorError::Numerical)?;
        QuantityField::try_new(self.output.clone(), output).map_err(QuantityOperatorError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::kind;
    use crate::PhysicalDimension;
    struct Dense([f64; 4]);
    impl QuantityOperatorStorage for Dense {
        fn rows(&self) -> usize {
            2
        }
        fn columns(&self) -> usize {
            2
        }
        fn all_finite(&self) -> bool {
            self.0.iter().all(|x| x.is_finite())
        }
        fn coefficient(&self, r: usize, c: usize) -> f64 {
            self.0[r * 2 + c]
        }
        fn apply_canonical(&self, x: &[f64]) -> Result<Vec<f64>, &'static str> {
            Ok(vec![
                self.0[0] * x[0] + self.0[1] * x[1],
                self.0[2] * x[0] + self.0[3] * x[1],
            ])
        }
    }
    #[test]
    fn physical_operator_retains_input_output_kinds_and_rejects_bad_storage() {
        let input = UniformQuantityLayout::<kind::TemperatureDifference>::new(2);
        let output = UniformQuantityLayout::<kind::Power>::new(2);
        let operator =
            QuantityLinearOperator::try_new(input, output, Dense([2., 0., 0., 3.])).unwrap();
        let temperature = QuantityField::try_new(input, vec![10., 20.]).unwrap();
        let power = operator.apply(&temperature).unwrap();
        assert_eq!(
            power.component(0).unwrap().semantic_kind(),
            Some(QuantityKind::Power)
        );
        assert_eq!(power.component(1).unwrap().canonical_value(), 60.);
        assert_eq!(
            operator.coefficient(0, 0).unwrap().dimension(),
            QuantityKind::Power
                .dimension()
                .divide(PhysicalDimension::TEMPERATURE)
                .unwrap()
        );
        assert!(
            QuantityLinearOperator::try_new(input, output, Dense([f64::NAN, 0., 0., 3.])).is_err()
        );
        assert!(QuantityLinearOperator::try_new(
            UniformQuantityLayout::<kind::TemperatureDifference>::new(3),
            output,
            Dense([1.; 4])
        )
        .is_err());
    }
}
