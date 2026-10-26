//! Pressure weak-form coordinates, before identity-row or phase reduction.
//! K p has volume/time², M p has volume, B p has volume/time. Applying the
//! explicit phasor time generator makes every term volume/time². Reduced
//! coordinates inherit pressure because phase-constraint transforms are ratios;
//! prescribed-pressure offsets retain pressure and are lifted before reduction.
use super::AcousticHarmonicError;
use crate::algebra::RealSparseMatrix;
use runmat_types::physical::{
    kind, DynamicDimension, QuantityKind, QuantityLinearOperator, UniformQuantityLayout,
};
pub(super) type PressureOperator = QuantityLinearOperator<
    UniformQuantityLayout<kind::Pressure>,
    UniformQuantityLayout<DynamicDimension>,
    RealSparseMatrix,
>;
pub(super) fn equation_descriptor(time_order: u8) -> DynamicDimension {
    let mut value =
        DynamicDimension::try_new(QuantityKind::Volume.dimension(), Some(QuantityKind::Volume))
            .expect("static volume");
    let time = DynamicDimension::try_new(QuantityKind::Time.dimension(), Some(QuantityKind::Time))
        .expect("static time");
    for _ in 0..time_order {
        value = value.divide(time).expect("acoustic time derivative");
    }
    value
}
pub(super) fn pressure_operator(
    matrix: RealSparseMatrix,
    time_order: u8,
) -> Result<PressureOperator, AcousticHarmonicError> {
    let n = matrix.rows();
    QuantityLinearOperator::try_new(
        UniformQuantityLayout::new(n),
        UniformQuantityLayout::from_descriptor(n, equation_descriptor(time_order)),
        matrix,
    )
    .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn time_derivatives_have_distinct_weak_equation_dimensions() {
        let k = equation_descriptor(2);
        let m = equation_descriptor(0);
        let b = equation_descriptor(1);
        assert_ne!(k, m);
        assert_ne!(b, m);
        assert_eq!(
            k.dimension(),
            runmat_types::PhysicalDimension::new([3, 0, -2, 0, 0, 0, 0, 0])
        );
        assert_eq!(k.angle_power(), 0);
    }
}

/// Quadrature writes through the checked physical source owner. Invalid numeric
/// increments fail before they can poison a subsequently admitted weak equation.
pub(super) fn accumulate_source(
    source: &mut runmat_types::physical::Quantity<
        DynamicDimension,
        Vec<crate::excitation::ProjectionScalar>,
    >,
    node: usize,
    real: f64,
    imaginary: f64,
) -> Result<(), AcousticHarmonicError> {
    let old = source
        .canonical_values()
        .get(node)
        .ok_or_else(|| AcousticHarmonicError::Boundary("acoustic source index".into()))?;
    let next = crate::excitation::ProjectionScalar {
        real: old.real + real,
        imaginary: old.imaginary + imaginary,
    };
    let value = runmat_types::physical::Quantity::try_from_descriptor(equation_descriptor(2), next)
        .map_err(|e| AcousticHarmonicError::Boundary(e.to_string()))?;
    source
        .try_replace(node, value)
        .map_err(|e| AcousticHarmonicError::Boundary(e.to_string()))
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ReducedPressureLayout {
    pub reduction: std::sync::Arc<crate::constraints::AffineConstraintReduction>,
    pub full_node_count: usize,
}
impl ReducedPressureLayout {
    pub fn try_new(
        reduction: crate::constraints::AffineConstraintReduction,
        full_node_count: usize,
    ) -> Result<Self, AcousticHarmonicError> {
        if full_node_count.checked_mul(2) != Some(reduction.full_dof_count)
            || reduction.offset.len() != reduction.full_dof_count
            || reduction.recovery_row_offsets.len() != reduction.full_dof_count + 1
            || reduction.recovery_coefficients.len() != reduction.recovery_column_indices.len()
            || reduction.recovery_row_offsets.first() != Some(&0)
            || reduction.recovery_row_offsets.last() != Some(&reduction.recovery_coefficients.len())
            || reduction
                .recovery_row_offsets
                .windows(2)
                .any(|x| x[0] > x[1])
            || reduction
                .recovery_column_indices
                .iter()
                .any(|x| *x >= reduction.reduced_dof_count)
            || reduction
                .recovery_coefficients
                .iter()
                .chain(&reduction.offset)
                .any(|x| !x.is_finite())
        {
            return Err(AcousticHarmonicError::Operator(
                "invalid pressure phase basis or lifting".into(),
            ));
        }
        Ok(Self {
            reduction: std::sync::Arc::new(reduction),
            full_node_count,
        })
    }
}
impl runmat_types::physical::QuantityLayout for ReducedPressureLayout {
    fn component_count(&self) -> usize {
        self.reduction.reduced_dof_count
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        (index < self.reduction.reduced_dof_count).then(|| {
            DynamicDimension::try_new(
                QuantityKind::Pressure.dimension(),
                Some(QuantityKind::Pressure),
            )
            .expect("static pressure")
        })
    }
}
pub(super) type ReducedPressureOperator = QuantityLinearOperator<
    ReducedPressureLayout,
    UniformQuantityLayout<DynamicDimension>,
    RealSparseMatrix,
>;
