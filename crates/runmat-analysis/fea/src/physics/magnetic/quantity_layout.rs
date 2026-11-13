//! Coordinates of the actual scaled weak Coulomb-gauge equation.
//! Edge circulations are Wb. A gauge coordinate is the physical multiplier
//! (A/m) multiplied by mu_ref * length_ref^2, also Wb. The two roles remain
//! explicit even though their normalized dimensions agree. Multiplying the
//! edge equation by mu_ref and the gauge equation by length_ref^-2 gives
//! Wb/m rows; it does not make the saddle matrix dimensionless.
use crate::algebra::RealSparseMatrix;
use runmat_types::physical::{
    kind, DynamicDimension, Length, MagneticPermeability, Quantity, QuantityLayout,
    QuantityLinearOperator, UniformQuantityLayout,
};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MagneticUnknownLayout {
    pub edge_count: usize,
    pub gauge_count: usize,
    pub permeability: MagneticPermeability,
    pub length: Length,
}
impl MagneticUnknownLayout {
    pub fn try_new(
        edge_count: usize,
        gauge_count: usize,
        permeability: MagneticPermeability,
        length: Length,
    ) -> Result<Self, &'static str> {
        if edge_count.checked_add(gauge_count).is_none()
            || !permeability.canonical_value().is_finite()
            || permeability.canonical_value() <= 0.0
            || !length.canonical_value().is_finite()
            || length.canonical_value() <= 0.0
            || !(permeability.canonical_value() * length.canonical_value().powi(2)).is_finite()
            || permeability.canonical_value() * length.canonical_value().powi(2) <= 0.0
        {
            return Err("invalid magnetic gauge coordinate normalization");
        }
        Ok(Self {
            edge_count,
            gauge_count,
            permeability,
            length,
        })
    }
    /// Checked inverse of the explicitly retained gauge normalization.
    pub fn physical_gauge(
        &self,
        value: Quantity<kind::MagneticFlux>,
    ) -> Result<Quantity<kind::MagneticFieldIntensity>, runmat_types::physical::QuantityValueError>
    {
        Quantity::try_new(
            value.canonical_value()
                / self.permeability.canonical_value()
                / self.length.canonical_value().powi(2),
        )
    }
}
impl QuantityLayout for MagneticUnknownLayout {
    fn component_count(&self) -> usize {
        self.edge_count + self.gauge_count
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        (index < self.component_count()).then(|| {
            DynamicDimension::try_new(
                runmat_types::physical::QuantityKind::MagneticFlux.dimension(),
                Some(runmat_types::physical::QuantityKind::MagneticFlux),
            )
            .expect("static magnetic flux descriptor")
        })
    }
}
pub(super) type MagneticSaddleOperator = QuantityLinearOperator<
    MagneticUnknownLayout,
    UniformQuantityLayout<kind::MagneticVectorPotential>,
    RealSparseMatrix,
>;
pub(super) type MagneticPhysicalOperator = QuantityLinearOperator<
    UniformQuantityLayout<kind::MagneticFlux>,
    UniformQuantityLayout<kind::ElectricCurrent>,
    RealSparseMatrix,
>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gauge_scaling_keeps_physical_multiplier_and_roles() {
        let layout =
            MagneticUnknownLayout::try_new(4, 2, MagneticPermeability::new(2.0), Length::new(0.01))
                .unwrap();
        assert_eq!(layout.component_count(), 6);
        assert!(
            (layout
                .physical_gauge(Quantity::new(0.0006))
                .unwrap()
                .canonical_value()
                - 3.0)
                .abs()
                < 1e-14
        );
        assert_eq!(
            layout.component_descriptor(5).unwrap().kind(),
            Some(runmat_types::physical::QuantityKind::MagneticFlux)
        );
        assert!(layout.component_descriptor(6).is_none());
        assert!(MagneticUnknownLayout::try_new(
            4,
            2,
            MagneticPermeability::new(1.0),
            Length::new(0.0)
        )
        .is_err());
    }
}
