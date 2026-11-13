//! Explicit modal coordinate convention. Numerical mass normalization uses a
//! one-kilogram reference; generalized coordinates retain length dimensions.
use crate::assembly::dofs::StructuralDofLayout;
use runmat_types::physical::{DynamicDimension, Mass, QuantityField, QuantityKind, QuantityLayout};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalBasisCoordinates {
    full_dof_layout: StructuralDofLayout,
    reference_generalized_mass: Mass,
}
impl ModalBasisCoordinates {
    pub fn for_layout(full_dof_layout: StructuralDofLayout) -> Self {
        Self {
            full_dof_layout,
            reference_generalized_mass: Self::canonical_reference_mass(),
        }
    }
    pub const fn canonical_reference_mass() -> Mass {
        Mass::new(1.0)
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        self.full_dof_layout.validate()?;
        if self.reference_generalized_mass.canonical_value() != 1.0 {
            return Err("modal coordinates require the canonical one-kilogram mass reference");
        }
        Ok(())
    }
    pub fn full_dof_layout(&self) -> &StructuralDofLayout {
        &self.full_dof_layout
    }
    pub fn generalized_coordinate(&self) -> DynamicDimension {
        DynamicDimension::quantity_type(QuantityKind::Displacement)
            .with_semantic_kind(Some(QuantityKind::Displacement))
            .expect("static modal coordinate kind")
    }
    pub fn reference_generalized_mass(&self) -> Mass {
        self.reference_generalized_mass
    }
    pub fn shape_layout(&self) -> ModalShapeLayout<'_> {
        ModalShapeLayout { coordinates: self }
    }
    pub fn shape_field<'a>(
        &'a self,
        values: &'a [f64],
    ) -> Result<QuantityField<ModalShapeLayout<'a>, &'a [f64]>, &'static str> {
        self.validate()?;
        QuantityField::try_new(self.shape_layout(), values)
            .map_err(|_| "modal transform does not match its retained full coordinate layout")
    }
}
#[derive(Clone, Copy)]
pub struct ModalShapeLayout<'a> {
    coordinates: &'a ModalBasisCoordinates,
}
impl QuantityLayout for ModalShapeLayout<'_> {
    fn component_count(&self) -> usize {
        self.coordinates.full_dof_layout.total_dof_count()
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        let output = self
            .coordinates
            .full_dof_layout
            .row_kind(index)?
            .state_quantity();
        DynamicDimension::quantity_type(output).divide(self.coordinates.generalized_coordinate())
    }
}

pub(super) mod shared_wire {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &std::sync::Arc<ModalBasisCoordinates>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_ref().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<std::sync::Arc<ModalBasisCoordinates>, D::Error> {
        let value = ModalBasisCoordinates::deserialize(deserializer)?;
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(std::sync::Arc::new(value))
    }
}
