use runmat_analysis_core::{MechanicalDampingModel, MissingMaterialCapability};

/// Mechanical coefficients projected from one compiled material-bearing element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralElementMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub density_kg_per_m3: f64,
    pub damping: MechanicalDampingModel,
}

impl StructuralElementMaterial {
    pub(crate) fn from_model(
        material: &runmat_analysis_core::MaterialModel,
    ) -> Result<Self, MissingMaterialCapability> {
        let mechanical = material.require_mechanical()?;
        Ok(Self {
            youngs_modulus_pa: mechanical.youngs_modulus_pa,
            poisson_ratio: mechanical.poisson_ratio,
            density_kg_per_m3: material.require_mass_density()?,
            damping: material.damping_or_undamped(),
        })
    }

    pub fn shear_modulus_pa(self) -> f64 {
        self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio))
    }
}
