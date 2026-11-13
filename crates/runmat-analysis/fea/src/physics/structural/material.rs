use runmat_analysis_core::{MechanicalDampingModel, MissingMaterialCapability};

/// Mechanical coefficients projected from one compiled material-bearing element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralElementMaterial {
    pub youngs_modulus_pa: runmat_types::physical::YoungsModulus,
    pub poisson_ratio: runmat_types::physical::PoissonRatio,
    pub density_kg_per_m3: runmat_types::physical::MassDensity,
    pub damping: MechanicalDampingModel,
}

impl StructuralElementMaterial {
    pub(crate) fn from_model(
        material: &runmat_analysis_core::MaterialModel,
    ) -> Result<Self, MissingMaterialCapability> {
        let mechanical = material.require_mechanical()?;
        Ok(Self {
            youngs_modulus_pa: mechanical.youngs_modulus_pa,
            poisson_ratio: runmat_types::physical::PoissonRatio::new(
                mechanical.poisson_ratio.canonical_value(),
            ),
            density_kg_per_m3: material.require_mass_density()?,
            damping: material.damping_or_undamped(),
        })
    }

    pub fn shear_modulus_pa(self) -> runmat_types::physical::ShearModulus {
        runmat_types::physical::ShearModulus::new(
            self.youngs_modulus_pa.canonical_value()
                / (2.0 * (1.0 + self.poisson_ratio.canonical_value())),
        )
    }
}
