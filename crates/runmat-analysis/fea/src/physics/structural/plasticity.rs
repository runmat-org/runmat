//! Pure small-strain isotropic elasticity and associative J2 radial return.
//!
//! Engineering strain order is xx, yy, zz, yz, xz, xy. Stress uses physical shear;
//! tangent entries are derivatives with respect to engineering shear strain. Committed
//! history is never mutated by a trial update. No global convergence/commit is implied.

mod return_map;
mod state;
#[cfg(test)]
mod tests;

use runmat_analysis_core::{MaterialModel, MaterialPlasticModel};
pub use state::{PlasticityPointState, PlasticityResponse};

#[derive(Debug, Clone, PartialEq)]
pub enum PlasticityError {
    MissingIsotropicElasticity,
    InvalidElasticity,
    InvalidPlasticLaw,
    InvalidState(&'static str),
    PlasticHistoryWithoutLaw,
    NonFiniteStrain,
    NonFiniteResponse,
}

impl std::fmt::Display for PlasticityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "invalid small-strain isotropic constitutive update: {self:?}"
        )
    }
}

impl std::error::Error for PlasticityError {}

#[derive(Debug, Clone, Copy)]
struct LinearHardening {
    yield_stress_pa: runmat_types::physical::YieldStress,
    modulus_pa: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct IsotropicConstitutiveModel {
    bulk_modulus_pa: f64,
    shear_modulus_pa: f64,
    plastic: Option<LinearHardening>,
}

impl IsotropicConstitutiveModel {
    pub fn from_material(material: &MaterialModel) -> Result<Self, PlasticityError> {
        let elasticity = material
            .require_mechanical()
            .map_err(|_| PlasticityError::MissingIsotropicElasticity)?;
        let youngs = elasticity.youngs_modulus_pa.canonical_value();
        let poisson = elasticity.poisson_ratio.canonical_value();
        if !youngs.is_finite()
            || youngs <= 0.0
            || !poisson.is_finite()
            || poisson <= -1.0
            || poisson >= 0.5
        {
            return Err(PlasticityError::InvalidElasticity);
        }
        let bulk_modulus_pa = youngs / (3.0 * (1.0 - 2.0 * poisson));
        let shear_modulus_pa = youngs / (2.0 * (1.0 + poisson));
        if !bulk_modulus_pa.is_finite()
            || !shear_modulus_pa.is_finite()
            || bulk_modulus_pa <= 0.0
            || shear_modulus_pa <= 0.0
        {
            return Err(PlasticityError::InvalidElasticity);
        }
        let plastic = match &material.plastic {
            None => None,
            Some(MaterialPlasticModel::J2IsotropicLinearHardening {
                yield_stress_pa,
                isotropic_hardening_modulus_pa,
            }) => {
                let yield_stress = yield_stress_pa.canonical_value();
                let hardening_modulus = isotropic_hardening_modulus_pa.canonical_value();
                if !yield_stress.is_finite()
                    || yield_stress <= 0.0
                    || !hardening_modulus.is_finite()
                    || hardening_modulus < 0.0
                    || !(3.0 * shear_modulus_pa + hardening_modulus).is_finite()
                {
                    return Err(PlasticityError::InvalidPlasticLaw);
                }
                Some(LinearHardening {
                    yield_stress_pa: *yield_stress_pa,
                    modulus_pa: hardening_modulus,
                })
            }
        };
        Ok(Self {
            bulk_modulus_pa,
            shear_modulus_pa,
            plastic,
        })
    }

    pub fn elastic_tangent_pa(&self) -> [[f64; 6]; 6] {
        return_map::elastic_tangent(self.bulk_modulus_pa, self.shear_modulus_pa)
    }

    pub fn update(
        &self,
        total_engineering_strain: [f64; 6],
        committed: &PlasticityPointState,
    ) -> Result<PlasticityResponse, PlasticityError> {
        self.validate_committed(committed)?;
        if total_engineering_strain
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PlasticityError::NonFiniteStrain);
        }
        return_map::update(self, total_engineering_strain, committed)
    }

    /// Exact accumulated irreversible work for this fixed-yield J2 law, per physical volume.
    /// Hardening energy is stored separately and must not be counted as dissipation.
    pub fn committed_plastic_dissipation_density_pa(
        &self,
        committed: &PlasticityPointState,
    ) -> Result<f64, PlasticityError> {
        self.validate_committed(committed)?;
        let value = self.plastic.map_or(0.0, |law| {
            law.yield_stress_pa.canonical_value() * committed.equivalent_plastic_strain
        });
        if !value.is_finite() {
            return Err(PlasticityError::NonFiniteResponse);
        }
        Ok(value)
    }

    /// Constitutive replay roundoff in equivalent-plastic-strain units, not an accuracy gate.
    /// Bounds subtraction/mean projection and the six-component invariant, yield evaluation,
    /// return-map division, and accumulated-alpha addition with a conservative 128-epsilon
    /// operation envelope. The scale is actual stress and strain history, never a fixed alpha floor.
    pub fn replay_increment_roundoff_bound(
        &self,
        committed: &PlasticityPointState,
        total_engineering_strain: [f64; 6],
    ) -> Result<f64, PlasticityError> {
        self.validate_committed(committed)?;
        if total_engineering_strain
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PlasticityError::NonFiniteStrain);
        }
        let Some(law) = self.plastic else {
            return Ok(0.0);
        };
        let trial = return_map::trial_stress(self, total_engineering_strain, committed)?;
        let yield_stress = law.yield_stress_pa.canonical_value()
            + law.modulus_pa * committed.equivalent_plastic_strain;
        // Cancellation in total - plastic must be scaled by both operands, not just their
        // small difference. Engineering shear has half its engineering magnitude in tensor space.
        let tensor_norm = |strain: &[f64; 6]| {
            strain.iter().enumerate().fold(0.0_f64, |norm, (i, value)| {
                norm.hypot(
                    value
                        * if i < 3 {
                            1.0
                        } else {
                            std::f64::consts::FRAC_1_SQRT_2
                        },
                )
            })
        };
        let strain_scale = tensor_norm(&total_engineering_strain)
            + tensor_norm(&committed.engineering_plastic_strain);
        let cancellation_stress = 2.0 * self.shear_modulus_pa * strain_scale;
        let denominator = 3.0 * self.shear_modulus_pa + law.modulus_pa;
        let bound = 128.0
            * f64::EPSILON
            * ((trial.equivalent + yield_stress + cancellation_stress) / denominator
                + committed.equivalent_plastic_strain);
        if !yield_stress.is_finite()
            || !cancellation_stress.is_finite()
            || !bound.is_finite()
            || bound < 0.0
        {
            return Err(PlasticityError::NonFiniteResponse);
        }
        Ok(bound)
    }

    fn validate_committed(&self, committed: &PlasticityPointState) -> Result<(), PlasticityError> {
        committed.validate()?;
        if self.plastic.is_none()
            && (committed.equivalent_plastic_strain != 0.0
                || committed
                    .engineering_plastic_strain
                    .iter()
                    .any(|value| *value != 0.0))
        {
            return Err(PlasticityError::PlasticHistoryWithoutLaw);
        }
        Ok(())
    }
}
