use serde::{Deserialize, Serialize};

use super::PlasticityError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PlasticityPointState {
    pub engineering_plastic_strain: [f64; 6],
    pub equivalent_plastic_strain: f64,
}

impl PlasticityPointState {
    pub fn validate(&self) -> Result<(), PlasticityError> {
        let alpha = self.equivalent_plastic_strain;
        let strain = self.engineering_plastic_strain;
        if !alpha.is_finite() || alpha < 0.0 || strain.iter().any(|value| !value.is_finite()) {
            return Err(PlasticityError::InvalidState(
                "nonfinite or negative plastic history",
            ));
        }
        let scale = strain
            .iter()
            .fold(alpha, |scale, value| scale.max(value.abs()));
        let roundoff = 128.0 * f64::EPSILON * scale;
        if (strain[0] + strain[1] + strain[2]).abs() > roundoff {
            return Err(PlasticityError::InvalidState(
                "plastic strain is not trace free",
            ));
        }
        // Accumulated equivalent plastic strain is at least the norm of the net plastic
        // strain. Engineering shear gamma contributes gamma^2/2 to the tensor norm.
        let mut norm = 0.0_f64;
        for (index, value) in strain.iter().enumerate() {
            let weighted = value
                * if index < 3 {
                    (2.0_f64 / 3.0).sqrt()
                } else {
                    (1.0_f64 / 3.0).sqrt()
                };
            norm = norm.hypot(weighted);
        }
        if !norm.is_finite() || norm > alpha + roundoff {
            return Err(PlasticityError::InvalidState(
                "plastic strain exceeds accumulated equivalent strain",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlasticityResponse {
    pub stress_pa: [f64; 6],
    pub tangent_pa: [[f64; 6]; 6],
    pub state: PlasticityPointState,
    /// Recoverable elastic energy per physical volume, in J/m^3 = Pa.
    pub elastic_energy_density_pa: f64,
    /// Stored isotropic-hardening energy H*alpha^2/2, not dissipation.
    pub hardening_energy_density_pa: f64,
    /// Exact constitutive irreversible work sigma_y0*delta_alpha for this trial increment.
    /// This is not a global time-integration or external-work balance assertion.
    pub plastic_dissipation_increment_pa: f64,
}
