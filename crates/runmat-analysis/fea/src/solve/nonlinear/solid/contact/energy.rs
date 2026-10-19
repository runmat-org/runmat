//! Accepted contact work. Only the converged trial advances these totals;
//! replay and rejected trial diagnostics never change the original history.
use super::*;
use crate::physics::structural::contact::ContactAssemblyDiagnostics;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedContactEnergy {
    last_increment: ContactAssemblyDiagnostics,
    frictional_dissipation_j: f64,
    algorithmic_dissipation_j: f64,
    separation_release_j: f64,
}
impl AcceptedContactEnergy {
    pub fn last_increment(&self) -> &ContactAssemblyDiagnostics {
        &self.last_increment
    }
    pub fn frictional_dissipation_j(&self) -> f64 {
        self.frictional_dissipation_j
    }
    /// Numerical loss in the tangential update; this is not friction heat.
    pub fn algorithmic_dissipation_j(&self) -> f64 {
        self.algorithmic_dissipation_j
    }
    /// Tangential stored energy released on opening; this is not friction heat.
    pub fn separation_release_j(&self) -> f64 {
        self.separation_release_j
    }
    pub(in crate::solve::nonlinear::solid) fn advance(
        &self,
        diagnostics: ContactAssemblyDiagnostics,
    ) -> Result<Self, SolidNewtonError> {
        let next = Self {
            last_increment: diagnostics,
            frictional_dissipation_j: self.frictional_dissipation_j
                + diagnostics.frictional_dissipation_increment_j,
            algorithmic_dissipation_j: self.algorithmic_dissipation_j
                + diagnostics.algorithmic_dissipation_increment_j,
            separation_release_j: self.separation_release_j
                + diagnostics.separation_release_energy_j,
        };
        next.validate(1)?;
        Ok(next)
    }
    pub(in crate::solve::nonlinear::solid) fn validate_current_energy(
        &self,
        current: &ContactAssemblyDiagnostics,
    ) -> Result<(), SolidNewtonError> {
        for (retained, recomputed) in [
            (
                self.last_increment.maximum_penetration_m,
                current.maximum_penetration_m,
            ),
            (
                self.last_increment.normal_elastic_energy_j,
                current.normal_elastic_energy_j,
            ),
            (
                self.last_increment.tangential_elastic_energy_j,
                current.tangential_elastic_energy_j,
            ),
        ] {
            if (retained - recomputed).abs()
                > 512. * f64::EPSILON * retained.max(recomputed).max(f64::MIN_POSITIVE)
            {
                return Err(SolidNewtonError::Invalid(
                    "contact current energy differs from original configuration".into(),
                ));
            }
        }
        Ok(())
    }
    /// Historical numerical losses cannot be independently recovered from a
    /// terminal configuration. This validates shape, not the past trajectory.
    pub(in crate::solve::nonlinear::solid) fn validate(
        &self,
        ordinal: u64,
    ) -> Result<(), SolidNewtonError> {
        let d = &self.last_increment;
        let scalars = [
            self.frictional_dissipation_j,
            self.algorithmic_dissipation_j,
            self.separation_release_j,
            d.maximum_penetration_m,
            d.normal_elastic_energy_j,
            d.tangential_elastic_energy_j,
            d.frictional_dissipation_increment_j,
            d.algorithmic_dissipation_increment_j,
            d.separation_release_energy_j,
        ];
        if scalars.iter().any(|v| !v.is_finite() || *v < 0.)
            || d.frictional_dissipation_increment_j > self.frictional_dissipation_j
            || d.algorithmic_dissipation_increment_j > self.algorithmic_dissipation_j
            || d.separation_release_energy_j > self.separation_release_j
            || d.stick_points
                .checked_add(d.slip_points)
                .is_none_or(|n| n > d.active_points)
            || (ordinal == 0 && *self != Self::default())
        {
            return Err(SolidNewtonError::Invalid(
                "malformed accepted contact energy".into(),
            ));
        }
        Ok(())
    }
}
