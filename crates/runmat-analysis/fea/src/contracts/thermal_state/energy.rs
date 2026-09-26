//! Signed global heat-equation power balance retained for each solve.
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalEnergyBalance {
    pub storage_rate_w: f64,
    pub boundary_loss_w: f64,
    pub applied_power_w: f64,
    /// Signed correction from prescribed-temperature reactions in the residual equation.
    pub prescribed_boundary_power_w: f64,
    pub residual_w: f64,
    /// Absolute residual divided by the largest absolute power term, with a one-watt floor.
    pub relative_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ThermalEnergyBalanceError {
    #[error("thermal energy balance contains nonfinite powers or residuals")]
    NonFinite,
    #[error("thermal energy balance residuals do not match its signed power terms")]
    Inconsistent,
}

impl ThermalEnergyBalance {
    pub fn from_powers(
        storage_rate_w: f64,
        boundary_loss_w: f64,
        applied_power_w: f64,
        prescribed_boundary_power_w: f64,
    ) -> Self {
        let residual_w =
            storage_rate_w + boundary_loss_w - applied_power_w + prescribed_boundary_power_w;
        let scale = storage_rate_w
            .abs()
            .max(boundary_loss_w.abs())
            .max(applied_power_w.abs())
            .max(prescribed_boundary_power_w.abs())
            .max(1.0);
        Self {
            storage_rate_w,
            boundary_loss_w,
            applied_power_w,
            prescribed_boundary_power_w,
            residual_w,
            relative_residual: residual_w.abs() / scale,
        }
    }

    /// Validate the recorded arithmetic, not conservation against independent physics data.
    pub fn validate(&self) -> Result<(), ThermalEnergyBalanceError> {
        if [
            self.storage_rate_w,
            self.boundary_loss_w,
            self.applied_power_w,
            self.prescribed_boundary_power_w,
            self.residual_w,
            self.relative_residual,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(ThermalEnergyBalanceError::NonFinite);
        }
        let expected = Self::from_powers(
            self.storage_rate_w,
            self.boundary_loss_w,
            self.applied_power_w,
            self.prescribed_boundary_power_w,
        );
        if !expected.residual_w.is_finite() || !expected.relative_residual.is_finite() {
            return Err(ThermalEnergyBalanceError::NonFinite);
        }
        if self.residual_w != expected.residual_w
            || self.relative_residual != expected.relative_residual
        {
            return Err(ThermalEnergyBalanceError::Inconsistent);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
