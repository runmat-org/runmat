//! Signed global heat-equation power balance retained for each solve.
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Absolute physical transfer sums, retained before global signed cancellation.
/// These are powers, never absolute-temperature energy or an operator-norm floor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalPowerScale {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub storage_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub boundary_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub applied_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub prescribed_w: runmat_types::physical::Power,
}

impl ThermalPowerScale {
    pub fn maximum(self) -> runmat_types::physical::Power {
        self.storage_w
            .max(self.boundary_w)
            .max(self.applied_w)
            .max(self.prescribed_w)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalEnergyBalance {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub storage_rate_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub boundary_loss_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub applied_power_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    /// Signed correction from prescribed-temperature reactions in the residual equation.
    pub prescribed_boundary_power_w: runmat_types::physical::Power,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub residual_w: runmat_types::physical::Power,
    pub scale: ThermalPowerScale,
    /// Absolute residual divided by the largest retained physical transfer sum.
    /// Exactly zero power has zero residual; there is no dimensional tolerance floor.
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
        storage_rate_w: runmat_types::physical::Power,
        boundary_loss_w: runmat_types::physical::Power,
        applied_power_w: runmat_types::physical::Power,
        prescribed_boundary_power_w: runmat_types::physical::Power,
    ) -> Self {
        Self::from_power_transfers(
            storage_rate_w,
            boundary_loss_w,
            applied_power_w,
            prescribed_boundary_power_w,
            ThermalPowerScale {
                storage_w: storage_rate_w.abs(),
                boundary_w: boundary_loss_w.abs(),
                applied_w: applied_power_w.abs(),
                prescribed_w: prescribed_boundary_power_w.abs(),
            },
        )
    }

    pub fn from_power_transfers(
        storage_rate_w: runmat_types::physical::Power,
        boundary_loss_w: runmat_types::physical::Power,
        applied_power_w: runmat_types::physical::Power,
        prescribed_boundary_power_w: runmat_types::physical::Power,
        scale: ThermalPowerScale,
    ) -> Self {
        let residual_w =
            storage_rate_w + boundary_loss_w - applied_power_w + prescribed_boundary_power_w;
        let normalization = scale.maximum();
        Self {
            storage_rate_w,
            boundary_loss_w,
            applied_power_w,
            prescribed_boundary_power_w,
            residual_w,
            scale,
            relative_residual: if normalization.canonical_value() == 0.0 {
                0.0
            } else {
                residual_w.abs().ratio_to(normalization).canonical_value()
            },
        }
    }

    /// Validate retained signed-power arithmetic, without a dimensional floor.
    pub fn validate(&self) -> Result<(), ThermalEnergyBalanceError> {
        if [
            self.storage_rate_w,
            self.boundary_loss_w,
            self.applied_power_w,
            self.prescribed_boundary_power_w,
            self.residual_w,
            self.scale.storage_w,
            self.scale.boundary_w,
            self.scale.applied_w,
            self.scale.prescribed_w,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || !self.relative_residual.is_finite()
        {
            return Err(ThermalEnergyBalanceError::NonFinite);
        }
        if self.scale.storage_w < self.storage_rate_w.abs()
            || self.scale.boundary_w < self.boundary_loss_w.abs()
            || self.scale.applied_w < self.applied_power_w.abs()
            || self.scale.prescribed_w < self.prescribed_boundary_power_w.abs()
        {
            return Err(ThermalEnergyBalanceError::Inconsistent);
        }
        let expected = Self::from_power_transfers(
            self.storage_rate_w,
            self.boundary_loss_w,
            self.applied_power_w,
            self.prescribed_boundary_power_w,
            self.scale,
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
