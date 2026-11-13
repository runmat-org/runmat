use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolidMaterial {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub youngs_modulus_pa: runmat_types::physical::YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub poisson_ratio: runmat_types::physical::PoissonRatio,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub density_kg_per_m3: runmat_types::physical::MassDensity,
}

pub(super) type ElasticityMatrix = [[f64; 6]; 6];

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SolidMaterialError {
    #[error("solid material Young's modulus must be positive and finite")]
    InvalidYoungsModulus,
    #[error("solid material Poisson ratio must be finite and in (-1, 0.5)")]
    InvalidPoissonRatio,
    #[error("solid material density must be positive and finite")]
    InvalidDensity,
    #[error("solid material numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
}

impl SolidMaterial {
    pub fn validate(self) -> Result<(), SolidMaterialError> {
        if !self.youngs_modulus_pa.canonical_value().is_finite()
            || self.youngs_modulus_pa.canonical_value() <= 0.0
        {
            return Err(SolidMaterialError::InvalidYoungsModulus);
        }
        if !self.poisson_ratio.canonical_value().is_finite()
            || self.poisson_ratio.canonical_value() <= -1.0
            || self.poisson_ratio.canonical_value() >= 0.5
        {
            return Err(SolidMaterialError::InvalidPoissonRatio);
        }
        if !self.density_kg_per_m3.canonical_value().is_finite()
            || self.density_kg_per_m3.canonical_value() <= 0.0
        {
            return Err(SolidMaterialError::InvalidDensity);
        }
        Ok(())
    }

    pub fn lame_lambda_pa(self) -> Result<f64, SolidMaterialError> {
        self.validate()?;
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(6))
            })
            .transpose()?;
        let lambda = self.youngs_modulus_pa.canonical_value()
            * self.poisson_ratio.canonical_value()
            / ((1.0 + self.poisson_ratio.canonical_value())
                * (1.0 - 2.0 * self.poisson_ratio.canonical_value()));
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(6))?;
        }
        Ok(lambda)
    }

    pub fn shear_modulus_pa(self) -> Result<f64, SolidMaterialError> {
        self.validate()?;
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(3))
            })
            .transpose()?;
        let shear = self.youngs_modulus_pa.canonical_value()
            / (2.0 * (1.0 + self.poisson_ratio.canonical_value()));
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(3))?;
        }
        Ok(shear)
    }
}

pub(super) fn elasticity_matrix(
    material: SolidMaterial,
) -> Result<ElasticityMatrix, SolidMaterialError> {
    let lambda = material.lame_lambda_pa()?;
    let mu = material.shear_modulus_pa()?;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(6))
        })
        .transpose()?;
    let mut matrix = [[0.0; 6]; 6];
    for (row, diagonal) in matrix.iter_mut().enumerate().take(3) {
        diagonal[..3].fill(lambda);
        diagonal[row] += 2.0 * mu;
    }
    matrix[3][3] = mu;
    matrix[4][4] = mu;
    matrix[5][5] = mu;
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(6))?;
    }
    Ok(matrix)
}
