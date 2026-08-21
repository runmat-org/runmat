use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolidMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
}

pub(super) type ElasticityMatrix = [[f64; 6]; 6];

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SolidMaterialError {
    #[error("solid material Young's modulus must be positive and finite")]
    InvalidYoungsModulus,
    #[error("solid material Poisson ratio must be finite and in (-1, 0.5)")]
    InvalidPoissonRatio,
}

impl SolidMaterial {
    pub fn validate(self) -> Result<(), SolidMaterialError> {
        if !self.youngs_modulus_pa.is_finite() || self.youngs_modulus_pa <= 0.0 {
            return Err(SolidMaterialError::InvalidYoungsModulus);
        }
        if !self.poisson_ratio.is_finite()
            || self.poisson_ratio <= -1.0
            || self.poisson_ratio >= 0.5
        {
            return Err(SolidMaterialError::InvalidPoissonRatio);
        }
        Ok(())
    }

    pub fn lame_lambda_pa(self) -> Result<f64, SolidMaterialError> {
        self.validate()?;
        Ok(self.youngs_modulus_pa * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio)))
    }

    pub fn shear_modulus_pa(self) -> Result<f64, SolidMaterialError> {
        self.validate()?;
        Ok(self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio)))
    }
}

pub(super) fn elasticity_matrix(
    material: SolidMaterial,
) -> Result<ElasticityMatrix, SolidMaterialError> {
    let lambda = material.lame_lambda_pa()?;
    let mu = material.shear_modulus_pa()?;
    let mut matrix = [[0.0; 6]; 6];
    for (row, diagonal) in matrix.iter_mut().enumerate().take(3) {
        diagonal[..3].fill(lambda);
        diagonal[row] += 2.0 * mu;
    }
    matrix[3][3] = mu;
    matrix[4][4] = mu;
    matrix[5][5] = mu;
    Ok(matrix)
}
