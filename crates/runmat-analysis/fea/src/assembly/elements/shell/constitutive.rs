use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};
use serde::{Deserialize, Serialize};

use super::{ShellElementError, ShellMaterial, ShellSection};

const FORMULATION_DRILLING_SCALE: f64 = 1.0e-4;

pub type ShellMatrix3 = [[f64; 3]; 3];
pub type ShellMatrix2 = [[f64; 2]; 2];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellConstitutiveMatrices {
    pub extensional: ShellMatrix3,
    pub coupling: ShellMatrix3,
    pub bending: ShellMatrix3,
    pub transverse_shear: ShellMatrix2,
    pub drilling_stiffness_per_area: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellRecoveryLayer {
    pub layer_id: String,
    pub bottom_m: f64,
    pub top_m: f64,
    pub plane_stress: ShellMatrix3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellConstitutive {
    pub matrices: ShellConstitutiveMatrices,
    pub recovery_layers: Vec<ShellRecoveryLayer>,
}

impl ShellConstitutiveMatrices {
    pub fn zero() -> Self {
        Self {
            extensional: [[0.0; 3]; 3],
            coupling: [[0.0; 3]; 3],
            bending: [[0.0; 3]; 3],
            transverse_shear: [[0.0; 2]; 2],
            drilling_stiffness_per_area: 0.0,
        }
    }

    pub(super) fn validate_elastic(&self) -> Result<(), ShellElementError> {
        if !matrix_is_finite_and_symmetric(&self.extensional)
            || !matrix_is_finite_and_symmetric(&self.coupling)
            || !matrix_is_finite_and_symmetric(&self.bending)
            || !matrix_is_finite_and_symmetric(&self.transverse_shear)
            || !positive_finite(self.drilling_stiffness_per_area)
            || !positive_definite_3(&self.extensional)
            || !positive_definite_3(&self.bending)
            || !positive_definite_2(&self.transverse_shear)
        {
            return Err(ShellElementError::InvalidConstitutive);
        }
        Ok(())
    }

    pub(super) fn validate_damping(&self) -> Result<(), ShellElementError> {
        if matrix_is_finite_and_symmetric(&self.extensional)
            && matrix_is_finite_and_symmetric(&self.coupling)
            && matrix_is_finite_and_symmetric(&self.bending)
            && matrix_is_finite_and_symmetric(&self.transverse_shear)
            && self.drilling_stiffness_per_area.is_finite()
            && self.drilling_stiffness_per_area >= 0.0
        {
            Ok(())
        } else {
            Err(ShellElementError::InvalidConstitutive)
        }
    }
}

impl ShellConstitutive {
    pub(super) fn validate(&self) -> Result<(), ShellElementError> {
        self.matrices.validate_elastic()?;
        if self.recovery_layers.is_empty()
            || self.recovery_layers.iter().any(|layer| {
                layer.layer_id.trim().is_empty()
                    || !layer.bottom_m.is_finite()
                    || !layer.top_m.is_finite()
                    || layer.top_m <= layer.bottom_m
                    || !matrix_is_finite_and_symmetric(&layer.plane_stress)
                    || !positive_definite_3(&layer.plane_stress)
            })
        {
            return Err(ShellElementError::InvalidConstitutive);
        }
        Ok(())
    }
}

fn homogeneous_constitutive(section: ShellSection, material: ShellMaterial) -> ShellConstitutive {
    let plane_stress = isotropic_plane_stress(
        material.youngs_modulus_pa.canonical_value(),
        material.poisson_ratio.canonical_value(),
    );
    let thickness = section.thickness_m.canonical_value();
    let bottom = -0.5 * thickness - section.reference_offset_m.canonical_value();
    let top = 0.5 * thickness - section.reference_offset_m.canonical_value();
    ShellConstitutive {
        matrices: ShellConstitutiveMatrices {
            extensional: scale3(plane_stress, thickness),
            coupling: scale3(plane_stress, 0.5 * (top.powi(2) - bottom.powi(2))),
            bending: scale3(plane_stress, (top.powi(3) - bottom.powi(3)) / 3.0),
            transverse_shear: [
                [
                    material.shear_modulus_pa.canonical_value()
                        * thickness
                        * section.shear_correction,
                    0.0,
                ],
                [
                    0.0,
                    material.shear_modulus_pa.canonical_value()
                        * thickness
                        * section.shear_correction,
                ],
            ],
            drilling_stiffness_per_area: FORMULATION_DRILLING_SCALE
                * material.youngs_modulus_pa.canonical_value()
                * thickness,
        },
        recovery_layers: vec![ShellRecoveryLayer {
            layer_id: "homogeneous".into(),
            bottom_m: bottom,
            top_m: top,
            plane_stress,
        }],
    }
}

/// Fallible execution boundary for production homogeneous constitutive work.
pub(crate) fn admitted_homogeneous_constitutive(
    section: ShellSection,
    material: ShellMaterial,
) -> Result<ShellConstitutive, ShellElementError> {
    section.validate()?;
    material.validate()?;
    // Plane stress (8), top/bottom (4), three 3x3 scales (27), coupling
    // factor (4), bending factor (6), transverse shear (4), drilling (2).
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(55))
        })
        .transpose()?;
    let constitutive = homogeneous_constitutive(section, material);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(55))?;
    }
    Ok(constitutive)
}

pub(super) fn isotropic_plane_stress(youngs_modulus_pa: f64, poisson_ratio: f64) -> ShellMatrix3 {
    let factor = youngs_modulus_pa / (1.0 - poisson_ratio.powi(2));
    [
        [factor, factor * poisson_ratio, 0.0],
        [factor * poisson_ratio, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - poisson_ratio) / 2.0],
    ]
}

pub(super) fn scale3(mut matrix: ShellMatrix3, factor: f64) -> ShellMatrix3 {
    for row in &mut matrix {
        for value in row {
            *value *= factor;
        }
    }
    matrix
}

fn matrix_is_finite_and_symmetric<const N: usize>(matrix: &[[f64; N]; N]) -> bool {
    (0..N).all(|row| {
        (0..N).all(|column| {
            matrix[row][column].is_finite()
                && approximately_equal(matrix[row][column], matrix[column][row])
        })
    })
}

fn positive_definite_2(matrix: &ShellMatrix2) -> bool {
    matrix[0][0] > 0.0 && determinant_2(matrix) > 0.0
}

fn positive_definite_3(matrix: &ShellMatrix3) -> bool {
    matrix[0][0] > 0.0
        && matrix[0][0] * matrix[1][1] - matrix[0][1].powi(2) > 0.0
        && determinant_3(matrix) > 0.0
}

fn determinant_2(matrix: &ShellMatrix2) -> f64 {
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
}

fn determinant_3(matrix: &ShellMatrix3) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-12 * left.abs().max(right.abs()).max(1.0)
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
