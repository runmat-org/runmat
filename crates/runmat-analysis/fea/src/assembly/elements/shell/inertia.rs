use super::{
    transform_matrix_to_global, ShellElementError, ShellLocalFrame, ShellMatrix18, ShellSection,
    SHELL_ELEMENT_DOF_COUNT, SHELL_ELEMENT_NODE_COUNT, SHELL_NODE_DOF_COUNT,
};

/// Through-thickness mass moments per unit reference-surface area.
///
/// The moments are evaluated about the shell reference surface. The first moment therefore
/// preserves translation/rotation coupling when that surface is not the mass centroid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellMassMoments {
    pub zeroth_kg_per_m2: f64,
    pub first_kg_per_m: f64,
    pub second_kg: f64,
}

impl ShellMassMoments {
    pub const ZERO: Self = Self {
        zeroth_kg_per_m2: 0.0,
        first_kg_per_m: 0.0,
        second_kg: 0.0,
    };

    pub fn validate(self, allow_zero: bool) -> Result<(), ShellElementError> {
        let finite = self.zeroth_kg_per_m2.is_finite()
            && self.first_kg_per_m.is_finite()
            && self.second_kg.is_finite();
        let nonnegative = self.zeroth_kg_per_m2 >= 0.0 && self.second_kg >= 0.0;
        let positive = allow_zero || (self.zeroth_kg_per_m2 > 0.0 && self.second_kg > 0.0);
        let determinant = self.zeroth_kg_per_m2 * self.second_kg - self.first_kg_per_m.powi(2);
        let tolerance = 64.0
            * f64::EPSILON
            * (self.zeroth_kg_per_m2 * self.second_kg)
                .abs()
                .max(self.first_kg_per_m.powi(2));
        if finite && nonnegative && positive && determinant >= -tolerance {
            Ok(())
        } else {
            Err(ShellElementError::InvalidConstitutive)
        }
    }
}

pub fn homogeneous_mass_moments(
    section: ShellSection,
    density_kg_per_m3: f64,
) -> Result<ShellMassMoments, ShellElementError> {
    section.validate()?;
    if !density_kg_per_m3.is_finite() || density_kg_per_m3 <= 0.0 {
        return Err(ShellElementError::InvalidConstitutive);
    }
    let bottom = -0.5 * section.thickness_m - section.reference_offset_m;
    let top = 0.5 * section.thickness_m - section.reference_offset_m;
    let moments = ShellMassMoments {
        zeroth_kg_per_m2: density_kg_per_m3 * (top - bottom),
        first_kg_per_m: density_kg_per_m3 * 0.5 * (top.powi(2) - bottom.powi(2)),
        second_kg: density_kg_per_m3 * (top.powi(3) - bottom.powi(3)) / 3.0,
    };
    moments.validate(false)?;
    Ok(moments)
}

/// Builds a nodally lumped shell mass operator while retaining the exact thickness moments.
///
/// The two physical director rotations use the second moment. The drilling rotation receives the
/// same positive rotary term as a numerical inertia, matching the formulation's drilling
/// stabilization and preventing an otherwise massless explicit degree of freedom.
pub fn global_lumped_mass_matrix(
    moments: ShellMassMoments,
    frame: ShellLocalFrame,
) -> Result<ShellMatrix18, ShellElementError> {
    moments.validate(true)?;
    let scale = frame.area_m2 / SHELL_ELEMENT_NODE_COUNT as f64;
    let mut local = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];
    for node in 0..SHELL_ELEMENT_NODE_COUNT {
        let base = node * SHELL_NODE_DOF_COUNT;
        for component in 0..3 {
            local[base + component][base + component] = scale * moments.zeroth_kg_per_m2;
        }
        for component in 3..6 {
            local[base + component][base + component] = scale * moments.second_kg;
        }
        let first = scale * moments.first_kg_per_m;
        local[base][base + 4] = first;
        local[base + 4][base] = first;
        local[base + 1][base + 3] = -first;
        local[base + 3][base + 1] = -first;
    }
    Ok(transform_matrix_to_global(local, frame))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_homogeneous_section_retains_first_mass_moment() {
        let moments = homogeneous_mass_moments(
            ShellSection {
                thickness_m: 0.2,
                reference_offset_m: 0.1,
                shear_correction: 5.0 / 6.0,
            },
            10.0,
        )
        .unwrap();
        assert!((moments.zeroth_kg_per_m2 - 2.0).abs() < 1.0e-12);
        assert!((moments.first_kg_per_m + 0.2).abs() < 1.0e-12);
        assert!((moments.second_kg - 0.02666666666666667).abs() < 1.0e-12);
    }

    #[test]
    fn shell_mass_coupling_is_symmetric_and_tracks_kinematic_signs() {
        let matrix = global_lumped_mass_matrix(
            ShellMassMoments {
                zeroth_kg_per_m2: 2.0,
                first_kg_per_m: 0.25,
                second_kg: 0.1,
            },
            ShellLocalFrame {
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
                z: [0.0, 0.0, 1.0],
                area_m2: 3.0,
            },
        )
        .unwrap();
        assert_eq!(matrix[0][4], 0.25);
        assert_eq!(matrix[4][0], 0.25);
        assert_eq!(matrix[1][3], -0.25);
        assert_eq!(matrix[3][1], -0.25);
    }
}
