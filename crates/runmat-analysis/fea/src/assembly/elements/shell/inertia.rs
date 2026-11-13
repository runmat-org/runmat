use super::{
    transform_matrix_to_global, ShellElementError, ShellLocalFrame, ShellMatrix18, ShellSection,
    SHELL_ELEMENT_DOF_COUNT, SHELL_ELEMENT_NODE_COUNT, SHELL_NODE_DOF_COUNT,
};

/// Through-thickness coefficients retain their order-dependent mass dimensions.
/// DAMPING adds one inverse-time power; an inertia triplet cannot be substituted
/// for a damping triplet merely because the numerical integration is identical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellMomentLayout<const DAMPING: bool>;
impl<const DAMPING: bool> runmat_types::physical::QuantityLayout for ShellMomentLayout<DAMPING> {
    fn component_count(&self) -> usize {
        3
    }
    fn component_descriptor(
        &self,
        index: usize,
    ) -> Option<runmat_types::physical::DynamicDimension> {
        if index >= 3 {
            return None;
        }
        runmat_types::physical::DynamicDimension::try_new(
            runmat_types::PhysicalDimension::new([
                index as i8 - 2,
                1,
                if DAMPING { -1 } else { 0 },
                0,
                0,
                0,
                0,
                0,
            ]),
            None,
        )
        .ok()
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellMassMoments<const DAMPING: bool = false> {
    coefficients: runmat_types::physical::QuantityField<ShellMomentLayout<DAMPING>, [f64; 3]>,
}
impl<const DAMPING: bool> ShellMassMoments<DAMPING> {
    pub fn zero() -> Self {
        Self::from_integrated_coefficients([0.0; 3]).expect("finite zero shell moments")
    }
    fn from_integrated_coefficients(values: [f64; 3]) -> Result<Self, ShellElementError> {
        Ok(Self {
            coefficients: runmat_types::physical::QuantityField::try_new(ShellMomentLayout, values)
                .map_err(|_| ShellElementError::InvalidConstitutive)?,
        })
    }
    pub fn with_coefficients<T>(&self, numerical: impl FnOnce(&[f64]) -> T) -> T {
        self.coefficients.with_canonical_storage(numerical)
    }
    pub fn accumulate(&mut self, contribution: Self) -> Result<(), ShellElementError> {
        let values = self.with_coefficients(|left| {
            contribution
                .with_coefficients(|right| std::array::from_fn(|index| left[index] + right[index]))
        });
        *self = Self::from_integrated_coefficients(values)?;
        Ok(())
    }
    pub fn validate(self, allow_zero: bool) -> Result<(), ShellElementError> {
        self.with_coefficients(|values| {
            let [zeroth, first, second] =
                <[f64; 3]>::try_from(values).expect("three shell moments");
            let nonnegative = zeroth >= 0.0 && second >= 0.0;
            let positive = allow_zero || (zeroth > 0.0 && second > 0.0);
            let determinant = zeroth * second - first.powi(2);
            let tolerance = 64.0 * f64::EPSILON * (zeroth * second).abs().max(first.powi(2));
            if nonnegative && positive && determinant >= -tolerance {
                Ok(())
            } else {
                Err(ShellElementError::InvalidConstitutive)
            }
        })
    }
}
impl ShellMassMoments {
    pub fn integrated_layer(
        density: runmat_types::physical::MassDensity,
        bottom: runmat_types::physical::Length,
        top: runmat_types::physical::Length,
    ) -> Result<Self, ShellElementError> {
        let density = density.canonical_value();
        let (bottom, top) = (bottom.canonical_value(), top.canonical_value());
        Self::from_integrated_coefficients([
            density * (top - bottom),
            density * 0.5 * (top.powi(2) - bottom.powi(2)),
            density * (top.powi(3) - bottom.powi(3)) / 3.0,
        ])
    }
    pub fn proportional_damping(
        self,
        rate: runmat_types::physical::RayleighMassCoefficient,
    ) -> Result<ShellMassMoments<true>, ShellElementError> {
        self.with_coefficients(|values| {
            ShellMassMoments::<true>::from_integrated_coefficients(std::array::from_fn(|index| {
                values[index] * rate.canonical_value()
            }))
        })
    }
}

pub fn homogeneous_mass_moments(
    section: ShellSection,
    density_kg_per_m3: runmat_types::physical::MassDensity,
) -> Result<ShellMassMoments, ShellElementError> {
    section.validate()?;
    let density_value = density_kg_per_m3.canonical_value();
    if !density_value.is_finite() || density_value <= 0.0 {
        return Err(ShellElementError::InvalidConstitutive);
    }
    let bottom =
        -0.5 * section.thickness_m.canonical_value() - section.reference_offset_m.canonical_value();
    let top =
        0.5 * section.thickness_m.canonical_value() - section.reference_offset_m.canonical_value();
    let moments = ShellMassMoments::integrated_layer(
        density_kg_per_m3,
        runmat_types::physical::Length::new(bottom),
        runmat_types::physical::Length::new(top),
    )?;
    moments.validate(false)?;
    Ok(moments)
}

/// Builds a nodally lumped shell mass operator while retaining the exact thickness moments.
///
/// The two physical director rotations use the second moment. The drilling rotation receives the
/// same positive rotary term as a numerical inertia, matching the formulation's drilling
/// stabilization and preventing an otherwise massless explicit degree of freedom.
pub fn global_lumped_mass_matrix<const DAMPING: bool>(
    moments: ShellMassMoments<DAMPING>,
    frame: ShellLocalFrame,
) -> Result<ShellMatrix18, ShellElementError> {
    moments.validate(true)?;
    let [zeroth, first_moment, second] = moments
        .with_coefficients(|values| <[f64; 3]>::try_from(values).expect("three shell moments"));
    let scale = frame.area_m2.canonical_value() / SHELL_ELEMENT_NODE_COUNT as f64;
    let mut local = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];
    for node in 0..SHELL_ELEMENT_NODE_COUNT {
        let base = node * SHELL_NODE_DOF_COUNT;
        for component in 0..3 {
            local[base + component][base + component] = scale * zeroth;
        }
        for component in 3..6 {
            local[base + component][base + component] = scale * second;
        }
        let first = scale * first_moment;
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
    fn inertia_and_proportional_damping_retain_distinct_time_roles() {
        use runmat_types::physical::{
            Length, MassDensity, QuantityLayout, RayleighMassCoefficient,
        };
        let inertia = ShellMassMoments::integrated_layer(
            MassDensity::new(10.0),
            Length::new(-0.1),
            Length::new(0.1),
        )
        .unwrap();
        let damping = inertia
            .proportional_damping(RayleighMassCoefficient::new(3.0))
            .unwrap();
        for order in 0..3 {
            assert_ne!(
                inertia.coefficients.layout().component_descriptor(order),
                damping.coefficients.layout().component_descriptor(order)
            );
            assert_eq!(
                damping.with_coefficients(|values| values[order]),
                3.0 * inertia.with_coefficients(|values| values[order])
            );
        }
        assert!(inertia
            .proportional_damping(RayleighMassCoefficient::new(f64::INFINITY))
            .is_err());
    }

    #[test]
    fn offset_homogeneous_section_retains_first_mass_moment() {
        let moments = homogeneous_mass_moments(
            ShellSection {
                thickness_m: runmat_types::physical::Length::new(0.2),
                reference_offset_m: runmat_types::physical::Length::new(0.1),
                shear_correction: 5.0 / 6.0,
            },
            runmat_types::physical::MassDensity::new(10.0),
        )
        .unwrap();
        assert!((moments.with_coefficients(|values| values[0]) - 2.0).abs() < 1.0e-12);
        assert!((moments.with_coefficients(|values| values[1]) + 0.2).abs() < 1.0e-12);
        assert!(
            (moments.with_coefficients(|values| values[2]) - 0.02666666666666667).abs() < 1.0e-12
        );
    }

    #[test]
    fn shell_mass_coupling_is_symmetric_and_tracks_kinematic_signs() {
        let matrix = global_lumped_mass_matrix(
            ShellMassMoments::<false>::from_integrated_coefficients([2.0, 0.25, 0.1]).unwrap(),
            ShellLocalFrame {
                x: [1.0, 0.0, 0.0],
                y: [0.0, 1.0, 0.0],
                z: [0.0, 0.0, 1.0],
                area_m2: runmat_types::physical::Area::new(3.0),
            },
        )
        .unwrap();
        assert_eq!(matrix[0][4], 0.25);
        assert_eq!(matrix[4][0], 0.25);
        assert_eq!(matrix[1][3], -0.25);
        assert_eq!(matrix[3][1], -0.25);
    }
}
