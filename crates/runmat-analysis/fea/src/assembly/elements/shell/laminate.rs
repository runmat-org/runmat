use super::constitutive::{
    ShellConstitutive, ShellConstitutiveMatrices, ShellMatrix2, ShellMatrix3, ShellRecoveryLayer,
};
use super::ShellElementError;
use super::ShellMassMoments;

const FORMULATION_SHEAR_CORRECTION: f64 = 5.0 / 6.0;
const FORMULATION_DRILLING_SCALE: f64 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaminaElasticity {
    pub longitudinal_modulus_pa: runmat_types::physical::YoungsModulus,
    pub transverse_modulus_pa: runmat_types::physical::YoungsModulus,
    pub major_poisson_ratio: runmat_types::physical::PoissonRatio,
    pub in_plane_shear_modulus_pa: runmat_types::physical::ShearModulus,
    pub transverse_shear_modulus_13_pa: runmat_types::physical::ShearModulus,
    pub transverse_shear_modulus_23_pa: runmat_types::physical::ShearModulus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaminateLayer {
    pub layer_id: String,
    pub thickness_m: runmat_types::physical::Length,
    pub angle_rad: runmat_types::physical::Angle,
    pub elasticity: LaminaElasticity,
    pub density_kg_per_m3: runmat_types::physical::MassDensity,
    pub mass_damping_per_second: runmat_types::physical::RayleighMassCoefficient,
    pub stiffness_damping_seconds: runmat_types::physical::RayleighStiffnessCoefficient,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaminatePlyOrder {
    AlongSurfaceNormal,
    OppositeSurfaceNormal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaminateReferenceSurface {
    Mid,
    FirstPly,
    LastPly,
    OffsetFromMid(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaminateDefinition {
    pub layers: Vec<LaminateLayer>,
    pub ply_order: LaminatePlyOrder,
    pub reference_surface: LaminateReferenceSurface,
    pub shear_correction: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaminateMassProperties {
    pub inertia: ShellMassMoments,
    pub mass_damping: ShellMassMoments<true>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaminateResponse {
    pub elastic: ShellConstitutive,
    pub stiffness_damping: ShellConstitutiveMatrices,
    pub mass: LaminateMassProperties,
}

impl LaminateDefinition {
    pub fn integrate(&self) -> Result<LaminateResponse, ShellElementError> {
        self.validate()?;
        let bounds = self.layer_bounds();
        let shear_correction = self
            .shear_correction
            .unwrap_or(FORMULATION_SHEAR_CORRECTION);
        let mut elastic = ShellConstitutiveMatrices::zero();
        let mut stiffness_damping = ShellConstitutiveMatrices::zero();
        let mut recovery_layers = Vec::with_capacity(self.layers.len());
        let mut mass = LaminateMassProperties {
            inertia: ShellMassMoments::zero(),
            mass_damping: ShellMassMoments::<true>::zero(),
        };

        for (layer, [bottom, top]) in self.layers.iter().zip(bounds) {
            let plane_stress =
                transformed_plane_stress(layer.elasticity, layer.angle_rad.canonical_value());
            let transverse_shear =
                transformed_transverse_shear(layer.elasticity, layer.angle_rad.canonical_value());
            integrate_layer(
                &mut elastic,
                plane_stress,
                transverse_shear,
                bottom,
                top,
                shear_correction,
                1.0,
            );
            integrate_layer(
                &mut stiffness_damping,
                plane_stress,
                transverse_shear,
                bottom,
                top,
                shear_correction,
                layer.stiffness_damping_seconds.canonical_value(),
            );
            recovery_layers.push(ShellRecoveryLayer {
                layer_id: layer.layer_id.clone(),
                bottom_m: bottom,
                top_m: top,
                plane_stress,
            });

            let layer_moments = ShellMassMoments::integrated_layer(
                layer.density_kg_per_m3,
                runmat_types::physical::Length::new(bottom),
                runmat_types::physical::Length::new(top),
            )?;
            mass.inertia.accumulate(layer_moments)?;
            mass.mass_damping
                .accumulate(layer_moments.proportional_damping(layer.mass_damping_per_second)?)?;
        }

        elastic.drilling_stiffness_per_area =
            drilling_stiffness(&elastic.extensional, FORMULATION_DRILLING_SCALE);
        stiffness_damping.drilling_stiffness_per_area =
            drilling_stiffness(&stiffness_damping.extensional, FORMULATION_DRILLING_SCALE);
        let constitutive = ShellConstitutive {
            matrices: elastic,
            recovery_layers,
        };
        constitutive.validate()?;
        stiffness_damping.validate_damping()?;
        mass.inertia.validate(false)?;
        mass.mass_damping.validate(true)?;
        Ok(LaminateResponse {
            elastic: constitutive,
            stiffness_damping,
            mass,
        })
    }

    fn validate(&self) -> Result<(), ShellElementError> {
        if self.layers.is_empty()
            || self
                .shear_correction
                .is_some_and(|value| !positive_finite(value))
            || self.layers.iter().any(|layer| {
                layer.layer_id.trim().is_empty()
                    || !positive_finite(layer.thickness_m.canonical_value())
                    || !layer.angle_rad.is_finite()
                    || !positive_finite(layer.density_kg_per_m3.canonical_value())
                    || !nonnegative_finite(layer.mass_damping_per_second.canonical_value())
                    || !nonnegative_finite(layer.stiffness_damping_seconds.canonical_value())
                    || layer.elasticity.validate().is_err()
            })
        {
            return Err(ShellElementError::InvalidConstitutive);
        }
        if let LaminateReferenceSurface::OffsetFromMid(offset) = self.reference_surface {
            if !offset.is_finite() {
                return Err(ShellElementError::InvalidConstitutive);
            }
        }
        Ok(())
    }

    fn layer_bounds(&self) -> Vec<[f64; 2]> {
        let total_thickness: f64 = self
            .layers
            .iter()
            .map(|layer| layer.thickness_m.canonical_value())
            .sum();
        let mut geometric = Vec::with_capacity(self.layers.len());
        match self.ply_order {
            LaminatePlyOrder::AlongSurfaceNormal => {
                let mut cursor = -0.5 * total_thickness;
                for layer in &self.layers {
                    geometric.push([cursor, cursor + layer.thickness_m.canonical_value()]);
                    cursor += layer.thickness_m.canonical_value();
                }
            }
            LaminatePlyOrder::OppositeSurfaceNormal => {
                let mut cursor = 0.5 * total_thickness;
                for layer in &self.layers {
                    geometric.push([cursor - layer.thickness_m.canonical_value(), cursor]);
                    cursor -= layer.thickness_m.canonical_value();
                }
            }
        }
        let first_outer = match self.ply_order {
            LaminatePlyOrder::AlongSurfaceNormal => geometric[0][0],
            LaminatePlyOrder::OppositeSurfaceNormal => geometric[0][1],
        };
        let last_outer = match self.ply_order {
            LaminatePlyOrder::AlongSurfaceNormal => geometric.last().expect("non-empty")[1],
            LaminatePlyOrder::OppositeSurfaceNormal => geometric.last().expect("non-empty")[0],
        };
        let reference = match self.reference_surface {
            LaminateReferenceSurface::Mid => 0.0,
            LaminateReferenceSurface::FirstPly => first_outer,
            LaminateReferenceSurface::LastPly => last_outer,
            LaminateReferenceSurface::OffsetFromMid(offset) => offset,
        };
        geometric
            .into_iter()
            .map(|[bottom, top]| [bottom - reference, top - reference])
            .collect()
    }
}

impl LaminaElasticity {
    fn validate(self) -> Result<(), ShellElementError> {
        let reciprocal_poisson = self.major_poisson_ratio.canonical_value()
            * self.transverse_modulus_pa.canonical_value()
            / self.longitudinal_modulus_pa.canonical_value();
        if positive_finite(self.longitudinal_modulus_pa.canonical_value())
            && positive_finite(self.transverse_modulus_pa.canonical_value())
            && positive_finite(self.in_plane_shear_modulus_pa.canonical_value())
            && positive_finite(self.transverse_shear_modulus_13_pa.canonical_value())
            && positive_finite(self.transverse_shear_modulus_23_pa.canonical_value())
            && self.major_poisson_ratio.canonical_value().is_finite()
            && 1.0 - self.major_poisson_ratio.canonical_value() * reciprocal_poisson > 0.0
        {
            Ok(())
        } else {
            Err(ShellElementError::InvalidConstitutive)
        }
    }
}

fn integrate_layer(
    matrices: &mut ShellConstitutiveMatrices,
    plane_stress: ShellMatrix3,
    transverse_shear: ShellMatrix2,
    bottom: f64,
    top: f64,
    shear_correction: f64,
    scale: f64,
) {
    let thickness = top - bottom;
    add_scaled3(&mut matrices.extensional, plane_stress, scale * thickness);
    add_scaled3(
        &mut matrices.coupling,
        plane_stress,
        scale * 0.5 * (top.powi(2) - bottom.powi(2)),
    );
    add_scaled3(
        &mut matrices.bending,
        plane_stress,
        scale * (top.powi(3) - bottom.powi(3)) / 3.0,
    );
    add_scaled2(
        &mut matrices.transverse_shear,
        transverse_shear,
        scale * shear_correction * thickness,
    );
}

fn transformed_plane_stress(material: LaminaElasticity, angle: f64) -> ShellMatrix3 {
    let reciprocal_poisson = material.major_poisson_ratio.canonical_value()
        * material.transverse_modulus_pa.canonical_value()
        / material.longitudinal_modulus_pa.canonical_value();
    let denominator = 1.0 - material.major_poisson_ratio.canonical_value() * reciprocal_poisson;
    let q11 = material.longitudinal_modulus_pa.canonical_value() / denominator;
    let q22 = material.transverse_modulus_pa.canonical_value() / denominator;
    let q12 = material.major_poisson_ratio.canonical_value()
        * material.transverse_modulus_pa.canonical_value()
        / denominator;
    let q66 = material.in_plane_shear_modulus_pa.canonical_value();
    let m = angle.cos();
    let n = angle.sin();
    let m2 = m * m;
    let n2 = n * n;
    let m4 = m2 * m2;
    let n4 = n2 * n2;
    let m2n2 = m2 * n2;
    let q16 = (q11 - q12 - 2.0 * q66) * m * m2 * n - (q22 - q12 - 2.0 * q66) * m * n * n2;
    let q26 = (q11 - q12 - 2.0 * q66) * m * n * n2 - (q22 - q12 - 2.0 * q66) * m * m2 * n;
    [
        [
            q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2n2 + q22 * n4,
            (q11 + q22 - 4.0 * q66) * m2n2 + q12 * (m4 + n4),
            q16,
        ],
        [
            (q11 + q22 - 4.0 * q66) * m2n2 + q12 * (m4 + n4),
            q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2n2 + q22 * m4,
            q26,
        ],
        [
            q16,
            q26,
            (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2n2 + q66 * (m4 + n4),
        ],
    ]
}

fn transformed_transverse_shear(material: LaminaElasticity, angle: f64) -> ShellMatrix2 {
    let cosine = angle.cos();
    let sine = angle.sin();
    let g13 = material.transverse_shear_modulus_13_pa.canonical_value();
    let g23 = material.transverse_shear_modulus_23_pa.canonical_value();
    [
        [
            g13 * cosine.powi(2) + g23 * sine.powi(2),
            (g13 - g23) * cosine * sine,
        ],
        [
            (g13 - g23) * cosine * sine,
            g13 * sine.powi(2) + g23 * cosine.powi(2),
        ],
    ]
}

fn add_scaled3(target: &mut ShellMatrix3, source: ShellMatrix3, scale: f64) {
    for row in 0..3 {
        for column in 0..3 {
            target[row][column] += scale * source[row][column];
        }
    }
}

fn add_scaled2(target: &mut ShellMatrix2, source: ShellMatrix2, scale: f64) {
    for row in 0..2 {
        for column in 0..2 {
            target[row][column] += scale * source[row][column];
        }
    }
}

fn drilling_stiffness(extensional: &ShellMatrix3, scale: f64) -> f64 {
    scale * 0.5 * (extensional[0][0] + extensional[1][1])
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::elements::shell::{
        recover_resultants_for_constitutive, ShellElementGeometry, SHELL_ELEMENT_DOF_COUNT,
        SHELL_NODE_DOF_COUNT,
    };

    fn carbon_epoxy(angle_rad: f64, thickness_m: f64) -> LaminateLayer {
        LaminateLayer {
            layer_id: format!("ply-{angle_rad}"),
            thickness_m: runmat_types::physical::Length::new(thickness_m),
            angle_rad: runmat_types::physical::Angle::new(angle_rad),
            elasticity: LaminaElasticity {
                longitudinal_modulus_pa: runmat_types::physical::YoungsModulus::new(135.0e9),
                transverse_modulus_pa: runmat_types::physical::YoungsModulus::new(10.0e9),
                major_poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
                in_plane_shear_modulus_pa: runmat_types::physical::ShearModulus::new(5.0e9),
                transverse_shear_modulus_13_pa: runmat_types::physical::ShearModulus::new(5.0e9),
                transverse_shear_modulus_23_pa: runmat_types::physical::ShearModulus::new(3.8e9),
            },
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1_600.0),
            mass_damping_per_second: runmat_types::physical::RayleighMassCoefficient::new(0.0),
            stiffness_damping_seconds: runmat_types::physical::RayleighStiffnessCoefficient::new(
                0.0,
            ),
        }
    }

    #[test]
    fn symmetric_laminate_has_zero_membrane_bending_coupling() {
        let ply = 0.000_125;
        let definition = LaminateDefinition {
            layers: vec![
                carbon_epoxy(0.0, ply),
                carbon_epoxy(std::f64::consts::FRAC_PI_2, ply),
                carbon_epoxy(std::f64::consts::FRAC_PI_2, ply),
                carbon_epoxy(0.0, ply),
            ],
            ply_order: LaminatePlyOrder::AlongSurfaceNormal,
            reference_surface: LaminateReferenceSurface::Mid,
            shear_correction: None,
        };
        let response = definition.integrate().unwrap();
        for row in response.elastic.matrices.coupling {
            for value in row {
                assert!(value.abs() < 1.0e-9);
            }
        }
        assert!(
            (response.mass.inertia.with_coefficients(|values| values[0]) - 0.8).abs() < 1.0e-12
        );
    }

    #[test]
    fn reference_surface_changes_coupling_but_not_extensional_stiffness() {
        let layers = vec![carbon_epoxy(0.0, 0.001), carbon_epoxy(0.3, 0.002)];
        let mid = LaminateDefinition {
            layers: layers.clone(),
            ply_order: LaminatePlyOrder::AlongSurfaceNormal,
            reference_surface: LaminateReferenceSurface::Mid,
            shear_correction: Some(1.0),
        }
        .integrate()
        .unwrap();
        let first = LaminateDefinition {
            layers,
            ply_order: LaminatePlyOrder::AlongSurfaceNormal,
            reference_surface: LaminateReferenceSurface::FirstPly,
            shear_correction: Some(1.0),
        }
        .integrate()
        .unwrap();
        assert_eq!(
            mid.elastic.matrices.extensional,
            first.elastic.matrices.extensional
        );
        assert_ne!(
            mid.elastic.matrices.coupling,
            first.elastic.matrices.coupling
        );
        assert!(mid.mass.inertia.with_coefficients(|values| values[1]).abs() < 1.0e-12);
        assert!(first.mass.inertia.with_coefficients(|values| values[1]) > 0.0);
    }

    #[test]
    fn ninety_degree_rotation_swaps_orthotropic_extensional_axes() {
        let zero = transformed_plane_stress(carbon_epoxy(0.0, 1.0).elasticity, 0.0);
        let ninety = transformed_plane_stress(
            carbon_epoxy(0.0, 1.0).elasticity,
            std::f64::consts::FRAC_PI_2,
        );
        assert!((zero[0][0] - ninety[1][1]).abs() < 1.0e-12 * zero[0][0]);
        assert!((zero[1][1] - ninety[0][0]).abs() < 1.0e-12 * zero[0][0]);
        assert!(ninety[0][2].abs() < 1.0e-5);
        assert!(ninety[1][2].abs() < 1.0e-5);
    }

    // witness: FEA-STRUCTURAL-REFERENCE-001/patch_reference_convergence @
    //   linear_structural/shell/numerical_kernel; coverage: partial;
    //   check: shell-membrane-patch; boundary: A one-layer membrane patch matches the independent plane-stress reference; no
    //   convergence series.
    #[test]
    fn one_layer_membrane_patch_matches_the_independent_plane_stress_reference() {
        let thickness = 0.001;
        let layer = carbon_epoxy(0.0, thickness);
        let material = layer.elasticity;
        let response = LaminateDefinition {
            layers: vec![layer],
            ply_order: LaminatePlyOrder::AlongSurfaceNormal,
            reference_surface: LaminateReferenceSurface::Mid,
            shear_correction: None,
        }
        .integrate()
        .unwrap();
        let geometry = ShellElementGeometry {
            nodes_m: [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            reference_axis: [1.0, 0.0, 0.0],
        };
        let strain_x = 2.5e-4;
        let mut displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
        for (node, coordinates) in geometry.nodes_m.iter().enumerate() {
            displacement[node * SHELL_NODE_DOF_COUNT] = strain_x * coordinates[0];
        }
        let recovered =
            recover_resultants_for_constitutive(&response.elastic, geometry, displacement).unwrap();
        let reciprocal_poisson = material.major_poisson_ratio.canonical_value()
            * material.transverse_modulus_pa.canonical_value()
            / material.longitudinal_modulus_pa.canonical_value();
        let denominator = 1.0 - material.major_poisson_ratio.canonical_value() * reciprocal_poisson;
        let stress_x = material.longitudinal_modulus_pa.canonical_value() * strain_x / denominator;
        let stress_y = material.major_poisson_ratio.canonical_value()
            * material.transverse_modulus_pa.canonical_value()
            * strain_x
            / denominator;
        assert_close(recovered.membrane_force_n_per_m[0], stress_x * thickness);
        assert_close(recovered.membrane_force_n_per_m[1], stress_y * thickness);
        assert_close(recovered.membrane_force_n_per_m[2], 0.0);
        assert!(recovered
            .bending_moment_n
            .iter()
            .all(|value| value.abs() < 1.0e-12));
        let expected_von_mises = (stress_x.powi(2) - stress_x * stress_y + stress_y.powi(2)).sqrt();
        assert_close(recovered.surface_von_mises_pa, expected_von_mises);
    }

    #[test]
    fn uniform_rayleigh_coefficients_integrate_to_the_same_layered_operator_scaling() {
        let alpha = 0.35;
        let beta = 1.7e-5;
        let mut layers = vec![carbon_epoxy(0.0, 0.000_2), carbon_epoxy(0.4, 0.000_3)];
        for layer in &mut layers {
            layer.mass_damping_per_second =
                runmat_types::physical::RayleighMassCoefficient::new(alpha);
            layer.stiffness_damping_seconds =
                runmat_types::physical::RayleighStiffnessCoefficient::new(beta);
        }
        let response = LaminateDefinition {
            layers,
            ply_order: LaminatePlyOrder::AlongSurfaceNormal,
            reference_surface: LaminateReferenceSurface::Mid,
            shear_correction: Some(0.9),
        }
        .integrate()
        .unwrap();
        for row in 0..3 {
            for column in 0..3 {
                assert_close(
                    response.stiffness_damping.extensional[row][column],
                    beta * response.elastic.matrices.extensional[row][column],
                );
                assert_close(
                    response.stiffness_damping.coupling[row][column],
                    beta * response.elastic.matrices.coupling[row][column],
                );
                assert_close(
                    response.stiffness_damping.bending[row][column],
                    beta * response.elastic.matrices.bending[row][column],
                );
            }
        }
        assert_close(
            response
                .mass
                .mass_damping
                .with_coefficients(|values| values[0]),
            alpha * response.mass.inertia.with_coefficients(|values| values[0]),
        );
        assert_close(
            response
                .mass
                .mass_damping
                .with_coefficients(|values| values[2]),
            alpha * response.mass.inertia.with_coefficients(|values| values[2]),
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1.0e-11 * actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:e} expected={expected:e} tolerance={tolerance:e}"
        );
    }
}
