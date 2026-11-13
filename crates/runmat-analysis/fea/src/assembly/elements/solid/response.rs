//! Shared physical small-strain solid response for recovery and initial-stress integration.
//!
//! Quadrature selection and global DOF gathering belong to callers. Thermal response at an
//! arbitrary point uses the retained local nodal solution, never a fit to recovery samples.

use super::{material::SolidMaterialError, SolidMaterial};
use crate::assembly::SolidRecoveryElementSummary;
use crate::topology::VolumeQuadraturePoint;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SolidResponsePoint {
    /// Engineering-shear ordering: xx, yy, zz, yz, xz, xy.
    pub total_strain: [f64; 6],
    pub elastic_strain: [f64; 6],
    pub stress: [f64; 6],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SolidResponseError {
    #[error("solid response numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
    #[error("solid response material is invalid")]
    InvalidMaterial,
    #[error("solid response input is inconsistent or non-finite")]
    InvalidInput,
}

pub(crate) fn response_from_strain(
    material: SolidMaterial,
    total_strain: [f64; 6],
    eigenstrain: f64,
) -> Result<SolidResponsePoint, SolidResponseError> {
    let lambda = material.lame_lambda_pa().map_err(material_error)?;
    let mu = material.shear_modulus_pa().map_err(material_error)?;
    if !eigenstrain.is_finite() || total_strain.iter().any(|value| !value.is_finite()) {
        return Err(SolidResponseError::InvalidInput);
    }
    let elastic_strain =
        std::array::from_fn(|i| total_strain[i] - if i < 3 { eigenstrain } else { 0.0 });
    let trace = elastic_strain[..3].iter().sum::<f64>();
    let stress = [
        lambda * trace + 2.0 * mu * elastic_strain[0],
        lambda * trace + 2.0 * mu * elastic_strain[1],
        lambda * trace + 2.0 * mu * elastic_strain[2],
        mu * elastic_strain[3],
        mu * elastic_strain[4],
        mu * elastic_strain[5],
    ];
    if elastic_strain
        .iter()
        .chain(stress.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SolidResponseError::InvalidInput);
    }
    Ok(SolidResponsePoint {
        total_strain,
        elastic_strain,
        stress,
    })
}

fn material_error(error: SolidMaterialError) -> SolidResponseError {
    match error {
        SolidMaterialError::Accounting(error) => SolidResponseError::Accounting(error),
        _ => SolidResponseError::InvalidMaterial,
    }
}

pub(crate) fn response_at_volume_point(
    element: &SolidRecoveryElementSummary,
    local_displacement: &[[f64; 3]],
    point: &VolumeQuadraturePoint,
) -> Result<SolidResponsePoint, SolidResponseError> {
    let count = element.geometry.node_indices().len();
    let gradients = point.physical_shape_gradients();
    let shapes = point.shape_values();
    if local_displacement.len() != count
        || gradients.len() != count
        || shapes.len() != count
        || local_displacement
            .iter()
            .flatten()
            .chain(gradients.iter().flatten())
            .chain(shapes.iter())
            .any(|value| !value.is_finite())
    {
        return Err(SolidResponseError::InvalidInput);
    }
    let mut gradient = [[0.0; 3]; 3];
    for (u, shape_gradient) in local_displacement.iter().zip(gradients) {
        for i in 0..3 {
            for (j, derivative) in shape_gradient.iter().enumerate() {
                gradient[i][j] += u[i] * derivative;
            }
        }
    }
    let total_strain = [
        gradient[0][0],
        gradient[1][1],
        gradient[2][2],
        gradient[1][2] + gradient[2][1],
        gradient[0][2] + gradient[2][0],
        gradient[0][1] + gradient[1][0],
    ];
    let eigenstrain = if let Some(thermal) = &element.solved_thermal_quadrature {
        if element.thermal_eigenstrain.is_some()
            || thermal.nodal_temperature_k.canonical_values().len() != count
            || thermal
                .nodal_temperature_k
                .canonical_values()
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || !thermal
                .reference_temperature_k
                .canonical_value()
                .is_finite()
            || thermal.reference_temperature_k.canonical_value() <= 0.0
            || !thermal
                .expansion_coefficient_per_k
                .canonical_value()
                .is_finite()
        {
            return Err(SolidResponseError::InvalidInput);
        }
        let temperature = thermal
            .nodal_temperature_k
            .canonical_values()
            .iter()
            .zip(shapes)
            .map(|(value, shape)| value * shape)
            .sum::<f64>();
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(SolidResponseError::InvalidInput);
        }
        thermal.expansion_coefficient_per_k.canonical_value()
            * (temperature - thermal.reference_temperature_k.canonical_value())
    } else {
        element.thermal_eigenstrain.unwrap_or(0.0)
    };
    response_from_strain(element.material, total_strain, eigenstrain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_response_subtracts_only_isotropic_normal_eigenstrain() {
        let material = SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(100.0),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
        };
        let free = response_from_strain(material, [0.01, 0.01, 0.01, 0.0, 0.0, 0.0], 0.01).unwrap();
        assert_eq!(free.stress, [0.0; 6]);
        let fixed = response_from_strain(material, [0.0; 6], 0.01).unwrap();
        assert_eq!(fixed.stress, [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0]);
        let shear =
            response_from_strain(material, [0.01, 0.01, 0.01, 0.1, 0.2, 0.3], 0.01).unwrap();
        assert_eq!(shear.stress, [0.0, 0.0, 0.0, 4.0, 8.0, 12.0]);
        assert!(response_from_strain(material, [f64::NAN; 6], 0.0).is_err());
    }

    #[test]
    fn arbitrary_points_use_quadratic_nodal_temperature_and_reject_truncated_inputs() {
        use crate::{
            assembly::solid_recovery,
            fixtures::{fixture_model, FixtureId},
            properties::compile_structural_property_map,
            topology::quadrature::{visit_volume_rule, TETRAHEDRON_ELEVEN_POINT_RULE},
        };
        use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let thermal = model.materials[0].require_thermal().unwrap();
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let temperature = mesh
            .topology
            .nodes
            .iter()
            .map(|node| {
                thermal.reference_temperature_k.canonical_value()
                    + 100.0 * node.coordinates_m.canonical_value()[0].powi(2)
            })
            .collect::<Vec<_>>();
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let mut elements = solid_recovery::build(
            &mesh,
            &properties,
            None,
            Some(&crate::execution::SolvedTemperatureField::nodal(
                temperature.clone(),
            )),
        )
        .unwrap();
        let element = &mut elements[0];
        // Deliberately poison the four visualization/recovery samples: arbitrary-point
        // integration must use the complete nodal polynomial, not fit these samples.
        let cached = element.solved_thermal_quadrature.as_mut().unwrap();
        cached.temperature_k = runmat_types::physical::Quantity::new([1.0; 4]);
        cached.eigenstrain = [123.0; 4];
        let coordinates = mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        let mut visited = 0;
        visit_volume_rule(
            ElementOrder::Tet10,
            &coordinates,
            TETRAHEDRON_ELEVEN_POINT_RULE,
            |point| {
                visited += 1;
                let response = response_at_volume_point(element, &[[0.0; 3]; 10], &point).unwrap();
                let expected = -mechanical.youngs_modulus_pa.canonical_value()
                    * thermal.expansion_coefficient_per_k.canonical_value()
                    * 100.0
                    * point.position_m[0].powi(2)
                    / (1.0 - 2.0 * mechanical.poisson_ratio.canonical_value());
                for actual in &response.stress[..3] {
                    assert!((actual - expected).abs() < 1.0e-6 + expected.abs() * 1.0e-10);
                }
                assert_eq!(response.stress[3..], [0.0; 3]);
                assert_eq!(
                    response_at_volume_point(element, &[[0.0; 3]; 9], &point),
                    Err(SolidResponseError::InvalidInput)
                );
                let mut invalid = element.clone();
                let nodal = &mut invalid
                    .solved_thermal_quadrature
                    .as_mut()
                    .unwrap()
                    .nodal_temperature_k;
                *nodal = nodal
                    .clone()
                    .try_map_canonical_storage(|mut values| {
                        values.pop();
                        values
                    })
                    .unwrap();
                assert_eq!(
                    response_at_volume_point(&invalid, &[[0.0; 3]; 10], &point),
                    Err(SolidResponseError::InvalidInput)
                );
                invalid = element.clone();
                invalid.thermal_eigenstrain = Some(1.0);
                assert_eq!(
                    response_at_volume_point(&invalid, &[[0.0; 3]; 10], &point),
                    Err(SolidResponseError::InvalidInput)
                );
            },
        )
        .unwrap();
        assert_eq!(visited, 11);
    }
}
