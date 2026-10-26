use super::*;
use crate::contracts::FeaThermalRunResult;
use crate::pipeline::linear_static::ThermoelasticStaticResult;
use crate::topology::quadrature::{
    tensor_tetrahedron_rule, visit_volume_rule, GAUSS_LEGENDRE_FIVE_UNIT,
};

pub(super) fn measure(
    mesh: &SolverMeshArtifact,
    heat: &FeaThermalRunResult,
    source: &CanonicalSemanticState,
    result: &ThermoelasticStaticResult,
) -> Errors {
    let projection = AnalysisProjection::derive(mesh).unwrap();
    let temperatures = source
        .payload
        .thermal()
        .unwrap()
        .temperature
        .as_host_f64()
        .unwrap();
    let displacements = field(&result.run, crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    let stresses = field(&result.run, crate::FEA_FIELD_STRUCTURAL_STRESS);
    let fluxes = heat.heat_flux_snapshots[0].as_host_f64().unwrap();
    assert_eq!(temperatures.len(), projection.nodes.len());
    assert_eq!(displacements.len(), projection.nodes.len() * 3);
    assert_eq!(stresses.len(), projection.elements.len() * 6);
    assert_eq!(fluxes.len(), projection.elements.len() * 3);
    assert_eq!(
        heat.ordered_node_ids,
        mesh.topology
            .nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        heat.ordered_element_ids,
        projection
            .elements
            .iter()
            .map(|element| element.element_id)
            .collect::<Vec<_>>()
    );
    let balance = heat.energy_balances[0];
    // Unit volume, no storage/convection: q W enters and q W leaves through end reactions.
    assert!((balance.applied_power_w.canonical_value() / GENERATION - 1.0).abs() < 1.0e-10,
        "uniform q={GENERATION} W/m³ on unit volume must apply {GENERATION} W; actual balance {balance:?}");
    assert!(
        (balance.prescribed_boundary_power_w.canonical_value() / GENERATION - 1.0).abs() < 1.0e-8
    );
    assert!(balance.storage_rate_w.canonical_value().abs() < GENERATION * 1.0e-12);
    assert!(balance.boundary_loss_w.canonical_value().abs() < GENERATION * 1.0e-12);
    assert!(balance.relative_residual < 1.0e-8);

    let mut temperature_error = 0.0;
    let mut displacement_error = 0.0;
    let mut flux_error = 0.0;
    let mut stress_error = 0.0;
    let mut total_volume = 0.0;
    for (index, element) in projection.elements.iter().enumerate() {
        let volume = element.signed_volume_m3;
        total_volume += volume;
        // These shared shape functions only evaluate the computed field for an error norm.
        // The reference remains the PDE polynomial, not an assembled FE matrix or solution.
        let coordinates = element
            .node_indices
            .iter()
            .map(|&node| projection.nodes[node].coordinates_m.canonical_value())
            .collect::<Vec<_>>();
        // Positive five-point tensor rule integrates the squared cubic error exactly on
        // these affine tetrahedra. It cannot hide interior error behind nodal coincidence.
        visit_volume_rule(
            element.order,
            &coordinates,
            tensor_tetrahedron_rule(&GAUSS_LEGENDRE_FIVE_UNIT),
            |point| {
                let temperature = element
                    .node_indices
                    .iter()
                    .zip(point.shape_values())
                    .map(|(&node, &shape)| shape * (temperatures[node] - REFERENCE))
                    .sum::<f64>();
                let ux = element
                    .node_indices
                    .iter()
                    .zip(point.shape_values())
                    .map(|(&node, &shape)| shape * displacements[3 * node])
                    .sum::<f64>();
                temperature_error += point.volume_weight_m3
                    * (temperature - excess_temperature(point.position_m[0])).powi(2);
                displacement_error +=
                    point.volume_weight_m3 * (ux - displacement(point.position_m[0])).powi(2);
            },
        )
        .unwrap();
        // Exact simplex moments use only its four physical corners, not FE recovery quadrature.
        // E[x²]=(sum(x_i²)+(sum(x_i))²)/20, E[x]=sum(x_i)/4.
        let x = element.node_indices[..4]
            .iter()
            .map(|&node| projection.nodes[node].coordinates_m.canonical_value()[0])
            .collect::<Vec<_>>();
        let sum = x.iter().sum::<f64>();
        let mean_x = sum / 4.0;
        let mean_x_squared = (x.iter().map(|x| x * x).sum::<f64>() + sum * sum) / 20.0;
        let mean_excess = GENERATION * (mean_x - mean_x_squared) / (2.0 * CONDUCTIVITY);
        let expected_flux = -GENERATION * (1.0 - 2.0 * mean_x) / 2.0;
        for axis in 0..3 {
            flux_error += volume
                * (fluxes[3 * index + axis] - if axis == 0 { expected_flux } else { 0.0 }).powi(2);
        }
        let expected_stress = [
            0.0,
            -YOUNGS * ALPHA * mean_excess,
            -YOUNGS * ALPHA * mean_excess,
            0.0,
            0.0,
            0.0,
        ];
        for (actual, expected) in stresses[6 * index..6 * index + 6]
            .iter()
            .zip(expected_stress)
        {
            stress_error += volume * (actual - expected).powi(2);
        }
    }
    assert!((total_volume - 1.0).abs() < 1.0e-12);
    assert!(displacements
        .chunks_exact(3)
        .all(|u| u[1].abs() < 1.0e-12 && u[2].abs() < 1.0e-12));
    let reactions = field(&result.run, crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    assert_eq!(reactions.len(), projection.nodes.len() * 3);
    let expected_transverse_moment = -YOUNGS * ALPHA * GENERATION / (12.0 * CONDUCTIVITY);
    let mut reaction_moment_error: f64 = 0.0;
    for axis in 0..3 {
        let resultant = reactions
            .chunks_exact(3)
            .map(|reaction| reaction[axis])
            .sum::<f64>();
        assert!(resultant.abs() < YOUNGS * ALPHA * GENERATION * 1.0e-8);
        let moment = projection
            .nodes
            .iter()
            .zip(reactions.chunks_exact(3))
            .map(|(node, reaction)| node.coordinates_m.canonical_value()[axis] * reaction[axis])
            .sum::<f64>();
        let expected = if axis == 0 {
            0.0
        } else {
            expected_transverse_moment
        };
        reaction_moment_error =
            reaction_moment_error.max((moment - expected).abs() / expected_transverse_moment.abs());
    }
    let energy = field(&result.run, crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0];
    // sigma_y,z each contribute 1/2 E alpha² (T-Tref)²; integral x²(1-x)² dx=1/30.
    let expected_energy =
        YOUNGS * ALPHA.powi(2) * GENERATION.powi(2) / (120.0 * CONDUCTIVITY.powi(2));
    assert!(energy.is_finite() && energy >= 0.0);
    Errors {
        temperature: temperature_error.sqrt() / (GENERATION / (8.0 * CONDUCTIVITY)),
        flux: flux_error.sqrt() / (GENERATION / 2.0),
        displacement: displacement_error.sqrt() / displacement(1.0),
        stress: stress_error.sqrt() / (YOUNGS * ALPHA * GENERATION / (8.0 * CONDUCTIVITY)),
        energy: (energy / expected_energy - 1.0).abs(),
        reaction_moment: reaction_moment_error,
    }
}
