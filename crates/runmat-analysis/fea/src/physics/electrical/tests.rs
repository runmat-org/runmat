use super::*;
use crate::{
    fixtures::{fixture_model, FixtureId},
    topology::{
        test_meshes::{quadratic_slab_mesh, slab_mesh},
        AnalysisProjection,
    },
};
use runmat_analysis_core::*;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};
pub(crate) mod charge_density;
mod energy;

pub(crate) fn mesh(order: ElementOrder, cells: usize) -> SolverMeshArtifact {
    match order {
        ElementOrder::Tet4 => slab_mesh(cells),
        ElementOrder::Tet10 => quadratic_slab_mesh(cells),
    }
}

pub(crate) fn model(equation: StaticElectricalEquation) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.cfd = None;
    model.electromagnetic = Some(ElectromagneticDomain::Static { equation });
    model.sequence = AnalysisSequence::from_steps(
        "electrical",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.material_assignments[0].target = PropertyRegionTarget::solid("slab");
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: 293.15,
        conductivity_s_per_m: 3.0,
        relative_permittivity: 4.0,
        relative_permeability: 1.0,
        resistive_heating_coefficient: 0.0,
        conductivity_frequency_response: Vec::new(),
    });
    model
}

pub(crate) fn prescribe(model: &mut AnalysisModel, id: String, target: RegionTarget, volts: f64) {
    model
        .constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: id,
            kind: ElectricalConstraintKind::PrescribedPotential {
                target,
                value: PrescribedValue::Constant(ElectricPotential::new(volts)),
            },
        }));
}

fn solve(model: &AnalysisModel, mesh: &SolverMeshArtifact) -> CanonicalElectricalResponse {
    solve_canonical_static(
        model,
        mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-11,
    )
    .unwrap()
}

fn coefficient(equation: StaticElectricalEquation) -> f64 {
    match equation {
        StaticElectricalEquation::Electrostatics => 4.0 * VACUUM_PERMITTIVITY_F_PER_M,
        StaticElectricalEquation::SteadyConduction => 3.0,
    }
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(f64::MIN_POSITIVE),
        "actual={actual:e} expected={expected:e}"
    );
}

#[test]
fn homogeneous_capacitor_and_conductor_match_independent_slab_field_terminals_and_energy() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = mesh(order, 2);
            let mut model = model(equation);
            prescribe(
                &mut model,
                "left".into(),
                RegionTarget::boundary("slab_left"),
                0.0,
            );
            prescribe(
                &mut model,
                "right".into(),
                RegionTarget::boundary("slab_right"),
                2.0,
            );
            let response = solve(&model, &mesh);
            assert_eq!(response.ordered_node_ids.len(), mesh.topology.nodes.len());
            assert_eq!(
                response.ordered_element_ids.len(),
                mesh.topology.volume_elements.len()
            );
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            for (&potential, node) in response.potential_v.iter().zip(&projection.nodes) {
                assert!((potential - 2.0 * node.coordinates_m[0]).abs() < 1e-8);
            }
            for field in &response.element_electric_field_v_per_m {
                assert!((field[0] + 2.0).abs() < 1e-8);
                assert!(field[1].abs() < 1e-8 && field[2].abs() < 1e-8);
            }
            let a = coefficient(equation);
            assert_relative(response.terminals[0].outward_flux, 2.0 * a, 1e-8);
            assert_relative(response.terminals[1].outward_flux, -2.0 * a, 1e-8);
            assert!(response.relative_flux_imbalance < 1e-8);
            assert!(response.relative_energy_identity_error < 1e-8);
            match response.constitutive {
                ElectricalConstitutiveResponse::Electrostatics {
                    total_energy_j,
                    displacement_c_per_m2,
                    ..
                } => {
                    assert_relative(total_energy_j, 2.0 * a, 1e-8);
                    for displacement in displacement_c_per_m2 {
                        assert_relative(displacement[0], -2.0 * a, 1e-8);
                    }
                }
                ElectricalConstitutiveResponse::SteadyConduction {
                    total_joule_power_w,
                    current_density_a_per_m2,
                    ..
                } => {
                    assert_relative(total_joule_power_w, 4.0 * a, 1e-8);
                    for current in current_density_a_per_m2 {
                        assert_relative(current[0], -2.0 * a, 1e-8);
                    }
                }
            }
        }
    }
}

#[test]
fn layered_materials_match_series_capacitance_and_resistance_with_interface_flux_continuity() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut mesh = mesh(order, 2);
            let left = crate::topology::test_meshes::entity(
                runmat_meshing_core::PersistentEntityKind::Solid,
                "layer_left",
            );
            let right = crate::topology::test_meshes::entity(
                runmat_meshing_core::PersistentEntityKind::Solid,
                "layer_right",
            );
            for element in &mut mesh.topology.volume_elements {
                let x = element.node_ids[..4]
                    .iter()
                    .map(|id| {
                        mesh.topology
                            .nodes
                            .iter()
                            .find(|node| node.node_id == *id)
                            .unwrap()
                            .coordinates_m[0]
                    })
                    .sum::<f64>()
                    / 4.0;
                element.provenance = vec![if x < 0.5 { left.clone() } else { right.clone() }];
            }
            mesh.seal_canonical_digest().unwrap();
            let mut model = model(equation);
            let mut second = model.materials[0].clone();
            second.material_id = "second".into();
            let electrical = second.electrical.as_mut().unwrap();
            electrical.conductivity_s_per_m *= 4.0;
            electrical.relative_permittivity *= 4.0;
            model.materials.push(second);
            model.material_assignments[0].target = PropertyRegionTarget::solid("layer_left");
            let mut second = model.material_assignments[0].clone();
            second.target = PropertyRegionTarget::solid("layer_right");
            second.expected_material_id = "second".into();
            second.assigned_material_id = "second".into();
            model.material_assignments.push(second);
            prescribe(
                &mut model,
                "left".into(),
                RegionTarget::boundary("slab_left"),
                0.0,
            );
            prescribe(
                &mut model,
                "right".into(),
                RegionTarget::boundary("slab_right"),
                1.0,
            );
            let response = solve(&model, &mesh);
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            for (node, &potential) in projection.nodes.iter().zip(&response.potential_v) {
                let x = node.coordinates_m[0];
                let exact = if x <= 0.5 {
                    1.6 * x
                } else {
                    0.8 + 0.4 * (x - 0.5)
                };
                assert!(
                    (potential - exact).abs() < 1e-8,
                    "x={x} phi={potential} exact={exact}"
                );
            }
            let a = coefficient(equation);
            let flux = match &response.constitutive {
                ElectricalConstitutiveResponse::Electrostatics {
                    displacement_c_per_m2,
                    total_energy_j,
                    ..
                } => {
                    assert_relative(*total_energy_j, 0.8 * a, 1e-8);
                    displacement_c_per_m2
                }
                ElectricalConstitutiveResponse::SteadyConduction {
                    current_density_a_per_m2,
                    total_joule_power_w,
                    ..
                } => {
                    assert_relative(*total_joule_power_w, 1.6 * a, 1e-8);
                    current_density_a_per_m2
                }
            };
            for value in flux {
                assert_relative(value[0], -1.6 * a, 1e-8);
                assert!(value[1].abs() < a * 1e-8 && value[2].abs() < a * 1e-8);
            }
            assert_relative(response.terminals[0].outward_flux, 1.6 * a, 1e-8);
            assert_relative(response.terminals[1].outward_flux, -1.6 * a, 1e-8);
        }
    }
}

#[test]
fn constant_gauge_potential_produces_exactly_zero_field_and_power() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        let mesh = mesh(ElementOrder::Tet10, 2);
        let mut model = model(equation);
        prescribe(
            &mut model,
            "left".into(),
            RegionTarget::boundary("slab_left"),
            1e6,
        );
        let response = solve(&model, &mesh);
        assert!(response.potential_v.iter().all(|value| *value == 1e6));
        assert!(response
            .element_electric_field_v_per_m
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
        assert!(response
            .nodal_outward_flux
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(response.relative_energy_identity_error, 0.0);
        assert_eq!(response.relative_flux_imbalance, 0.0);
    }
}

#[test]
fn harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence() {
    // phi=x^2-y^2 is harmonic: no fabricated volume charge/current is needed.
    // On unit cube, integral |E|^2 = integral 4(x^2+y^2) = 8/3.
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mut errors = Vec::new();
            for cells in [2, 4, 8] {
                let mesh = mesh(order, cells);
                let mut model = model(equation);
                let projection = AnalysisProjection::derive(&mesh).unwrap();
                let boundary = projection
                    .boundary_faces
                    .iter()
                    .flat_map(|face| face.node_indices.iter().copied())
                    .collect::<std::collections::BTreeSet<_>>();
                for node in boundary {
                    let node = &projection.nodes[node];
                    let [x, y, _] = node.coordinates_m;
                    prescribe(
                        &mut model,
                        format!("node_{}", node.node_id),
                        RegionTarget::point(format!("node:{}", node.node_id)),
                        x * x - y * y,
                    );
                }
                let response = solve(&model, &mesh);
                let mut error = 0.0;
                let mut exact = 0.0;
                for element in &projection.elements {
                    projection
                        .visit_element_quadrature(element, |point| {
                            let mut coordinate = [0.0; 3];
                            let mut field = [0.0; 3];
                            for (local, &node) in element.node_indices.iter().enumerate() {
                                for axis in 0..3 {
                                    coordinate[axis] += point.shape_values()[local]
                                        * projection.nodes[node].coordinates_m[axis];
                                    field[axis] -= point.physical_shape_gradients()[local][axis]
                                        * response.potential_v[node];
                                }
                            }
                            let reference = [-2.0 * coordinate[0], 2.0 * coordinate[1], 0.0];
                            error += field
                                .iter()
                                .zip(reference)
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                * point.volume_weight_m3;
                            exact += reference.iter().map(|value| value * value).sum::<f64>()
                                * point.volume_weight_m3;
                        })
                        .unwrap();
                }
                let relative = (error / exact).sqrt();
                errors.push(relative);
                if order == ElementOrder::Tet10 {
                    assert!(relative < 1e-8, "quadratic exactness error {relative:e}");
                    let (total, factor) = match response.constitutive {
                        ElectricalConstitutiveResponse::Electrostatics {
                            total_energy_j, ..
                        } => (total_energy_j, 0.5),
                        ElectricalConstitutiveResponse::SteadyConduction {
                            total_joule_power_w,
                            ..
                        } => (total_joule_power_w, 1.0),
                    };
                    assert_relative(total, factor * coefficient(equation) * 8.0 / 3.0, 1e-8);
                }
            }
            if order == ElementOrder::Tet4 {
                assert!(
                    errors[1] < errors[0] * 0.6 && errors[2] < errors[1] * 0.6,
                    "{errors:?}"
                );
            }
        }
    }
}

#[test]
fn missing_anchors_conflicts_unsupported_intent_and_invalid_material_fail_closed() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    assert!(solve_canonical_static(
        &model,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary("slab_left"),
        0.0,
    );
    prescribe(
        &mut model,
        "right".into(),
        RegionTarget::boundary("slab_right"),
        1.0,
    );
    assert!(solve_canonical_static(
        &model,
        &mesh,
        "potential",
        FormulationRequirement::Production,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    bad.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = 0.0;
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    prescribe(
        &mut bad,
        "conflict".into(),
        RegionTarget::boundary("slab_left"),
        1.0,
    );
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    bad.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(60.0),
    });
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    bad.constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "floating".into(),
            kind: ElectricalConstraintKind::FloatingConductor {
                target: RegionTarget::boundary("slab_right"),
            },
        }));
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    prescribe(
        &mut bad,
        "missing".into(),
        RegionTarget::point("node:999999"),
        1.0,
    );
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    bad.sequence.steps.push(bad.sequence.steps[0].clone());
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
    let mut bad = model.clone();
    bad.material_assignments.clear();
    assert!(solve_canonical_static(
        &bad,
        &mesh,
        "potential",
        FormulationRequirement::Experimental,
        1e-8
    )
    .is_err());
}
