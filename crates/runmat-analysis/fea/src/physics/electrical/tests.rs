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
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(4.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
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
    solve_canonical_static(model, mesh, "potential", 1e-11).unwrap()
}

fn coefficient(equation: StaticElectricalEquation) -> f64 {
    match equation {
        StaticElectricalEquation::Electrostatics => {
            4.0 * VACUUM_PERMITTIVITY_F_PER_M.canonical_value()
        }
        StaticElectricalEquation::SteadyConduction => 3.0,
    }
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(f64::MIN_POSITIVE),
        "actual={actual:e} expected={expected:e}"
    );
}

// witness: FEA-EM-SCALAR-001/operator @ electrostatic/tet4|tet10/numerical_kernel,
//   electrical_conduction/tet4|tet10/numerical_kernel;
//   boundary: Tet4/Tet10 homogeneous slab capacitor and conductor potentials, fields, terminal
//   flux, energy and Joule power; not arbitrary geometry or nonlinear material.
// witness: FEA-EM-SCALAR-001/recovery @ electrostatic/tet4|tet10/numerical_kernel,
//   electrical_conduction/tet4|tet10/numerical_kernel;
//   check: slab-capacitor-conductor; boundary: Same slab constitutive vectors and global terminal/energy identities; no general
//   charge-density reconstruction claim.
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
                RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
                0.0,
            );
            prescribe(
                &mut model,
                "right".into(),
                RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
                2.0,
            );
            let response = solve(&model, &mesh);
            assert_eq!(response.ordered_node_ids.len(), mesh.topology.nodes.len());
            assert_eq!(
                response.ordered_element_ids.len(),
                mesh.topology.volume_elements.len()
            );
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            for (&potential, node) in response
                .potential_v
                .canonical_values()
                .iter()
                .zip(&projection.nodes)
            {
                assert!((potential - 2.0 * node.coordinates_m.canonical_value()[0]).abs() < 1e-8);
            }
            for field in response.element_electric_field_v_per_m.canonical_values() {
                assert!((field[0] + 2.0).abs() < 1e-8);
                assert!(field[1].abs() < 1e-8 && field[2].abs() < 1e-8);
            }
            let a = coefficient(equation);
            assert_relative(
                response.terminals[0].outward_flux.canonical_value(),
                2.0 * a,
                1e-8,
            );
            assert_relative(
                response.terminals[1].outward_flux.canonical_value(),
                -2.0 * a,
                1e-8,
            );
            assert!(response.relative_flux_imbalance < 1e-8);
            assert!(response.relative_energy_identity_error < 1e-8);
            match response.constitutive {
                ElectricalConstitutiveResponse::Electrostatics {
                    total_energy_j,
                    displacement_c_per_m2,
                    ..
                } => {
                    assert_relative(total_energy_j.canonical_value(), 2.0 * a, 1e-8);
                    for displacement in displacement_c_per_m2.canonical_values() {
                        assert_relative(displacement[0], -2.0 * a, 1e-8);
                    }
                }
                ElectricalConstitutiveResponse::SteadyConduction {
                    total_joule_power_w,
                    current_density_a_per_m2,
                    ..
                } => {
                    assert_relative(total_joule_power_w.canonical_value(), 4.0 * a, 1e-8);
                    for current in current_density_a_per_m2.canonical_values() {
                        assert_relative(current[0], -2.0 * a, 1e-8);
                    }
                }
            }
        }
    }
}

// witness: FEA-EM-SCALAR-001/operator @ electrostatic/tet4|tet10/numerical_kernel,
//   electrical_conduction/tet4|tet10/numerical_kernel;
//   check: layered-slab; boundary: Both orders, aligned two-material slab series capacitance/resistance and interface
//   flux; no curved interface claim.
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
            let topo = &mut *mesh.topology;
            for element in &mut topo.volume_elements {
                let x = element.node_ids[..4]
                    .iter()
                    .map(|id| {
                        topo.nodes
                            .iter()
                            .find(|node| node.node_id == *id)
                            .unwrap()
                            .coordinates_m
                            .canonical_value()[0]
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
            electrical.conductivity_s_per_m = runmat_types::physical::Quantity::new(
                electrical.conductivity_s_per_m.canonical_value() * (4.0),
            );
            electrical.relative_permittivity = runmat_types::physical::Quantity::new(
                electrical.relative_permittivity.canonical_value() * (4.0),
            );
            model.materials.push(second);
            model.material_assignments[0].target = PropertyRegionTarget::solid(
                runmat_meshing_core::fixtures::fixture_solid("layer_left"),
            );
            let mut second = model.material_assignments[0].clone();
            second.target = PropertyRegionTarget::solid(
                runmat_meshing_core::fixtures::fixture_solid("layer_right"),
            );
            second.expected_material_id = "second".into();
            second.assigned_material_id = "second".into();
            model.material_assignments.push(second);
            prescribe(
                &mut model,
                "left".into(),
                RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
                0.0,
            );
            prescribe(
                &mut model,
                "right".into(),
                RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
                1.0,
            );
            let response = solve(&model, &mesh);
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            for (node, &potential) in projection
                .nodes
                .iter()
                .zip(response.potential_v.canonical_values())
            {
                let x = node.coordinates_m.canonical_value()[0];
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
                    assert_relative(total_energy_j.canonical_value(), 0.8 * a, 1e-8);
                    displacement_c_per_m2.canonical_values()
                }
                ElectricalConstitutiveResponse::SteadyConduction {
                    current_density_a_per_m2,
                    total_joule_power_w,
                    ..
                } => {
                    assert_relative(total_joule_power_w.canonical_value(), 1.6 * a, 1e-8);
                    current_density_a_per_m2.canonical_values()
                }
            };
            for value in flux {
                assert_relative(value[0], -1.6 * a, 1e-8);
                assert!(value[1].abs() < a * 1e-8 && value[2].abs() < a * 1e-8);
            }
            assert_relative(
                response.terminals[0].outward_flux.canonical_value(),
                1.6 * a,
                1e-8,
            );
            assert_relative(
                response.terminals[1].outward_flux.canonical_value(),
                -1.6 * a,
                1e-8,
            );
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
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
            1e6,
        );
        let response = solve(&model, &mesh);
        assert!(response
            .potential_v
            .canonical_values()
            .iter()
            .all(|value| *value == 1e6));
        assert!(response
            .element_electric_field_v_per_m
            .canonical_values()
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
        assert!(response
            .nodal_outward_flux
            .canonical_values()
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(response.relative_energy_identity_error, 0.0);
        assert_eq!(response.relative_flux_imbalance, 0.0);
    }
}

// witness: FEA-EM-QUALIFICATION-001/references @ electrostatic/tet4|tet10/numerical_kernel,
//   electrical_conduction/tet4|tet10/numerical_kernel;
//   check: harmonic-polynomial-potential; boundary: Static harmonic polynomial potential x²-y² for both scalar equations: Tet10 exact
//   field/energy and Tet4 refinement across three meshes; harmonic polynomial is not harmonic
//   Maxwell.
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
                    let [x, y, _] = node.coordinates_m.canonical_value();
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
                                        * projection.nodes[node].coordinates_m.canonical_value()
                                            [axis];
                                    field[axis] -= point.physical_shape_gradients()[local][axis]
                                        * response.potential_v.canonical_values()[node];
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
                        } => (total_energy_j.canonical_value(), 0.5),
                        ElectricalConstitutiveResponse::SteadyConduction {
                            total_joule_power_w,
                            ..
                        } => (total_joule_power_w.canonical_value(), 1.0),
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

// witness: FEA-EM-INPUT-001/scalar_constraints @ electrical_conduction/tet4/numerical_kernel;
//   boundary: Tet4 rejects missing anchor, conflicting potentials, floating conductor, bad
//   material, target and wrong harmonic intent; rejection is not implementation of floating
//   conductors.
#[test]
fn missing_anchors_conflicts_unsupported_intent_and_invalid_material_fail_closed() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    assert!(solve_canonical_static(&model, &mesh, "potential", 1e-8).is_err());
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    prescribe(
        &mut model,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        1.0,
    );
    let mut bad = model.clone();
    bad.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = runmat_types::physical::ElectricalConductivity::new(0.0);
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
    let mut bad = model.clone();
    prescribe(
        &mut bad,
        "conflict".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        1.0,
    );
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
    let mut bad = model.clone();
    bad.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(60.0),
    });
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
    let mut bad = model.clone();
    bad.constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "floating".into(),
            kind: ElectricalConstraintKind::FloatingConductor {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
            },
        }));
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
    let mut bad = model.clone();
    prescribe(
        &mut bad,
        "missing".into(),
        RegionTarget::point("node:999999"),
        1.0,
    );
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
    let mut bad = model.clone();
    bad.sequence.steps.push(bad.sequence.steps[0].clone());
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
    let mut bad = model.clone();
    bad.material_assignments.clear();
    assert!(solve_canonical_static(&bad, &mesh, "potential", 1e-8).is_err());
}

// witness: FEA-EM-INPUTS-002/scalar_potential_inputs @
//   electrostatic|electrical_conduction/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: A terminal_voltage load holds its terminal like a prescribed potential for
//   electrostatics and steady conduction on two-cell Tet4 and Tet10 slabs; terminal current,
//   charge, equipotential and floating inputs are separate tests.
// witness: FEA-EM-INPUT-001/excitation @ electrostatic|electrical_conduction/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   boundary: A constant terminal-voltage signal on a face reproduces the prescribed-potential
//   field exactly on Tet4 and Tet10 slabs; no time history or phasor.
#[test]
fn terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential_on(order);
    }
}

fn terminal_voltage_load_holds_its_terminal_like_a_prescribed_potential_on(order: ElementOrder) {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        let mesh = mesh(order, 2);
        let mut constrained = model(equation);
        prescribe(
            &mut constrained,
            "left".into(),
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
            0.0,
        );
        let mut driven = constrained.clone();
        prescribe(
            &mut constrained,
            "right".into(),
            RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
            2.5,
        );
        driven
            .signals
            .push(PhysicalSignal::ElectricPotential(Signal {
                signal_id: SignalId("supply".into()),
                shape: SignalShape::Scalar,
                frame: None,
                representation: SignalRepresentation::Constant(SignalValue::Scalar(
                    ElectricPotential::new(2.5),
                )),
                precision: SignalPrecision::Real64,
                provenance: SignalProvenance::Authored,
                evidence: SignalApproximationEvidence {
                    max_absolute_error: Some(0.0),
                    max_relative_error: Some(0.0),
                    derivative_max_absolute_error: None,
                    sample_count: 1,
                },
            }));
        driven
            .excitations
            .push(ExcitationAction::TerminalVoltage(AnalysisExcitation {
                excitation_id: ExcitationId("supply".into()),
                target: ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("slab_right"),
                )),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::PreserveResultant,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("supply".into()),
                active_step_ids: vec!["potential".into()],
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            }));
        driven.excitation_cases.push(ExcitationCase {
            case_id: ExcitationCaseId("supply".into()),
            action_ids: vec![ExcitationId("supply".into())],
            activation: ExcitationCaseActivation::Always,
        });
        let expected = solve(&constrained, &mesh);
        let actual = solve(&driven, &mesh);
        for (a, b) in actual
            .potential_v
            .canonical_values()
            .iter()
            .zip(expected.potential_v.canonical_values())
        {
            assert!((a - b).abs() <= 1e-12 * 2.5, "{a} vs {b}");
        }
        assert_relative(
            actual.terminal_work.canonical_value(),
            expected.terminal_work.canonical_value(),
            1e-10,
        );
    }
}

// witness: FEA-EM-INPUT-001/scalar_constraints @ electrical_conduction|electrostatic/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   boundary: Integral-current (conduction) and uncharged floating (electrostatic) terminals
//   reach the common potential and current balance of Tet4 and Tet10 slabs; no charged
//   floating conductor.
// witness: FEA-EM-INPUTS-002/scalar_potential_inputs @
//   electrostatic|electrical_conduction/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Electric current terminal and floating inputs close the terminal current balance
//   on Tet4 and Tet10 slabs; no charged floating conductor or equipotential set.
#[test]
fn integral_current_and_floating_terminals_find_their_common_potential() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        integral_current_and_floating_terminals_find_their_common_potential_on(order);
    }
}

fn integral_current_and_floating_terminals_find_their_common_potential_on(order: ElementOrder) {
    let mesh = mesh(order, 2);
    // Reference: 1 V across the slab gives the terminal current I1, so R = 1/I1.
    let mut reference = model(StaticElectricalEquation::SteadyConduction);
    prescribe(
        &mut reference,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    prescribe(
        &mut reference,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        1.0,
    );
    let unit = solve(&reference, &mesh);
    let i1 = unit
        .terminals
        .iter()
        .find(|t| t.constraint_id == "right")
        .unwrap()
        .outward_flux
        .canonical_value()
        .abs();
    // A 2.5 A integral current into the right terminal holds it at 2.5 R.
    let mut driven = model(StaticElectricalEquation::SteadyConduction);
    prescribe(
        &mut driven,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    driven
        .constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "right".into(),
            kind: ElectricalConstraintKind::IntegralCurrent {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
                value: PrescribedValue::Constant(ElectricCurrent::new(2.5)),
            },
        }));
    let response = solve(&driven, &mesh);
    let terminal = response
        .terminals
        .iter()
        .find(|t| t.constraint_id == "right")
        .unwrap();
    assert_relative(-terminal.outward_flux.canonical_value(), 2.5, 1e-8);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let right_potentials: Vec<f64> = terminal
        .ordered_node_ids
        .iter()
        .map(|id| {
            let index = projection
                .nodes
                .iter()
                .position(|n| n.node_id == *id)
                .unwrap();
            response.potential_v.canonical_values()[index]
        })
        .collect();
    for potential in right_potentials {
        assert_relative(potential, 2.5 / i1, 1e-8);
    }
    // An uncharged floating electrode carries no net flux.
    let mut floating = model(StaticElectricalEquation::Electrostatics);
    prescribe(
        &mut floating,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        3.0,
    );
    floating
        .constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "right".into(),
            kind: ElectricalConstraintKind::FloatingConductor {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_right",
                )),
            },
        }));
    let response = solve(&floating, &mesh);
    let terminal = response
        .terminals
        .iter()
        .find(|t| t.constraint_id == "right")
        .unwrap();
    assert!(
        terminal.outward_flux.canonical_value().abs()
            <= 1e-9 * 3.0 * coefficient(StaticElectricalEquation::Electrostatics)
    );
    // With no net flux the whole slab sits at the anchor potential.
    assert!(response
        .potential_v
        .canonical_values()
        .iter()
        .all(|v| (v - 3.0).abs() < 1e-8));
}

// witness: FEA-EM-INPUT-001/excitation @ electrical_conduction/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   boundary: A face current density realizes its integrated normal current as the grounded
//   terminal's outward flux on Tet4 and Tet10 slabs; no time history.
#[test]
fn face_current_density_injects_its_normal_flux() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        face_current_density_injects_its_normal_flux_on(order);
    }
}

fn face_current_density_injects_its_normal_flux_on(order: ElementOrder) {
    let mesh = mesh(order, 2);
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    model
        .signals
        .push(PhysicalSignal::ElectricCurrentDensity(Signal {
            signal_id: SignalId("j".into()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Vector3([
                ElectricCurrentDensity::new(-4.0),
                ElectricCurrentDensity::new(0.0),
                ElectricCurrentDensity::new(0.0),
            ])),
            precision: SignalPrecision::Real64,
            provenance: SignalProvenance::Authored,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
        }));
    model
        .excitations
        .push(ExcitationAction::CurrentDensity(AnalysisExcitation {
            excitation_id: ExcitationId("j".into()),
            target: ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_right"),
            )),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerArea,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("j".into()),
            active_step_ids: vec!["potential".into()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-12,
            },
        }));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("j".into()),
        action_ids: vec![ExcitationId("j".into())],
        activation: ExcitationCaseActivation::Always,
    });
    let response = solve(&model, &mesh);
    // Flux into the slab through the right face equals the current through the grounded left face.
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let area: f64 = projection
        .boundary_faces_for_target(&RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_right"),
        ))
        .map(|face| face.area_m2.canonical_value())
        .sum();
    let left = response
        .terminals
        .iter()
        .find(|t| t.constraint_id == "left")
        .unwrap();
    assert_relative(left.outward_flux.canonical_value(), 4.0 * area, 1e-8);
}

// witness: FEA-EM-INPUT-001/scalar_constraints @ electrostatic/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   boundary: A charged slab with a translation-periodic right face matches the same slab grounded
//   on both faces to 1e-8 on Tet4 and Tet10, and the grounded face carries the full flux; terminal
//   voltage, floating and integral-current constraints are separate tests.
#[test]
fn periodic_faces_share_their_potential() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        periodic_faces_share_their_potential_on(order);
    }
}

fn periodic_faces_share_their_potential_on(order: ElementOrder) {
    // A charged slab grounded on both faces, against the same slab grounded on the left and
    // periodic (by translation) on the right: the right face carries the left face's potential,
    // so the fields agree.
    let mesh = mesh(order, 2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mean_x = |target: &str| {
        let xs: Vec<f64> = projection
            .boundary_faces_for_target(&RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(target)),
            ))
            .flat_map(|face| {
                face.node_indices
                    .iter()
                    .map(|&n| projection.nodes[n].coordinates_m.canonical_value()[0])
            })
            .collect();
        xs.iter().sum::<f64>() / xs.len() as f64
    };
    let shift = mean_x("slab_right") - mean_x("slab_left");
    let both = charge_density::charged_model(1e-6, 0.0);
    let mut periodic = both.clone();
    periodic
        .constraints
        .retain(|c| c.constraint_id() != "right");
    periodic
        .constraints
        .push(AnalysisConstraint::Electrical(ElectricalConstraint {
            constraint_id: "periodic".into(),
            kind: ElectricalConstraintKind::Periodic(PeriodicConstraint {
                pair: PairedConstraint {
                    primary: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_left",
                    )),
                    secondary: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "slab_right",
                    )),
                    pairing: PairingPolicy::GeometricTolerance(Length::new(1e-9)),
                },
                transform: PeriodicTransform::Translation(Translation::new([shift, 0.0, 0.0])),
            }),
        }));
    let expected = solve(&both, &mesh);
    let actual = solve(&periodic, &mesh);
    let scale = expected
        .potential_v
        .canonical_values()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0);
    for (a, b) in actual
        .potential_v
        .canonical_values()
        .iter()
        .zip(expected.potential_v.canonical_values())
    {
        assert!((a - b).abs() <= 1e-8 * scale, "{a} vs {b}");
    }
    // All the charge's flux leaves through the grounded face once the periodic face's share
    // is passed on to it.
    let left = actual
        .terminals
        .iter()
        .find(|t| t.constraint_id == "left")
        .unwrap();
    let total: f64 = expected
        .terminals
        .iter()
        .map(|t| t.outward_flux.canonical_value())
        .sum();
    assert!((left.outward_flux.canonical_value() - total).abs() <= 1e-8 * total.abs());
}

// witness: FEA-EM-QUALIFICATION-001/references @ electrical_conduction/tet10/numerical_kernel; coverage: partial;
//   check: copper-bar-current; boundary: The validation table's steady-current check: copper (59.6 MS/m) with 1 mV across a
//   1 m bar gives the uniform current density sigma V / L, the Joule power V^2 / R from the loss
//   density and the terminal current, on a Tet10 slab; one geometry, no curved conductors.
#[test]
fn copper_bar_carries_sigma_v_over_l_and_dissipates_v_squared_over_r() {
    const SIGMA: f64 = 59.6e6;
    const VOLTS: f64 = 1e-3;
    let mesh = mesh(ElementOrder::Tet10, 2);
    let mut model = model(StaticElectricalEquation::SteadyConduction);
    model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = runmat_types::physical::Quantity::new(SIGMA);
    prescribe(
        &mut model,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    prescribe(
        &mut model,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        VOLTS,
    );
    let response = solve(&model, &mesh);
    // Unit cube: L = 1 m, A = 1 m^2, R = L / (sigma A).
    let ElectricalConstitutiveResponse::SteadyConduction {
        current_density_a_per_m2,
        total_joule_power_w,
        ..
    } = &response.constitutive
    else {
        panic!("steady conduction response expected");
    };
    let expected_j = SIGMA * VOLTS;
    for density in current_density_a_per_m2.canonical_values() {
        assert!(
            (density[0].abs() - expected_j).abs() <= 1e-9 * expected_j,
            "{density:?}"
        );
        assert!(density[1].abs() + density[2].abs() <= 1e-9 * expected_j);
    }
    let expected_power = VOLTS * VOLTS * SIGMA;
    assert!(
        (total_joule_power_w.canonical_value() - expected_power).abs() <= 1e-9 * expected_power
    );
    let current = response
        .terminals
        .iter()
        .find(|terminal| terminal.constraint_id == "right")
        .unwrap()
        .outward_flux
        .canonical_value()
        .abs();
    assert!((current * VOLTS - expected_power).abs() <= 1e-9 * expected_power);
    assert_eq!(
        response.operator_proof,
        ElectricalOperatorProof::LocalEnergy
    );
}
