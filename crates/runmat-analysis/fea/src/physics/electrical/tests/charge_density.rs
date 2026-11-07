//! Analytic charged slab: -epsilon phi''=rho, phi(0)=phi(1)=g.
use super::*;

pub(crate) fn charged_model(charge: f64, gauge: f64) -> AnalysisModel {
    let mut input = model(StaticElectricalEquation::Electrostatics);
    prescribe(
        &mut input,
        "left".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        gauge,
    );
    prescribe(
        &mut input,
        "right".into(),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        gauge,
    );
    input.signals = vec![PhysicalSignal::ElectricChargeDensity(Signal {
        signal_id: SignalId("rho".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(
            ElectricChargeDensity::new(charge),
        )),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })];
    input.excitations = vec![ExcitationAction::ElectricChargeDensity(
        AnalysisExcitation {
            excitation_id: ExcitationId("volume-charge".into()),
            target: ExcitationTarget::WholeModel,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerVolume,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("rho".into()),
            active_step_ids: vec!["potential".into()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-12,
            },
        },
    )];
    input.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("charge".into()),
        action_ids: vec![ExcitationId("volume-charge".into())],
        activation: ExcitationCaseActivation::Always,
    }];
    input
}

#[test]
fn tet10_charged_slab_matches_potential_field_charge_and_energy_with_both_signs_and_gauges() {
    let epsilon = coefficient(StaticElectricalEquation::Electrostatics);
    let mesh = mesh(ElementOrder::Tet10, 2);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    for sign in [-1.0, 1.0] {
        for gauge in [0.0, 1000.0, 1e12] {
            let response = solve(&charged_model(sign * 2.0 * epsilon, gauge), &mesh);
            for (node, departure) in projection
                .nodes
                .iter()
                .zip(response.potential_increment_v.canonical_values())
            {
                let x = node.coordinates_m.canonical_value()[0];
                assert!((departure - sign * x * (1.0 - x)).abs() < 1e-9);
            }
            for (element, field) in projection
                .elements
                .iter()
                .zip(response.element_electric_field_v_per_m.canonical_values())
            {
                assert!((field[0] - sign * (2.0 * element.centroid_m[0] - 1.0)).abs() < 1e-9);
                assert!(field[1].abs() < 1e-9 && field[2].abs() < 1e-9);
            }
            let source = response.charge_sources.as_ref().unwrap();
            assert_relative(source.integrated_source, sign * 2.0 * epsilon, 1e-10);
            assert!(source.projection_evidence.iter().all(|e| e.accepted));
            for terminal in &response.terminals {
                assert_relative(
                    terminal.outward_flux.canonical_value(),
                    sign * epsilon,
                    1e-9,
                );
            }
            let ElectricalConstitutiveResponse::Electrostatics { total_energy_j, .. } =
                response.constitutive
            else {
                unreachable!()
            };
            assert_relative(total_energy_j.canonical_value(), epsilon / 6.0, 1e-9);
            assert_relative(
                response.volume_source_work.canonical_value(),
                epsilon / 3.0,
                1e-9,
            );
            assert_eq!(response.terminal_work.canonical_value(), 0.0);
            assert!(response.relative_flux_imbalance < 1e-9);
            assert!(response.relative_energy_identity_error < 1e-9);
        }
    }
}

// witness: FEA-EM-QUALIFICATION-001/references @ electrostatic/tet4/numerical_kernel;
//   check: charged-slab; boundary: Uniform-charge Tet4 slab on three meshes: independent quadratic potential, linear
//   field, charge conservation and energy convergence; no Tet10 or nonlinear dielectric claim.
#[test]
fn tet4_charged_slab_has_independent_spatial_field_and_energy_convergence() {
    let epsilon = coefficient(StaticElectricalEquation::Electrostatics);
    let input = charged_model(2.0 * epsilon, 0.0);
    let mut errors = Vec::new();
    for cells in [2, 4, 8] {
        let mesh = mesh(ElementOrder::Tet4, cells);
        let response = solve(&input, &mesh);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let mut temperature_error = 0.0;
        let mut field_error = 0.0;
        for element in &projection.elements {
            projection
                .visit_element_quadrature(element, |point| {
                    let x = point.position_m[0];
                    let potential = element
                        .node_indices
                        .iter()
                        .zip(point.shape_values())
                        .map(|(&node, &shape)| {
                            shape * response.potential_increment_v.canonical_values()[node]
                        })
                        .sum::<f64>();
                    let field = (0..3)
                        .map(|axis| {
                            -element
                                .node_indices
                                .iter()
                                .zip(point.physical_shape_gradients())
                                .map(|(&node, gradient)| {
                                    response.potential_increment_v.canonical_values()[node]
                                        * gradient[axis]
                                })
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    temperature_error +=
                        point.volume_weight_m3 * (potential - x * (1.0 - x)).powi(2);
                    field_error += point.volume_weight_m3
                        * ((field[0] - (2.0 * x - 1.0)).powi(2)
                            + field[1].powi(2)
                            + field[2].powi(2));
                })
                .unwrap();
        }
        let ElectricalConstitutiveResponse::Electrostatics { total_energy_j, .. } =
            response.constitutive
        else {
            unreachable!()
        };
        let energy_error = (total_energy_j.canonical_value() / (epsilon / 6.0) - 1.0).abs();
        errors.push((temperature_error.sqrt(), field_error.sqrt(), energy_error));
        assert_relative(
            response.charge_sources.as_ref().unwrap().integrated_source,
            2.0 * epsilon,
            1e-10,
        );
        assert!(response.relative_flux_imbalance < 1e-9);
    }
    println!("charged-slab Tet4 errors: {errors:?}");
    for pair in errors.windows(2) {
        assert!((pair[0].0 / pair[1].0).log2() > 1.7);
        assert!((pair[0].1 / pair[1].1).log2() > 0.8);
        assert!(pair[1].2 < 0.35 * pair[0].2);
    }
    let finest = errors.last().unwrap();
    assert!(finest.0 < 0.005 && finest.1 < 0.1 && finest.2 < 0.025);
}

// witness: FEA-EM-INPUT-001/excitation @ electrostatic/tet4|tet10/numerical_kernel;
//   boundary: Both orders, constant real volume-charge exact target and signed superposition
//   preserve potential, net/absolute charge and two projection records; no generated or frequency
//   artifact equivalence.
#[test]
fn exact_volume_region_and_signed_superposition_preserve_charge_and_fields() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let epsilon = coefficient(StaticElectricalEquation::Electrostatics);
        let baseline = solve(&charged_model(2.0 * epsilon, 0.0), &mesh);
        let region = mesh.topology.volume_elements[0].region_id.clone();
        assert!(mesh
            .topology
            .volume_elements
            .iter()
            .all(|e| e.region_id == region));
        let mut input = charged_model(3.0 * epsilon, 0.0);
        let mut target = RegionTarget::volume(runmat_meshing_core::fixtures::fixture_region(
            "resolved-charge-volume",
        ));
        target.bind_resolved_entities(vec![region]).unwrap();
        input.excitations[0].excitation_mut().target = ExcitationTarget::Region(target);
        let mut negative = charged_model(-epsilon, 0.0);
        let PhysicalSignal::ElectricChargeDensity(signal) = &mut negative.signals[0] else {
            unreachable!()
        };
        signal.signal_id = SignalId("negative-rho".into());
        let action = negative.excitations[0].excitation_mut();
        action.excitation_id = ExcitationId("negative-charge".into());
        action.signal_id = SignalId("negative-rho".into());
        input.signals.extend(negative.signals);
        input.excitations.extend(negative.excitations);
        input.excitation_cases[0]
            .action_ids
            .push(ExcitationId("negative-charge".into()));
        let actual = solve(&input, &mesh);
        for (a, b) in actual
            .potential_increment_v
            .canonical_values()
            .iter()
            .zip(baseline.potential_increment_v.canonical_values())
        {
            assert!((a - b).abs() < 1e-9);
        }
        let source = actual.charge_sources.as_ref().unwrap();
        assert_relative(source.integrated_source, 2.0 * epsilon, 1e-10);
        assert_relative(source.absolute_source, 4.0 * epsilon, 1e-10);
        assert_eq!(source.projection_evidence.len(), 2);
    }
}

#[test]
fn charge_sources_reject_conduction_unmatched_region_complex_and_wrong_target() {
    let mesh = mesh(ElementOrder::Tet10, 1);
    let input = charged_model(1.0, 0.0);
    let mut wrong = input.clone();
    wrong.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::SteadyConduction,
    });
    assert!(solve_canonical_static(&wrong, &mesh, "potential", 1e-11).is_err());
    for target in [
        RegionTarget::volume(runmat_meshing_core::fixtures::fixture_region("missing")),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
    ] {
        let mut wrong = input.clone();
        wrong.excitations[0].excitation_mut().target = ExcitationTarget::Region(target);
        assert!(solve_canonical_static(&wrong, &mesh, "potential", 1e-11).is_err());
    }
    let mut wrong = input;
    let PhysicalSignal::ElectricChargeDensity(signal) = &mut wrong.signals[0] else {
        unreachable!()
    };
    signal.precision = SignalPrecision::Complex64;
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![SignalKnot::continuous(
            1.0,
            SignalOrdinate::Complex {
                real: SignalValue::Scalar(ElectricChargeDensity::new(1.0)),
                imaginary: SignalValue::Scalar(ElectricChargeDensity::new(1.0)),
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    assert!(solve_canonical_static(&wrong, &mesh, "potential", 1e-11).is_err());
}
