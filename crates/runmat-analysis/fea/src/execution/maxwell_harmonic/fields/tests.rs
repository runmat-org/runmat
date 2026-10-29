use super::*;
use crate::results::field::{SampledVolumeFieldTopology, VolumeFieldSamplingRule};
use runmat_analysis_core::*;
use runmat_meshing_core::fixtures;

fn workload(phase: [f64; 2]) -> MaxwellHarmonicWorkload {
    let frequency = 1e8;
    let mut model = crate::physics::magnetic::test_support::authored_model(1.0);
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(frequency),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("potential");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    model.sequence = AnalysisSequence::from_steps("maxwell", vec![step]);
    model.constraints.clear();
    for boundary in ["slab_left", "slab_right", "slab_insulated"] {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: boundary.into(),
                kind: ElectricalConstraintKind::ElectricPerfectConductor {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(boundary)),
                    ),
                },
            }));
    }
    let PhysicalSignal::ElectricCurrentDensity(signal) = &mut model.signals[0] else {
        panic!("fixture must retain authored current density")
    };
    let vector = |value| SignalValue::Vector3([0.0, value, 0.0].map(ElectricCurrentDensity::new));
    signal.precision = SignalPrecision::Complex64;
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![SignalKnot::continuous(
            frequency,
            SignalOrdinate::Complex {
                real: vector(phase[0]),
                imaginary: vector(phase[1]),
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    MaxwellHarmonicWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
        frequency: runmat_analysis_core::Frequency::new(1e8),
    }
}

#[test]
fn both_orders_cold_native_fields_preserve_exact_complex_phase_and_sample_authority() {
    for mesh in [fixtures::slab_mesh(2), fixtures::quadratic_slab_mesh(2)] {
        let w = workload([0.6, 0.8]);
        let result = w.execute(&mesh).unwrap();
        let original_sampling = result.sampling_topology().clone();
        let expected = native_maxwell_fields(&result)
            .map(|f| (f.field_id, f.shape, f.values.values().collect::<Vec<_>>()));
        assert!(expected[1]
            .2
            .chunks_exact(2)
            .any(|v| v[0] != 0.0 && v[1] != 0.0));
        let request = result.native_field_request(&w, &mesh).unwrap();
        let bytes = result.canonical_bytes().unwrap();
        drop(result);
        let retained = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &w,
            &mesh,
        )
        .unwrap();
        assert_eq!(retained.sampling_topology(), &original_sampling);
        assert_eq!(
            original_sampling.rule(),
            VolumeFieldSamplingRule::TetrahedronKeastFifteen
        );
        for (original, sample) in original_sampling
            .integration_samples_for(&mesh)
            .unwrap()
            .iter()
            .zip(&retained.response().samples)
        {
            assert_eq!(original.identity, sample.sample_identity);
            assert_eq!(original.element_identity, sample.element_identity);
            assert_eq!(original.ordinal, sample.ordinal);
            assert_eq!(original.position_m, sample.position_m);
            assert_eq!(original.volume_weight_m3, sample.volume_weight_m3);
        }
        let fields = retained.recover_native_fields(&w, &mesh, &request).unwrap();
        let edge_values = retained
            .response()
            .edge_circulations_v
            .canonical_values()
            .iter()
            .flat_map(|value| [value.real, value.imaginary])
            .collect::<Vec<_>>();
        assert_eq!(
            fields.fields()[0].storage().as_host_f64().unwrap(),
            edge_values
        );
        for index in 1..6 {
            let original = retained
                .response()
                .samples
                .iter()
                .flat_map(|sample| {
                    match index {
                        1 => sample.electric_field_v_per_m.canonical_value(),
                        2 => sample.flux_density_t.canonical_value(),
                        3 => sample.magnetic_field_a_per_m.canonical_value(),
                        4 => sample.electric_displacement_c_per_m2.canonical_value(),
                        5 => sample.conduction_current_density_a_per_m2.canonical_value(),
                        _ => unreachable!(),
                    }
                    .into_iter()
                    .flat_map(|value| [value.real, value.imaginary])
                })
                .collect::<Vec<_>>();
            assert_eq!(
                fields.fields()[index].storage().as_host_f64().unwrap(),
                original
            );
        }
        assert_eq!(
            fields.entity_counts.volume_integration_points,
            mesh.topology.volume_elements.len() * 15
        );
        assert_eq!(
            fields.entity_counts.oriented_edges,
            retained
                .response()
                .edge_circulations_v
                .canonical_values()
                .len()
        );
        assert_eq!(
            fields.topology_digest,
            request.topology.discretization_digest()
        );
        for (field, (name, shape, values)) in fields.fields().iter().zip(expected) {
            assert_eq!(field.storage().field_id, name);
            assert_eq!(field.storage().shape, shape);
            assert_eq!(field.storage().as_host_f64().unwrap(), values);
            // Cycle-averaged loss densities are real; every other native field is a phasor.
            let expected = if name.contains("joule") {
                RecoveredFieldNumericRepresentation::Real
            } else {
                RecoveredFieldNumericRepresentation::ComplexInterleaved
            };
            assert_eq!(field.definition().numeric_representation(), expected);
            assert_eq!(field.sample(), RecoveredFieldSample::Final);
        }
        assert_eq!(
            request.plan.requirements[0].quantity,
            ObservationQuantity::ElectricEdgeCirculation
        );
        assert_eq!(
            request.plan.requirements[0].value_shape,
            ObservationValueShape::Scalar
        );
        assert_eq!(
            fields.fields()[0].storage().shape,
            vec![fields.entity_counts.oriented_edges, 2]
        );
        let wire = fields.canonical_bytes().unwrap();
        assert_eq!(
            fields,
            RecoveredFieldArtifact::from_canonical_bytes(&wire).unwrap()
        );
    }
}

#[test]
fn native_inventory_rejects_source_rule_domain_shape_and_complex_count_substitution() {
    let mesh = fixtures::slab_mesh(2);
    let w = workload([0.6, 0.8]);
    let result = w.execute(&mesh).unwrap();
    let request = result.native_field_request(&w, &mesh).unwrap();
    let mut foreign = request.clone();
    foreign.topology =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronFourPoint)
            .unwrap()
            .identity()
            .unwrap();
    assert!(result.recover_native_fields_from_result(&foreign).is_err());
    let different_phase = workload([-0.8, 0.6]);
    let foreign_result = different_phase.execute(&mesh).unwrap();
    let foreign_request = foreign_result.native_field_request_from_result().unwrap();
    assert!(result
        .recover_native_fields_from_result(&foreign_request)
        .is_err());
    assert!(result
        .recover_native_fields(&different_phase, &mesh, &request)
        .is_err());
    assert!(result
        .recover_native_fields(&w, &fixtures::slab_mesh(3), &request)
        .is_err());
    let bytes = foreign_result.canonical_bytes().unwrap();
    assert!(MaxwellHarmonicSolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(result.canonical_bytes().unwrap()),
        &different_phase,
        &mesh
    )
    .is_err());
    for change in 0..6 {
        let mut changed = request.clone();
        match change {
            0 => {
                changed.plan.requirements.pop();
            }
            1 => changed.plan.requirements.swap(1, 2),
            2 => changed.plan.requirements[0].value_shape = ObservationValueShape::Vector3,
            3 => {
                changed.plan.requirements[0].entity_domain =
                    RecoveredFieldEntityDomain::VolumeIntegrationPoints
            }
            4 => changed.plan.requirements[1].cost.arithmetic_operation_count /= 2,
            5 => changed.plan.requirements[1].cost.source_entity_count += 1,
            _ => unreachable!(),
        }
        assert!(
            maxwell_native_field_definitions(&changed).is_err(),
            "substitution {change}"
        );
        assert!(result.recover_native_fields_from_result(&changed).is_err());
    }
}

#[test]
fn cold_result_rejects_sample_and_constitutive_phase_corruption() {
    let mesh = fixtures::slab_mesh(2);
    let w = workload([0.6, 0.8]);
    let result = w.execute(&mesh).unwrap();
    for field in ["position_m", "magnetic_field_a_per_m"] {
        let mut value = serde_json::to_value(&result).unwrap();
        let slot = if field == "position_m" {
            &mut value["response"]["samples"][0][field][0]
        } else {
            &mut value["response"]["samples"][0][field][0]["imaginary"]
        };
        *slot = serde_json::json!(slot.as_f64().unwrap() + 1.0);
        // Sample geometry is derived from the mesh, so a moved sample is rejected against it.
        let decoded = serde_json::from_value::<MaxwellHarmonicSolverResult>(value);
        assert!(
            decoded.is_err() || decoded.unwrap().validate_for(&w, &mesh).is_err(),
            "accepted corrupt {field}"
        );
    }
}
