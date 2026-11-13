use super::*;
use crate::{
    AuthoredSignalFieldLocation, AuthoredSignalTopologySelector, CoordinateFrame, Temperature,
};

fn declaration(width: usize, location: AuthoredSignalFieldLocation) -> Signal<Temperature> {
    Signal {
        signal_id: SignalId("temperature_field".into()),
        shape: SignalShape::DeferredTopologyField {
            selector: AuthoredSignalTopologySelector::StudySolverMesh { location },
            component_count: width,
        },
        frame: None,
        representation: SignalRepresentation::Generator(SignalGenerator {
            invocation: GeneratorInvocation {
                callable: GeneratorCallable::ProjectFunction {
                    qualified_name: "fields.temperature".into(),
                },
                arguments: Vec::new(),
                effects: GeneratorEffectPolicy::Pure,
            },
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 2,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    }
}

#[test]
fn deferred_domains_are_closed_and_never_numerically_counted() {
    for location in [
        AuthoredSignalFieldLocation::Node,
        AuthoredSignalFieldLocation::Cell,
        AuthoredSignalFieldLocation::Element,
        AuthoredSignalFieldLocation::Face,
        AuthoredSignalFieldLocation::Edge,
    ] {
        let signal = declaration(1, location);
        validate_signal(&signal, None).unwrap();
        assert_eq!(signal.shape.component_extents(), Some(vec![1]));
        for count in [None, Some(1), Some(17), Some(usize::MAX)] {
            assert_eq!(signal.shape.realized_component_count(count), None);
        }
        let bytes = serde_json::to_vec(&signal).unwrap();
        let recovered: Signal<Temperature> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(recovered, signal);
        validate_signal(&recovered, None).unwrap();
    }
    let mut vector = declaration(3, AuthoredSignalFieldLocation::Node);
    assert!(validate_signal(&vector, None).is_err());
    vector.frame = Some(CoordinateFrame::Global);
    validate_signal(&vector, None).unwrap();
    let zero = declaration(0, AuthoredSignalFieldLocation::Node);
    assert!(validate_signal(&zero, None).is_err());
    assert_eq!(zero.shape.component_extents(), None);
    let huge = declaration(usize::MAX, AuthoredSignalFieldLocation::Node);
    assert_eq!(huge.shape.component_extents(), Some(vec![usize::MAX]));
    assert_eq!(huge.shape.realized_component_count(Some(2)), None);
    vector.precision = SignalPrecision::Complex128;
    validate_signal(&vector, None).unwrap();
}

#[test]
fn deferred_selector_rejects_unknown_owners_locations_and_fields() {
    let shape = declaration(3, AuthoredSignalFieldLocation::Node).shape;
    let original = serde_json::to_value(&shape).unwrap();
    for location in ["integration_point", "point", "global", "arbitrary"] {
        let mut value = original.clone();
        value["deferred_topology_field"]["selector"]["study_solver_mesh"]["location"] =
            serde_json::json!(location);
        assert!(serde_json::from_value::<SignalShape>(value).is_err());
    }
    let mut value = original.clone();
    value["deferred_topology_field"]["selector"] =
        serde_json::json!({"named_mesh": {"location": "node"}});
    assert!(serde_json::from_value::<SignalShape>(value).is_err());
    let mut value = original.clone();
    value["deferred_topology_field"]["selector"]["study_solver_mesh"]["topology_id"] =
        serde_json::json!("not-an-authority");
    assert!(serde_json::from_value::<SignalShape>(value).is_err());
    let mut value = original;
    value["deferred_topology_field"]["topology_id"] = serde_json::json!("not-an-authority");
    assert!(serde_json::from_value::<SignalShape>(value).is_err());
}

#[test]
fn deferred_shape_requires_authored_generator_not_concrete_values() {
    let original = declaration(1, AuthoredSignalFieldLocation::Node);
    let mut signal = original.clone();
    signal.representation =
        SignalRepresentation::Constant(SignalValue::Scalar(Temperature::new(300.0)));
    assert!(validate_signal(&signal, None).is_err());
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![SignalKnot::continuous(
            0.0,
            SignalOrdinate::Real(SignalValue::Tensor {
                shape: vec![2, 1],
                values: vec![Temperature::new(300.0); 2],
            }),
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    assert!(validate_signal(&signal, None).is_err());
    signal.representation = SignalRepresentation::Waveform(StandardWaveform {
        kind: StandardWaveformKind::Step,
        amplitude: SignalValue::Scalar(Temperature::new(300.0)),
        offset: None,
        start_seconds: runmat_types::physical::Time::new(0.0),
        duration_seconds: (None).map(runmat_types::physical::Time::new),
        frequency_hz: (None).map(runmat_types::physical::Frequency::new),
        phase_radians: runmat_types::physical::Angle::new(0.0),
        duty_cycle: None,
    });
    assert!(validate_signal(&signal, None).is_err());
    signal.representation = SignalRepresentation::Artifact(SignalArtifactRef {
        artifact_id: "sealed".into(),
        content_digest: "digest".into(),
        schema: "schema".into(),
        domain: SignalArtifactDomain::TimeSeconds {
            start: 0.0,
            end: 1.0,
        },
        topology_digest: Some("digest".into()),
        sample_count: 2,
    });
    assert!(validate_signal(&signal, None).is_err());
    signal = original;
    signal.provenance = SignalProvenance::Imported {
        source_digest: "foreign".into(),
    };
    assert!(validate_signal(&signal, None).is_err());
    signal.provenance = SignalProvenance::Generated(GeneratorProvenance {
        project_revision: "revision".into(),
        callable_id: "callable".into(),
        frozen_input_digest: "digest".into(),
    });
    assert!(validate_signal(&signal, None).is_err());
}

#[test]
fn replacing_shape_preserves_authored_contract_and_does_not_resolve_source() {
    let source = PhysicalSignal::Temperature(declaration(1, AuthoredSignalFieldLocation::Node));
    let replacement = source.with_shape(SignalShape::TopologyField {
        topology_id: "mesh_digest".into(),
        location: crate::FieldLocation::Node,
        component_count: 1,
    });
    assert_eq!(replacement.quantity(), source.quantity());
    assert_eq!(replacement.signal_id(), source.signal_id());
    assert_eq!(replacement.with_shape(source.shape().clone()), source);
    assert!(matches!(
        source.shape(),
        SignalShape::DeferredTopologyField { .. }
    ));
}

#[test]
fn authored_excitation_requires_field_distribution_and_matching_physical_width() {
    let signal = PhysicalSignal::Temperature(declaration(1, AuthoredSignalFieldLocation::Node));
    let mut excitation = crate::AuthoredExcitation {
        excitation_id: "temperature_load".into(),
        kind: crate::AuthoredExcitationKind::TemperatureField,
        target: crate::AuthoredExcitationTarget::Region(crate::AuthoredRegionSource::Geometry(
            runmat_geometry_core::GeometryRegionRef {
                module_path: "project/main.fea".into(),
                geometry_id: "mesh".into(),
                region_id: "body".into(),
                role: runmat_geometry_core::GeometryRegionRole::Solid,
            },
        )),
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: crate::ExcitationDistribution::TopologyField,
        kinematics: crate::ExcitationKinematics::FixedFrame,
        signal_id: signal.signal_id().clone(),
        active_step_ids: vec!["step".into()],
        combination_group: None,
        projection: crate::ProjectionRequirement {
            preserve_resultant: false,
            preserve_moment: false,
            relative_tolerance: 1.0e-9,
        },
    };
    let cases = [crate::AuthoredExcitationCase {
        case_id: "case".into(),
        excitation_ids: vec![excitation.excitation_id.clone()],
        activation: crate::ExcitationCaseActivation::Always,
    }];
    crate::validate_authored_excitations(
        std::slice::from_ref(&excitation),
        &cases,
        std::slice::from_ref(&signal),
        &[],
    )
    .unwrap();
    excitation.distribution = crate::ExcitationDistribution::UniformKinematic;
    assert!(crate::validate_authored_excitations(
        std::slice::from_ref(&excitation),
        &cases,
        std::slice::from_ref(&signal),
        &[]
    )
    .is_err());
    excitation.distribution = crate::ExcitationDistribution::TopologyField;
    let wrong_width = signal.with_shape(SignalShape::DeferredTopologyField {
        selector: AuthoredSignalTopologySelector::StudySolverMesh {
            location: AuthoredSignalFieldLocation::Node,
        },
        component_count: 3,
    });
    assert!(
        crate::validate_authored_excitations(&[excitation], &cases, &[wrong_width], &[]).is_err()
    );
}
