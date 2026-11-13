use super::*;

#[test]
fn incident_pressure_is_uniform_scalar_without_caller_direction() {
    let kind = AuthoredExcitationKind::AcousticIncidentPressure;
    assert_eq!(kind.quantity(), PhysicalQuantityKind::Pressure);
    assert_eq!(
        kind.default_distribution(),
        ExcitationDistribution::UniformKinematic
    );
    assert!(valid_shape_and_direction(kind, &SignalShape::Scalar, false));
    assert!(!valid_shape_and_direction(kind, &SignalShape::Scalar, true));
    assert!(!valid_shape_and_direction(
        kind,
        &SignalShape::Vector3,
        false
    ));
    assert!(!valid_shape_and_direction(
        kind,
        &SignalShape::TopologyField {
            topology_id: "mesh".into(),
            location: crate::FieldLocation::Node,
            component_count: 1,
        },
        false
    ));
    for distribution in [
        ExcitationDistribution::IntensityPerArea,
        ExcitationDistribution::PreserveResultant,
        ExcitationDistribution::TopologyField,
    ] {
        assert!(!valid_distribution(kind, distribution));
    }
    assert_eq!(
        serde_json::to_string(&kind).unwrap(),
        "\"acoustic_incident_pressure\""
    );
}
