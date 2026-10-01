use super::*;

#[test]
fn magnetic_volume_current_is_vector_volume_intensity_distinct_from_surface_current() {
    let kind = AuthoredExcitationKind::MagneticVolumeCurrentDensity;
    assert_eq!(
        kind.quantity(),
        PhysicalQuantityKind::ElectricCurrentDensity
    );
    assert_eq!(
        kind.default_distribution(),
        ExcitationDistribution::IntensityPerVolume
    );
    assert_eq!(
        AuthoredExcitationKind::CurrentDensity.default_distribution(),
        ExcitationDistribution::IntensityPerArea
    );
    assert!(!kind.preserves_resultant());
    assert!(!kind.preserves_moment());
    assert!(valid_shape_and_direction(
        kind,
        &SignalShape::Vector3,
        false
    ));
    assert!(!valid_shape_and_direction(
        kind,
        &SignalShape::Scalar,
        false
    ));
    assert!(!valid_shape_and_direction(kind, &SignalShape::Scalar, true));
    assert!(!valid_shape_and_direction(
        kind,
        &SignalShape::Vector3,
        true
    ));
    assert!(!valid_shape_and_direction(
        kind,
        &SignalShape::TopologyField {
            topology_id: "mesh".into(),
            location: crate::FieldLocation::Node,
            component_count: 3,
        },
        false
    ));
    for distribution in [
        ExcitationDistribution::PreserveResultant,
        ExcitationDistribution::IntensityPerArea,
        ExcitationDistribution::TopologyField,
    ] {
        assert!(!valid_distribution(kind, distribution));
    }
    assert_eq!(
        serde_json::to_string(&kind).unwrap(),
        "\"magnetic_volume_current_density\""
    );
}
