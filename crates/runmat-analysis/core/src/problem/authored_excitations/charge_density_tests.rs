use super::*;

#[test]
fn charge_density_is_a_scalar_volume_intensity_not_total_or_surface_charge() {
    let kind = AuthoredExcitationKind::ElectricChargeDensity;
    assert_eq!(kind.quantity(), PhysicalQuantityKind::ElectricChargeDensity);
    assert_eq!(
        kind.default_distribution(),
        ExcitationDistribution::IntensityPerVolume
    );
    assert!(!kind.preserves_resultant());
    assert!(!kind.preserves_moment());
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
        ExcitationDistribution::Concentrated,
        ExcitationDistribution::PreserveResultant,
        ExcitationDistribution::IntensityPerArea,
        ExcitationDistribution::UniformKinematic,
        ExcitationDistribution::TopologyField,
    ] {
        assert!(!valid_distribution(kind, distribution));
    }
    assert_eq!(
        serde_json::to_string(&kind).unwrap(),
        "\"electric_charge_density\""
    );
}
