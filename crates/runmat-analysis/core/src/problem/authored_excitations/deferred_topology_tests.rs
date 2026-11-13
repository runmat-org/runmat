use super::*;
use crate::{AuthoredSignalFieldLocation, AuthoredSignalTopologySelector};

fn shape(width: usize, deferred: bool) -> SignalShape {
    if deferred {
        SignalShape::DeferredTopologyField {
            selector: AuthoredSignalTopologySelector::StudySolverMesh {
                location: AuthoredSignalFieldLocation::Node,
            },
            component_count: width,
        }
    } else {
        SignalShape::TopologyField {
            topology_id: "mesh_digest".into(),
            location: crate::FieldLocation::Node,
            component_count: width,
        }
    }
}

#[test]
fn topology_excitation_declarations_preserve_physical_width_and_direction() {
    for deferred in [false, true] {
        for kind in [
            AuthoredExcitationKind::BodyForce,
            AuthoredExcitationKind::Traction,
            AuthoredExcitationKind::CurrentDensity,
            AuthoredExcitationKind::MagneticVectorPotential,
            AuthoredExcitationKind::InletVelocity,
            AuthoredExcitationKind::WallMotion,
        ] {
            assert!(valid_shape_and_direction(kind, &shape(3, deferred), false));
            assert!(!valid_shape_and_direction(kind, &shape(1, deferred), false));
            assert!(!valid_shape_and_direction(kind, &shape(2, deferred), false));
            assert!(!valid_shape_and_direction(kind, &shape(3, deferred), true));
            assert!(valid_distribution(
                kind,
                ExcitationDistribution::TopologyField
            ));
        }
        for kind in [
            AuthoredExcitationKind::Pressure,
            AuthoredExcitationKind::HeatGeneration,
            AuthoredExcitationKind::HeatFlux,
            AuthoredExcitationKind::TemperatureField,
            AuthoredExcitationKind::AmbientTemperature,
            AuthoredExcitationKind::TerminalVoltage,
            AuthoredExcitationKind::AcousticPressure,
            AuthoredExcitationKind::NormalVelocity,
        ] {
            assert!(valid_shape_and_direction(kind, &shape(1, deferred), false));
            assert!(!valid_shape_and_direction(kind, &shape(3, deferred), false));
            assert!(!valid_shape_and_direction(kind, &shape(1, deferred), true));
            assert!(valid_distribution(
                kind,
                ExcitationDistribution::TopologyField
            ));
        }
        for kind in [
            AuthoredExcitationKind::Force,
            AuthoredExcitationKind::Moment,
            AuthoredExcitationKind::ElectricCurrent,
            AuthoredExcitationKind::ElectricCharge,
            AuthoredExcitationKind::MassFlow,
        ] {
            assert!(!valid_shape_and_direction(kind, &shape(1, deferred), false));
            assert!(!valid_shape_and_direction(kind, &shape(3, deferred), false));
        }
    }
}

#[test]
fn uniform_shape_rules_are_unchanged_and_zero_width_is_rejected() {
    assert!(valid_shape_and_direction(
        AuthoredExcitationKind::BodyForce,
        &SignalShape::Vector3,
        false
    ));
    assert!(valid_shape_and_direction(
        AuthoredExcitationKind::Force,
        &SignalShape::Scalar,
        true
    ));
    assert!(!valid_shape_and_direction(
        AuthoredExcitationKind::Force,
        &SignalShape::Scalar,
        false
    ));
    assert!(valid_shape_and_direction(
        AuthoredExcitationKind::HeatGeneration,
        &SignalShape::Scalar,
        false
    ));
    for deferred in [false, true] {
        assert!(!valid_shape_and_direction(
            AuthoredExcitationKind::BodyForce,
            &shape(0, deferred),
            false
        ));
        assert!(!valid_shape_and_direction(
            AuthoredExcitationKind::HeatGeneration,
            &shape(0, deferred),
            false
        ));
    }
}
