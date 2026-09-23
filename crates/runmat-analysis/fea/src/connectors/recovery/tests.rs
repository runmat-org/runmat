use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;

use super::*;
use crate::{
    assembly::dofs::StructuralNodeDofSet,
    connectors::{CanonicalConnectorNode, CanonicalStructuralConnectorPlan},
    physics::structural::topology::CanonicalStructuralTopologyIdentity,
};

#[test]
fn static_recovery_uses_the_exact_axial_and_ground_laws() {
    let layout = layout();
    let plan = CanonicalStructuralConnectorPlan::new(
        topology(),
        2,
        Digest::sha256(b"authored"),
        vec![
            CanonicalConnectorEntry::AxialSpring {
                connector_id: "axial".into(),
                source_digest: Digest::sha256(b"axial-source"),
                primary: node(0, 1),
                secondary: node(1, 2),
                unit_axis: [1.0, 0.0, 0.0],
                stiffness_newtons_per_meter: 100.0,
            },
            CanonicalConnectorEntry::IsotropicGroundSpring {
                connector_id: "ground".into(),
                source_digest: Digest::sha256(b"ground-source"),
                target: node(1, 2),
                stiffness_newtons_per_meter: 10.0,
            },
        ],
    )
    .unwrap();
    let result = CanonicalStructuralConnectorResult::recover_static(
        plan,
        &layout,
        &[0.0, 0.0, 0.0, 0.25, -0.5, 1.0],
    )
    .unwrap();
    assert_eq!(result.responses.len(), 2);
    assert!(matches!(
        &result.responses[0],
        CanonicalConnectorResponse::AxialSpring {
            extension_meters: 0.25,
            axial_force_newtons: 25.0,
            resultant_force_newtons: [25.0, 0.0, 0.0],
            stored_energy_joules: 3.125,
            ..
        }
    ));
    assert!(matches!(
        &result.responses[1],
        CanonicalConnectorResponse::IsotropicGroundSpring {
            displacement_meters: [0.25, -0.5, 1.0],
            resultant_force_newtons: [-2.5, 5.0, -10.0],
            stored_energy_joules: 6.5625,
            ..
        }
    ));
    assert_eq!(result.total_stored_energy_joules, 9.6875);
    assert_eq!(result.total_dissipated_power_watts, 0.0);
}

#[test]
fn dynamic_recovery_uses_relative_axial_velocity_for_dashpots() {
    let layout = layout();
    let plan = dashpot_plan([0.6, 0.8, 0.0], 50.0);
    let result = CanonicalStructuralConnectorResult::recover_dynamic(
        plan,
        &layout,
        &[0.0; 6],
        &[1.0, 2.0, 0.0, 4.0, 6.0, 0.0],
    )
    .unwrap();
    assert!(matches!(
        &result.responses[0],
        CanonicalConnectorResponse::AxialDashpot {
            relative_axial_velocity_meters_per_second: 5.0,
            axial_force_newtons: 250.0,
            resultant_force_newtons: [150.0, 200.0, 0.0],
            stored_energy_joules: 0.0,
            dissipated_power_watts: 1250.0,
            ..
        }
    ));
    assert_eq!(result.total_stored_energy_joules, 0.0);
    assert_eq!(result.total_dissipated_power_watts, 1250.0);
}

#[test]
fn dynamic_recovery_rejects_tampered_dashpot_power() {
    let layout = layout();
    let mut result = CanonicalStructuralConnectorResult::recover_dynamic(
        dashpot_plan([1.0, 0.0, 0.0], 10.0),
        &layout,
        &[0.0; 6],
        &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    )
    .unwrap();
    let CanonicalConnectorResponse::AxialDashpot {
        dissipated_power_watts,
        ..
    } = &mut result.responses[0]
    else {
        panic!("expected axial dashpot response");
    };
    *dissipated_power_watts = 1.0;
    assert!(result.validate().is_err());
}

#[test]
fn result_validation_rejects_energy_substitution() {
    let layout = layout();
    let plan = CanonicalStructuralConnectorPlan::new(
        topology(),
        2,
        Digest::sha256(b"authored"),
        vec![CanonicalConnectorEntry::AxialSpring {
            connector_id: "axial".into(),
            source_digest: Digest::sha256(b"source"),
            primary: node(0, 1),
            secondary: node(1, 2),
            unit_axis: [1.0, 0.0, 0.0],
            stiffness_newtons_per_meter: 100.0,
        }],
    )
    .unwrap();
    let mut result = CanonicalStructuralConnectorResult::recover_static(
        plan,
        &layout,
        &[0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
    )
    .unwrap();
    result.total_stored_energy_joules = 99.0;
    assert!(result.validate().is_err());
}

fn dashpot_plan(
    unit_axis: [f64; 3],
    damping_newton_seconds_per_meter: f64,
) -> CanonicalStructuralConnectorPlan {
    CanonicalStructuralConnectorPlan::new(
        topology(),
        2,
        Digest::sha256(b"authored"),
        vec![CanonicalConnectorEntry::AxialDashpot {
            connector_id: "dashpot".into(),
            source_digest: Digest::sha256(b"dashpot-source"),
            primary: node(0, 1),
            secondary: node(1, 2),
            unit_axis,
            damping_newton_seconds_per_meter,
        }],
    )
    .unwrap()
}

fn layout() -> StructuralDofLayout {
    StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational(),
        StructuralNodeDofSet::translational(),
    ])
}

fn topology() -> CanonicalStructuralTopologyIdentity {
    CanonicalStructuralTopologyIdentity::Explicit {
        structural_topology_digest: Digest::sha256(b"topology"),
    }
}

fn node(node_index: usize, byte: u8) -> CanonicalConnectorNode {
    CanonicalConnectorNode {
        node_index,
        node_identity: StableDigest::from_bytes([byte; 32]),
    }
}
