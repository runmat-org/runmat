use super::*;
use crate::results::field::{RecoveredFieldEntityCounts, RecoveredFieldEntityDomain};

#[test]
fn generic_entity_contract_has_no_old_domain_or_count_decoding() {
    for obsolete in [
        "structural_nodes",
        "solid_elements",
        "solid_integration_points",
    ] {
        assert!(
            serde_json::from_str::<RecoveredFieldEntityDomain>(&format!("\"{obsolete}\"")).is_err()
        );
    }
    let counts = RecoveredFieldEntityCounts {
        oriented_faces: 0,
        nodes: 4,
        volume_elements: 1,
        volume_integration_points: 4,
        oriented_edges: 0,
        beam_elements: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    };
    let mut old = serde_json::to_value(counts).unwrap();
    let fields = old.as_object_mut().unwrap();
    let nodes = fields.remove("nodes").unwrap();
    fields.insert("structural_nodes".into(), nodes);
    assert!(serde_json::from_value::<RecoveredFieldEntityCounts>(old).is_err());
}
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, SequenceIdentity,
    SolverResultIdentity, StepIdentity,
};

#[test]
fn recovery_request_binds_typed_topology_and_rejects_empty_identity() {
    let sequence = SequenceIdentity::new(Digest::sha256("sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
    let source = FieldRecoverySource::SolverResult(
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap(),
    );
    let mesh_digest = StableDigest::from_bytes(*Digest::sha256("mesh").bytes());
    let volume = FieldRecoveryRequest::new(
        producer,
        source,
        FieldTopologyIdentity::VolumeMesh { mesh_digest },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let structural = FieldRecoveryRequest::new(
        producer,
        source,
        FieldTopologyIdentity::Structural {
            topology: CanonicalStructuralTopologyIdentity::Solid { mesh_digest },
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    assert_eq!(
        volume.topology.discretization_digest(),
        structural.topology.discretization_digest()
    );
    assert_ne!(volume.identity(), structural.identity());
    assert!(volume.topology.require_structural().is_err());
    assert_eq!(
        FieldRecoveryRequest::from_canonical_bytes(&volume.canonical_bytes().unwrap()).unwrap(),
        volume
    );
    for mesh_digest in [
        StableDigest::ZERO,
        StableDigest::from_bytes(*Digest::sha256([]).bytes()),
    ] {
        assert!(FieldRecoveryRequest::new(
            producer,
            source,
            FieldTopologyIdentity::VolumeMesh { mesh_digest },
            FieldRecoveryPlan::default()
        )
        .is_err());
    }
}

#[test]
fn field_topology_keeps_family_authority_and_explicit_mixed_identity() {
    let digest = Digest::sha256("mesh");
    let mesh_digest = StableDigest::from_bytes(*digest.bytes());
    let volume = FieldTopologyIdentity::VolumeMesh { mesh_digest };
    let solid = FieldTopologyIdentity::Structural {
        topology: CanonicalStructuralTopologyIdentity::Solid { mesh_digest },
    };
    assert_eq!(
        volume.discretization_digest(),
        solid.discretization_digest()
    );
    assert_ne!(volume, solid);
    assert!(volume.require_structural().is_err());
    for topology in [
        CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: Digest::sha256("explicit"),
        },
        CanonicalStructuralTopologyIdentity::ConformalMixed {
            mesh_digest,
            structural_topology_digest: Digest::sha256("explicit"),
        },
    ] {
        let field = FieldTopologyIdentity::Structural { topology };
        assert_eq!(field.require_structural().unwrap(), topology);
        assert_eq!(
            field.discretization_digest(),
            topology.discretization_digest()
        );
        let cold: FieldTopologyIdentity =
            serde_json::from_slice(&serde_json::to_vec(&field).unwrap()).unwrap();
        assert_eq!(cold, field);
        assert!(cold.has_valid_identity());
    }
    let cold: FieldTopologyIdentity =
        serde_json::from_slice(&serde_json::to_vec(&volume).unwrap()).unwrap();
    assert_eq!(cold, volume);
    for mesh_digest in [
        StableDigest::ZERO,
        StableDigest::from_bytes(*Digest::sha256([]).bytes()),
    ] {
        assert!(!FieldTopologyIdentity::VolumeMesh { mesh_digest }.has_valid_identity());
    }
}
