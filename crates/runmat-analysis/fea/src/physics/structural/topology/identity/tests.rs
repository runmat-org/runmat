use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate;
use serde_json::json;

// Independent old sequence-of-owned-records representation: detects byte/identity changes from
// replacing topology vectors, connectivity copies, and cloned attachments with borrowed views.
fn legacy_topology(structural: &StructuralModel) -> serde_json::Value {
    json!({
        "nodes": structural.nodes.iter().map(|node| json!({
            "node_id": node.node_id,
            "coordinates_m": node.coordinates_m.map(f64::to_bits),
            "solid_attachment": node.solid_attachment.clone(),
        })).collect::<Vec<_>>(),
        "elements": structural.elements.iter().map(|element| json!({
            "element_id": element.element_id,
            "region_id": element.region_id,
            "node_ids": node_ids(&element.kind).to_vec(),
        })).collect::<Vec<_>>(),
    })
}

#[test]
fn borrowed_topology_and_element_contracts_preserve_canonical_bytes() {
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let mut structural = fixture_model(fixture).structural.unwrap();
        structural.nodes[0].coordinates_m[0] = -0.0;
        structural.nodes[0].solid_attachment = Some(ConformalSolidAttachment::SolverNode {
            stable_identity: [7; 32],
        });
        let expected = encode_contract(
            PREFIX,
            "explicit-structural-topology",
            &legacy_topology(&structural),
            LIMITS,
        )
        .unwrap();
        let actual = encode_contract(
            PREFIX,
            "explicit-structural-topology",
            &TopologyContract {
                nodes: &structural.nodes,
                elements: &structural.elements,
            },
            LIMITS,
        )
        .unwrap();
        assert_eq!(actual, expected);
        let digest = explicit_topology_digest(&structural).unwrap();
        assert_eq!(digest, Digest::sha256(&expected));
        assert!(
            explicit_identity_resource_bytes(&structural).unwrap()
                >= canonical_payload_materialization_resident_bytes_estimate(&expected).unwrap()
        );
        for element in &structural.elements {
            let expected = encode_contract(
                PREFIX,
                "explicit-structural-element",
                &(
                    digest,
                    &element.element_id,
                    &element.region_id,
                    node_ids(&element.kind).to_vec(),
                ),
                LIMITS,
            )
            .unwrap();
            let actual = explicit_element_identity(
                digest,
                &element.element_id,
                &element.region_id,
                &element.kind,
            )
            .unwrap();
            assert_eq!(actual, stable_hash(expected.clone()));
            assert!(
                explicit_identity_resource_bytes(&structural).unwrap()
                    >= canonical_payload_materialization_resident_bytes_estimate(&expected)
                        .unwrap()
            );
        }
        let node = &structural.nodes[0];
        let expected = encode_contract(
            PREFIX,
            "explicit-structural-node",
            &(digest, node.node_id, node.coordinates_m.map(f64::to_bits)),
            LIMITS,
        )
        .unwrap();
        assert_eq!(
            explicit_node_identity(digest, node.node_id, node.coordinates_m).unwrap(),
            stable_hash(expected)
        );
        let first = digest;
        structural.nodes[0].solid_attachment = None;
        assert_ne!(first, explicit_topology_digest(&structural).unwrap());
    }
}

#[test]
fn borrowed_validation_preserves_duplicate_nonfinite_and_missing_node_errors() {
    let structural = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference)
        .structural
        .unwrap();
    let mut duplicate_node = structural.clone();
    duplicate_node.nodes.push(duplicate_node.nodes[0].clone());
    assert!(matches!(
        explicit_topology_digest(&duplicate_node),
        Err(ExplicitStructuralTopologyError::DuplicateNode { .. })
    ));
    let mut nonfinite = structural.clone();
    nonfinite.nodes[0].coordinates_m[0] = f64::NAN;
    assert!(matches!(
        explicit_topology_digest(&nonfinite),
        Err(ExplicitStructuralTopologyError::NonFiniteNode { .. })
    ));
    let mut duplicate_element = structural.clone();
    duplicate_element
        .elements
        .push(duplicate_element.elements[0].clone());
    assert!(matches!(
        explicit_topology_digest(&duplicate_element),
        Err(ExplicitStructuralTopologyError::DuplicateElement { .. })
    ));
    let mut missing = structural;
    let StructuralElementKind::Beam(beam) = &mut missing.elements[0].kind else {
        panic!("beam fixture")
    };
    beam.node_ids[1] = u32::MAX;
    assert!(matches!(
        explicit_topology_digest(&missing),
        Err(ExplicitStructuralTopologyError::MissingElementNode {
            node_id: u32::MAX,
            ..
        })
    ));
}

#[test]
fn identity_profile_counts_large_selected_topology_metadata() {
    let mut structural = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference)
        .structural
        .unwrap();
    let baseline = explicit_identity_resource_bytes(&structural).unwrap();
    structural.elements[0].region_id = "region".repeat(4096);
    let expanded = explicit_identity_resource_bytes(&structural).unwrap();
    assert!(expanded > baseline);
    let bytes = encode_contract(
        PREFIX,
        "explicit-structural-topology",
        &legacy_topology(&structural),
        LIMITS,
    )
    .unwrap();
    assert!(expanded >= canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap());
    if usize::BITS == 64 {
        assert!(lookup_bytes::<u64>(usize::MAX).is_err());
    }
}
