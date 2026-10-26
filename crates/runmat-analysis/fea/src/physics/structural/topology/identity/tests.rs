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
            "coordinates_m": node.coordinates_m.canonical_value().map(f64::to_bits),
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
        {
            let quantity_change = -0.0;
            let quantity_index = 0;
            let quantity_slot = &mut structural.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
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
            &(
                digest,
                node.node_id,
                node.coordinates_m.canonical_value().map(f64::to_bits),
            ),
            LIMITS,
        )
        .unwrap();
        assert_eq!(
            explicit_node_identity(digest, node.node_id, node.coordinates_m.canonical_value())
                .unwrap(),
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
    {
        let quantity_change = f64::NAN;
        let quantity_index = 0;
        let quantity_slot = &mut nonfinite.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
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
        assert!(storage::slots::<(u64, usize)>(usize::MAX).is_err());
    }
}

// witness: FEA-SHARED-CONSTRAINTS-001/attributable_validation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Counted validation reports the first authored failure before later duplicates.
#[test]
fn counted_validation_reports_first_authored_failure_before_later_duplicates() {
    let mut structural = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference)
        .structural
        .unwrap();
    let mut duplicate = structural.nodes[0].clone();
    {
        let quantity_change = f64::NAN;
        let quantity_index = 0;
        let quantity_slot = &mut duplicate.coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    structural.nodes.push(duplicate);
    assert!(matches!(
        explicit_topology_digest(&structural),
        Err(ExplicitStructuralTopologyError::DuplicateNode { .. })
    ));
    {
        let quantity_change = f64::NAN;
        let quantity_index = 0;
        let quantity_slot = &mut structural.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(matches!(
        explicit_topology_digest(&structural),
        Err(ExplicitStructuralTopologyError::NonFiniteNode { .. })
    ));
    {
        let quantity_change = 0.0;
        let quantity_index = 0;
        let quantity_slot = &mut structural.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    structural.nodes.pop();
    structural.elements.push(structural.elements[0].clone());
    let StructuralElementKind::Beam(beam) = &mut structural.elements[0].kind else {
        panic!("beam")
    };
    beam.node_ids[0] = u32::MAX;
    assert!(matches!(
        explicit_topology_digest(&structural),
        Err(ExplicitStructuralTopologyError::MissingElementNode {
            node_id: u32::MAX,
            ..
        })
    ));
}
