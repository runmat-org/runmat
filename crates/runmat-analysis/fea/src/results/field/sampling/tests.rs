use super::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn shared_sample_rules_have_distinct_canonical_inventory_and_signed_edges() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut previous = None;
        for rule in [
            VolumeFieldSamplingRule::TetrahedronFourPoint,
            VolumeFieldSamplingRule::TetrahedronKeastFifteen,
        ] {
            let topology = SampledVolumeFieldTopology::derive(&mesh, rule).unwrap();
            topology.validate_for_mesh(&mesh).unwrap();
            assert_eq!(topology.entity_counts().oriented_edges, 6);
            assert_eq!(
                topology.integration_sample_count(),
                rule.points_per_element()
            );
            let samples = topology.integration_samples_for(&mesh).unwrap();
            assert_eq!(samples.len(), rule.points_per_element());
            for (index, sample) in samples.iter().enumerate() {
                assert_eq!(sample.ordinal, index);
                assert_eq!(
                    sample.identity,
                    topology.integration_sample_identity(index).unwrap()
                );
                assert_eq!(sample.element_identity, topology.element_identities()[0]);
            }
            assert_eq!(
                topology
                    .entity_counts()
                    .entity_count(super::super::RecoveredFieldEntityDomain::OrientedEdges),
                6
            );
            let volume: f64 = samples.iter().map(|sample| sample.volume_weight_m3).sum();
            assert!((volume - 1.0 / 6.0).abs() < 1e-13);
            for sample in &samples {
                assert!(sample
                    .position_m
                    .iter()
                    .all(|&value| (0.0..=1.0).contains(&value)));
                assert!(sample.volume_weight_m3 > 0.0);
            }
            for edge in topology.oriented_edges() {
                assert_eq!(edge.orientation_sign(edge.node_identities), Some(1));
                assert_eq!(
                    edge.orientation_sign([edge.node_identities[1], edge.node_identities[0]]),
                    Some(-1)
                );
                assert_eq!(edge.orientation_sign([edge.node_identities[0]; 2]), None);
            }
            // The layout is carried inside result contracts, whose limits come from their plan.
            let limits = crate::contracts::canonical_digest::wire_bounded(
                runmat_canonical_codec::CanonicalLimits::new(0, 0, 65_536, 64),
            );
            let bytes = runmat_canonical_codec::encode_contract(
                b"runmat-test-sampled-topology/v1\0",
                "sampled-volume-field-topology",
                &topology,
                limits,
            )
            .unwrap();
            let reopened: SampledVolumeFieldTopology = runmat_canonical_codec::decode_contract(
                b"runmat-test-sampled-topology/v1\0",
                "sampled-volume-field-topology",
                &bytes,
                limits,
            )
            .unwrap();
            assert_eq!(
                topology.sampling_digest(),
                Digest::sha256(
                    runmat_canonical_codec::encode_contract(
                        PREFIX,
                        "sampling-layout",
                        &topology.data,
                        limits
                    )
                    .unwrap()
                )
            );
            assert_eq!(reopened, topology);
            if let Some(previous) = previous {
                assert_ne!(topology.identity().unwrap(), previous);
            }
            previous = Some(topology.identity().unwrap());
        }
    }
}

#[test]
fn malformed_compact_layout_is_rejected_even_with_updated_digest() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let topology =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    for defect in 0..7 {
        let mut bad = topology.data.clone();
        match defect {
            0 => bad.element_corner_nodes[0][1] = bad.element_corner_nodes[0][0],
            1 => bad.element_corner_nodes[0][3] = bad.node_identities.len() as u32,
            2 => {
                bad.element_corner_nodes.pop().unwrap();
            }
            3 => bad.node_identities[1] = bad.node_identities[0],
            4 => bad.element_identities[0] = StableDigest::ZERO,
            5 => bad.schema_version = 1,
            6 => bad.projection_digest = Digest::from_bytes([0; 32]),
            _ => unreachable!(),
        }
        // A recomputed digest does not make a malformed layout admissible.
        assert!(
            SampledVolumeFieldTopology::admit(bad, None).is_err(),
            "accepted defect {defect}"
        );
    }
    // A stored digest that does not name the stored content is rejected.
    let mut renamed = topology.data.clone();
    renamed.element_corner_nodes[0].swap(0, 1);
    assert!(SampledVolumeFieldTopology::admit(renamed, Some(topology.sampling_digest)).is_err());
    for field in [
        "element_corner_nodes",
        "node_identities",
        "element_identities",
    ] {
        let mut corrupt = serde_json::to_value(&topology).unwrap();
        corrupt["data"][field][0] = corrupt["data"][field][1].clone();
        assert!(
            serde_json::from_value::<SampledVolumeFieldTopology>(corrupt).is_err(),
            "accepted tampered {field}"
        );
    }
    let mut corrupt = serde_json::to_value(&topology).unwrap();
    corrupt["data"]["rule"] = serde_json::json!("arbitrary_rule");
    assert!(serde_json::from_value::<SampledVolumeFieldTopology>(corrupt).is_err());
    let mut corrupt = serde_json::to_value(&topology).unwrap();
    corrupt["data"]["integration_samples"] = serde_json::json!([]);
    assert!(serde_json::from_value::<SampledVolumeFieldTopology>(corrupt).is_err());
}

#[test]
fn compact_layout_round_trips_and_its_wire_does_not_grow_with_rule_points() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let limits = crate::contracts::canonical_digest::wire_bounded(
        runmat_canonical_codec::CanonicalLimits::new(0, 0, 65_536, 64),
    );
    let mut sizes = Vec::new();
    for rule in [
        VolumeFieldSamplingRule::TetrahedronFourPoint,
        VolumeFieldSamplingRule::TetrahedronKeastFifteen,
    ] {
        let topology = SampledVolumeFieldTopology::derive(&mesh, rule).unwrap();
        let bytes =
            runmat_canonical_codec::encode_contract(PREFIX, "layout", &topology, limits).unwrap();
        let reopened: SampledVolumeFieldTopology =
            runmat_canonical_codec::decode_contract(PREFIX, "layout", &bytes, limits).unwrap();
        assert_eq!(reopened, topology);
        assert_eq!(reopened.oriented_edges(), topology.oriented_edges());
        reopened.validate_for_mesh(&mesh).unwrap();
        // The planning shape bounds the encoding of the same inventory.
        let shape = runmat_canonical_codec::profile_serialized_shape(
            &SampledTopologyShape::new(mesh.topology.nodes.len(), 1, rule),
            limits,
        )
        .unwrap();
        assert!(shape.encoded_bytes_upper_bound >= bytes.len() as u64);
        sizes.push(bytes.len());
    }
    // Samples are named by (element, rule point): the rule changes no stored inventory.
    assert!(sizes[0].abs_diff(sizes[1]) < 64, "{sizes:?}");
}

#[test]
fn altered_physical_layout_cannot_splice_into_source_mesh_or_recovery_request() {
    use crate::execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, SequenceIdentity,
        SolverResultIdentity, StepIdentity,
    };
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let topology =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    // Permuting an element's corners keeps a well-formed layout and edge space, but it is not
    // the mesh's layout.
    let mut data = topology.data.clone();
    data.element_corner_nodes[0].swap(0, 1);
    let altered = SampledVolumeFieldTopology::admit(data, None).unwrap();
    assert_eq!(altered.oriented_edges(), topology.oriented_edges());
    assert!(altered.validate_for_mesh(&mesh).is_err());
    assert_ne!(
        topology.identity().unwrap().discretization_digest(),
        altered.identity().unwrap().discretization_digest()
    );
    let sequence = SequenceIdentity::new(Digest::sha256("sampling-test")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("step")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap();
    let make_request = |topology| {
        FieldRecoveryRequest::new(
            producer,
            FieldRecoverySource::SolverResult(result),
            topology,
            FieldRecoveryPlan::default(),
        )
    };
    let original_request = make_request(topology.identity().unwrap()).unwrap();
    let altered_request = make_request(altered.identity().unwrap()).unwrap();
    assert_ne!(original_request.identity(), altered_request.identity());
}
