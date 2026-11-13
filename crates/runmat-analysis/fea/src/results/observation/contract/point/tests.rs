use super::*;

#[test]
fn one_point_payload_preserves_original_canonical_identity_and_bounds_encoding() {
    for count in [4, 10] {
        let sample = ResolvedObservationPoint {
            identity: StableDigest::ZERO,
            ordinal: 31,
            position_m: [0.2, 0.3, 0.1],
            source_element_index: 11,
            source_element_identity: StableDigest::ZERO,
            source_node_indices: (0..count).collect(),
            source_node_identities: vec![StableDigest::ZERO; count],
            shape_values: vec![1.0 / count as f64; count],
        };
        let topology = Digest::sha256("original point topology");
        let original = encode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-point",
            &(
                topology,
                sample.ordinal,
                sample.position_m,
                sample.source_element_identity,
                &sample.source_node_indices,
                &sample.shape_values,
            ),
            OBSERVATION_LIMITS,
        )
        .unwrap();
        let expected = StableDigest::from_bytes(*Digest::sha256(&original).bytes());
        let bound = resolved_point_identity_resident_bytes(count).unwrap();
        let (identity, _, peak) = crate::topology::measure_test_allocations(|| {
            derive_resolved_point_identity(topology, &sample)
        });
        assert_eq!(identity.unwrap(), expected);
        assert!(peak as u64 <= bound);
    }
    assert!(resolved_point_identity_resident_bytes(11).is_err());
}
