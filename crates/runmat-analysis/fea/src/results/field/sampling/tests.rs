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
                topology.integration_samples().len(),
                rule.points_per_element()
            );
            assert_eq!(
                topology
                    .entity_counts()
                    .entity_count(super::super::RecoveredFieldEntityDomain::OrientedEdges),
                6
            );
            let volume: f64 = topology
                .integration_samples()
                .iter()
                .map(|sample| sample.volume_weight_m3)
                .sum();
            assert!((volume - 1.0 / 6.0).abs() < 1e-13);
            for sample in topology.integration_samples() {
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
fn malformed_sample_and_edge_inventory_is_rejected_even_with_updated_digest() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let topology =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronKeastFifteen)
            .unwrap();
    for defect in 0..8 {
        let mut bad = topology.clone();
        match defect {
            0 => bad.data.oriented_edges[0].node_identities.reverse(),
            1 => bad
                .data
                .oriented_edges
                .push(bad.data.oriented_edges[0].clone()),
            2 => {
                bad.data.integration_samples.pop().unwrap();
            }
            3 => bad.data.integration_samples[0].ordinal = 1,
            4 => bad.data.integration_samples[0].volume_weight_m3 = -1.0,
            5 => bad.data.integration_samples[0].position_m[0] = f64::NAN,
            6 => bad.data.integration_samples[0].element_identity = StableDigest::ZERO,
            7 => {
                bad.data.oriented_edges.pop().unwrap();
            }
            _ => unreachable!(),
        }
        bad.sampling_digest = digest("sampling-layout", &bad.data).unwrap();
        assert!(bad.validate().is_err(), "accepted defect {defect}");
    }
    let mut corrupt = serde_json::to_value(&topology).unwrap();
    corrupt["data"]["rule"] = serde_json::json!("arbitrary_rule");
    assert!(serde_json::from_value::<SampledVolumeFieldTopology>(corrupt).is_err());
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
    let mut altered = topology.clone();
    altered.data.integration_samples[0].position_m[0] += 0.1;
    altered.sampling_digest = digest("sampling-layout", &altered.data).unwrap();
    altered.validate().unwrap();
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
