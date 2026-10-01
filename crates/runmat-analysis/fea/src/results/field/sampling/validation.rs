use super::*;
use std::collections::BTreeSet;

impl SampledVolumeFieldTopology {
    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        let data = &self.data;
        let unique = |ids: &[StableDigest]| {
            !ids.is_empty()
                && !ids.contains(&StableDigest::ZERO)
                && ids.iter().collect::<BTreeSet<_>>().len() == ids.len()
        };
        if data.schema_version != 1
            || !(FieldTopologyIdentity::SampledVolumeMesh {
                mesh_digest: data.mesh_digest,
                sampling_digest: self.sampling_digest,
            })
            .has_valid_identity()
            || data.projection_digest == Digest::from_bytes([0; 32])
            || data.projection_digest == Digest::sha256([])
            || !unique(&data.node_identities)
            || !unique(&data.element_identities)
            || data.element_corner_identities.len() != data.element_identities.len()
            || data.oriented_edges.is_empty()
            || data
                .element_identities
                .len()
                .checked_mul(data.rule.points_per_element())
                != Some(data.integration_samples.len())
            || self.sampling_digest != Digest::sha256(encode("sampling-layout", data)?)
        {
            return Err(invalid(
                "sampled volume topology identity or complete inventory is invalid",
            ));
        }
        let nodes: BTreeSet<_> = data.node_identities.iter().copied().collect();
        let mut expected_edges = BTreeSet::new();
        for corners in &data.element_corner_identities {
            if corners.iter().collect::<BTreeSet<_>>().len() != 4
                || corners.iter().any(|identity| !nodes.contains(identity))
            {
                return Err(invalid("sampled element corner inventory is invalid"));
            }
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let mut endpoints = [corners[a], corners[b]];
                    endpoints.sort_unstable();
                    expected_edges.insert(endpoints);
                }
            }
        }
        if data
            .oriented_edges
            .iter()
            .map(|edge| edge.node_identities)
            .collect::<Vec<_>>()
            != expected_edges.into_iter().collect::<Vec<_>>()
        {
            return Err(invalid(
                "sampled edge inventory is not the complete canonical corner-edge space",
            ));
        }
        let mut previous = None;
        for edge in &data.oriented_edges {
            let [a, b] = edge.node_identities;
            if a >= b
                || !nodes.contains(&a)
                || !nodes.contains(&b)
                || previous.is_some_and(|value| value >= edge.node_identities)
                || edge.identity != stable_identity("oriented-edge", &edge.node_identities)?
            {
                return Err(invalid(
                    "oriented edge identities/endpoints/order are invalid",
                ));
            }
            previous = Some(edge.node_identities);
        }
        for (element, samples) in data.element_identities.iter().zip(
            data.integration_samples
                .chunks_exact(data.rule.points_per_element()),
        ) {
            for (ordinal, sample) in samples.iter().enumerate() {
                if sample.element_identity != *element
                    || sample.ordinal != ordinal
                    || sample.identity
                        != stable_identity("integration-sample", &(data.rule, *element, ordinal))?
                    || !sample.volume_weight_m3.is_finite()
                    || sample.volume_weight_m3 <= 0.0
                    || sample
                        .position_m
                        .iter()
                        .any(|coordinate| !coordinate.is_finite())
                {
                    return Err(invalid(
                        "integration sample identity/order/physical geometry is invalid",
                    ));
                }
            }
        }
        self.entity_counts().validate()
    }
}
