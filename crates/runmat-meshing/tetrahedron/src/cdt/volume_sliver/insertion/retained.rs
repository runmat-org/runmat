//! Retain insertion trials as exact differences, not complete mesh inventories.
use super::super::retained_state::{QualityPatch, TopologyPatch};
use super::search::InsertionCandidate;
use super::*;

pub(super) struct RetainedInsertion {
    topology: TopologyPatch,
    quality: QualityPatch,
    mutations: Vec<DelaunayVolumeSliverMutation>,
    pub(super) last_node_identity: Option<StableDigest>,
}

impl RetainedInsertion {
    pub(super) fn capture(
        base: DelaunayVolumeRefinementInput<'_>,
        topology: &DelaunayVolumeTopology,
        quality: &DelaunayVolumeQuality,
        mutations: Vec<DelaunayVolumeSliverMutation>,
    ) -> Result<Self, DelaunayVolumeSliverError> {
        Ok(Self {
            topology: TopologyPatch::between(base.topology, topology)?,
            quality: QualityPatch::between(base.quality, quality)?,
            mutations,
            last_node_identity: topology.nodes.last().map(|node| node.identity),
        })
    }

    pub(super) fn compare_candidate(
        &self,
        candidate: &DelaunayVolumeQuality,
        base: &DelaunayVolumeQuality,
        policy: SliverSearchPolicy,
    ) -> std::cmp::Ordering {
        self.quality.compare_candidate(candidate, base, policy)
    }

    pub(super) fn materialize(
        self,
        base: DelaunayVolumeRefinementInput<'_>,
        options: DelaunayVolumeSliverOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<InsertionCandidate, DelaunayVolumeSliverError> {
        Ok(InsertionCandidate {
            topology: self
                .topology
                .materialize(base.topology, options, cancellation)?,
            quality: self.quality.materialize(base.quality),
            mutations: self.mutations,
        })
    }
}
