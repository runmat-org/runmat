//! One mutable topology owner throughout facet recovery. Read-only support uses
//! maintained incidence; canonical materialization is a provider/output boundary.
use super::*;
use crate::cdt::{
    insertion::topology_access::TetrahedralTopology,
    mutation::{TopologyMutation, TopologyMutationError},
};

pub(super) struct FacetRecoveryWorking {
    topology: TopologyMutation,
    /// The canonical form of `topology` while no edit has touched it since a provider produced
    /// or consumed it. Consecutive canonical providers reuse it instead of re-materializing the
    /// whole complex; every mutable access clears it.
    canonical: Option<DelaunayVolumeTopology>,
    pub(super) segments: Vec<crate::cdt::DelaunayRecoveredSegment>,
    steiner_node_identities: Vec<StableDigest>,
    recovery_passes: u32,
}

impl FacetRecoveryWorking {
    pub(super) fn new(source: &DelaunaySegmentRecovery) -> Self {
        Self {
            topology: TopologyMutation::from_canonical(source.topology.clone()),
            canonical: Some(source.topology.clone()),
            segments: source.segments.clone(),
            steiner_node_identities: source.steiner_node_identities.clone(),
            recovery_passes: source.recovery_passes,
        }
    }

    /// Existing cavity/refinement providers consume canonical contracts. This
    /// bridge is used only when a local support/flip cannot satisfy the request.
    /// It never becomes the ordinary read or successful local-flip path.
    pub(super) fn canonical_call<T>(
        &mut self,
        constraint_index: u32,
        work: &mut FacetRecoveryWork<'_>,
        provider: impl FnOnce(
            &mut DelaunaySegmentRecovery,
            &mut FacetRecoveryWork<'_>,
        ) -> Result<T, DelaunayFacetRecoveryError>,
    ) -> Result<T, DelaunayFacetRecoveryError> {
        let topology = match self.canonical.take() {
            Some(topology) => topology,
            None => self
                .topology
                .snapshot(
                    work.options.segment_recovery.insertion.topology,
                    work.cancellation,
                )
                .map_err(|failure| mutation_error(constraint_index, failure))?,
        };
        let mut source = DelaunaySegmentRecovery {
            topology,
            segments: self.segments.clone(),
            steiner_node_identities: self.steiner_node_identities.clone(),
            recovery_passes: self.recovery_passes,
        };
        let result = provider(&mut source, work)?;
        // Providers replace the canonical topology wholesale and never edit it in place, so the
        // returned value is the canonical form of the rebuilt working topology.
        self.topology = TopologyMutation::from_canonical(source.topology.clone());
        self.canonical = Some(source.topology);
        self.segments = source.segments;
        self.steiner_node_identities = source.steiner_node_identities;
        self.recovery_passes = source.recovery_passes;
        Ok(result)
    }

    pub(super) fn topology(&self) -> &TopologyMutation {
        &self.topology
    }

    /// Runs a local edit and clears the retained canonical form only when the edit reports a
    /// change.
    pub(super) fn edit<R, E>(
        &mut self,
        edit: impl FnOnce(&mut TopologyMutation) -> Result<(R, bool), E>,
    ) -> Result<R, E> {
        let (result, changed) = edit(&mut self.topology)?;
        if changed {
            self.canonical = None;
        }
        Ok(result)
    }

    pub(super) fn record_segment_split(
        &mut self,
        segment_index: usize,
        interval_index: usize,
        node: crate::cdt::DelaunayRecoveredSegmentNode,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        crate::cdt::segment_recovery::record_recovered_segment_split_on(
            &mut self.segments,
            &mut self.steiner_node_identities,
            segment_index,
            interval_index,
            node,
        )
        .map_err(super::segment_error)
    }

    pub(super) fn steiner_node_count(&self) -> usize {
        self.steiner_node_identities.len()
    }

    pub(super) fn support_edges_exist(&self, triangle: [StableDigest; 3]) -> bool {
        triangle_edges(triangle).into_iter().all(|edge| {
            let [Some(first), Some(second)] = edge.map(|id| self.topology.node_index(id)) else {
                return false;
            };
            self.topology.has_edge([first, second])
        })
    }

    pub(super) fn finish(
        self,
        work: &FacetRecoveryWork<'_>,
    ) -> Result<
        (
            DelaunayVolumeTopology,
            Vec<crate::cdt::DelaunayRecoveredSegment>,
        ),
        DelaunayFacetRecoveryError,
    > {
        let topology = self
            .topology
            .freeze(
                work.options.segment_recovery.insertion.topology,
                work.cancellation,
            )
            .map_err(|failure| mutation_error(0, failure))?;
        Ok((topology, self.segments))
    }
}

pub(super) fn mutation_error(
    constraint_index: u32,
    failure: TopologyMutationError,
) -> DelaunayFacetRecoveryError {
    use crate::cdt::mutation::TopologyMutationErrorKind;
    let kind = match failure.kind {
        TopologyMutationErrorKind::Cancelled => DelaunayFacetRecoveryErrorKind::Cancelled,
        TopologyMutationErrorKind::ResourceLimit => DelaunayFacetRecoveryErrorKind::ResourceLimit,
        TopologyMutationErrorKind::InvalidTopology => {
            DelaunayFacetRecoveryErrorKind::InvalidTopology
        }
    };
    error(kind, Some(constraint_index), failure.to_string())
}
