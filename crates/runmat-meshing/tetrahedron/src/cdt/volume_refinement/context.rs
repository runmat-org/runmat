//! Borrowed authorities shared by refinement and optimization of owned state.
use super::*;

#[derive(Clone, Copy)]
pub(in crate::cdt) struct RefinementContext<'a> {
    pub(in crate::cdt) metric_request: &'a MetricFieldRequest,
    pub(in crate::cdt) provenance: &'a DelaunayVolumeProvenance,
    pub(in crate::cdt) quality_options: DelaunayVolumeQualityOptions,
    pub(in crate::cdt) fixed_node_identities: &'a [StableDigest],
}

impl<'a> From<DelaunayVolumeRefinementInput<'a>> for RefinementContext<'a> {
    fn from(input: DelaunayVolumeRefinementInput<'a>) -> Self {
        Self {
            metric_request: input.metric_request,
            provenance: input.provenance,
            quality_options: input.quality_options,
            fixed_node_identities: input.fixed_node_identities,
        }
    }
}

impl RefinementContext<'_> {
    pub(in crate::cdt) fn with_state<'a>(
        &'a self,
        topology: &'a DelaunayVolumeTopology,
        quality: &'a DelaunayVolumeQuality,
    ) -> DelaunayVolumeRefinementInput<'a> {
        DelaunayVolumeRefinementInput {
            topology,
            quality,
            metric_request: self.metric_request,
            provenance: self.provenance,
            quality_options: self.quality_options,
            fixed_node_identities: self.fixed_node_identities,
        }
    }
}
