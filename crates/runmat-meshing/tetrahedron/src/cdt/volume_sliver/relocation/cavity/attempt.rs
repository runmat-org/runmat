//! Prepare a local remesh, then delegate all geometry and quality authority.
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
pub(super) mod continuation;
mod refill;
use crate::cdt::volume_sliver::insertion::{
    refill::{preserve_vertices, RefillPoint},
    visibility::{cavity_boundary_faces, grow_visibility_cavity},
    VisibilityCavityRequest,
};
use crate::cdt::{mutation::CavityRelocation, volume_quality::working::SpectrumDelta};
#[derive(Clone, Copy)]
pub(super) enum CavityGrowth {
    Visibility,
    Quality,
}
pub(super) struct CavityAttempt<'a> {
    pub(super) growth: CavityGrowth,
    pub(super) source: &'a DelaunayTetrahedronQuality,
    pub(super) primary: u32,
    pub(super) point: [f64; 3],
    pub(super) baseline: &'a SpectrumDelta,
    pub(super) policy: SliverSearchPolicy,
    pub(super) prefix: &'a [DelaunayVolumeSliverMutation],
}
pub(super) fn attempt<'ctx>(
    state: &mut impl SearchState<'ctx>,
    request: CavityAttempt<'_>,
    work: &mut OptimizationWork<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<refill::RefillCandidate>, DelaunayVolumeSliverError> {
    let CavityAttempt {
        source,
        primary,
        point,
        baseline: _,
        policy,
        prefix: _,
        growth: _,
    } = request;
    let topology = state.topology();
    let source_index = topology
        .tetrahedron_index(
            source
                .node_identities
                .map(|id| topology.node_index(id).expect("source corner")),
        )
        .expect("source cell");
    let Some(neighbor_index) = topology
        .cell(source_index)
        .neighbors
        .into_iter()
        .flatten()
        .find(|cell| topology.cell(*cell).region_id.as_ref() == Some(&source.region_id))
    else {
        return Ok(None);
    };
    let primary_old = topology.node(primary);
    let replacement =
        DelaunayVolumeNode::explicit(replacement_identity(primary_old.identity), point);
    let metrics = |face: [u32; 3]| {
        state
            .preparation()
            .insertion_metric(
                face.map(|vertex| topology.node(vertex).identity),
                &source.region_id,
            )
            .map_err(crate::cdt::volume_sliver::quality_error)
    };
    let cavity = grow_visibility_cavity(
        topology,
        VisibilityCavityRequest {
            node: replacement,
            source_index,
            neighbor_index,
            relocated_vertices: &[primary],
            protected_faces: state.preparation().protected_facets(),
            quality_growth: match request.growth {
                CavityGrowth::Visibility => None,
                CavityGrowth::Quality => Some((
                    &metrics,
                    state.quality_options().minimum_metric_scaled_jacobian,
                )),
            },
            options: policy.options,
            cancellation,
        },
    )?;
    let (removed, region) = match cavity {
        Ok(cavity) => cavity,
        Err(reason) => {
            tracing::debug!(%reason,?point,"joint cavity visibility rejected");
            return Ok(None);
        }
    };
    let boundary = cavity_boundary_faces(topology, &removed)
        .into_iter()
        .map(|(_, _, face)| face)
        .collect::<Vec<_>>();
    // Judge the supplied geometric proposal before numerical smoothing can
    // change its intended boundary height. All untouched points are retained.
    work.evaluate()?;
    let direct = refill::RefillTrial {
        request: &request,
        removed: &removed,
        region: &region,
        boundary: &boundary,
        replacements: vec![(primary, replacement)],
    };
    direct.apply(state, work, cancellation)
}
