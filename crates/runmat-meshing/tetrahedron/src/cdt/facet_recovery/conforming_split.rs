use std::collections::BTreeSet;

use runmat_meshing_core::{
    quality::predicate::{
        points_collinear_constructed, projected_orient2d_constructed,
        triangle_projection_axes_constructed, PredicateSign,
    },
    StableDigest,
};

use super::{
    cavity::recovered_segments_exist, error, node_index, DelaunayFacetRecoveryError,
    DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle, DelaunaySegmentRecovery,
    FacetRecoveryWork,
};
use crate::cdt::{
    insertion::{
        legalize_delaunay_volume_topology_with_barriers, legalize_inserted_node_and_check_star,
        topology_access::TetrahedralTopology, validate_constrained_delaunay_volume_topology,
    },
    mutation::{TopologyMutation, TopologyMutationErrorKind},
    segment_recovery::{
        record_recovered_segment_split, recovered_segment_midpoint, recovered_segment_midpoint_on,
    },
    topology::{orient_materialized_volume_nodes, orient_volume_nodes},
    DelaunayInsertionErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology,
};

mod cascade;
mod cavity;
mod corner;
mod insertion;
mod local;
mod pair;
pub(super) use cascade::segment_edge_location;
use cascade::*;
use cavity::*;
use corner::*;
use insertion::*;
use local::*;
#[cfg(test)]
pub(super) use pair::shared_vertex_split_points;
use pair::*;

pub(super) struct ProtectedFacetSplit {
    pub(super) source: DelaunayRecoveredFacetTriangle,
    pub(super) children: Vec<DelaunayRecoveredFacetTriangle>,
}

pub(super) struct ConformingInsertion {
    pub(super) recovery: DelaunaySegmentRecovery,
    pub(super) splits: Vec<ProtectedFacetSplit>,
}

pub(super) fn apply_protected_facet_splits(
    facets: &mut [super::DelaunayRecoveredFacet],
    protected: &mut Vec<DelaunayRecoveredFacetTriangle>,
    splits: Vec<ProtectedFacetSplit>,
) {
    for split in splits {
        for facet in facets.iter_mut() {
            if let Some(index) = facet
                .triangles
                .iter()
                .position(|triangle| *triangle == split.source)
            {
                facet
                    .triangles
                    .splice(index..=index, split.children.clone());
            }
        }
        if let Some(index) = protected
            .iter()
            .position(|triangle| *triangle == split.source)
        {
            protected.splice(index..=index, split.children);
        }
    }
}

enum PointLocation {
    Interior,
    Edge([StableDigest; 2]),
}

pub(super) fn try_insert_on_protected_facets(
    current: &DelaunaySegmentRecovery,
    node: DelaunayVolumeNode,
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    let mut hits = Vec::new();
    for triangle in protected {
        work.support_step(constraint_index)?;
        if let Some(location) = point_location(current, node, *triangle, constraint_index)? {
            hits.push((*triangle, location));
        }
    }
    if hits.is_empty() {
        return Ok(None);
    }
    let edge = hits.iter().find_map(|(_, location)| match location {
        PointLocation::Interior => None,
        PointLocation::Edge(edge) => Some(*edge),
    });
    insert_conformingly(
        current,
        node,
        protected,
        edge,
        &hits,
        constraint_index,
        work,
    )
}

pub(super) fn try_split_recovered_segment_edge(
    current: &DelaunaySegmentRecovery,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<ConformingInsertion>, DelaunayFacetRecoveryError> {
    work.support_step(constraint_index)?;
    let Some((segment_index, interval_index)) = recovered_segment_edge_location(current, edge)
    else {
        return Ok(None);
    };
    let (node, _) = recovered_segment_midpoint(current, segment_index, interval_index)
        .map_err(super::segment_error)?;
    if current.topology.nodes.iter().any(|existing| {
        crate::cdt::segment_recovery::coordinates_share_representable_location(
            existing.coordinates_m.canonical_value(),
            node.coordinates_m.canonical_value(),
        )
    }) {
        work.reject("conflicting segment midpoint duplicates an existing node position");
        return Ok(None);
    }
    insert_conformingly(
        current,
        node,
        protected,
        Some(edge),
        &[],
        constraint_index,
        work,
    )
}

/// Splits a recovered segment edge directly on the working topology: the split, Lawson flips in
/// the new node's link and an exact admission of every changed cell run in a cavity trial, so
/// the cost follows the size of the change. Returns `None` with the topology unchanged when the
/// local repair cannot finish; the caller then runs the canonical split with complete
/// legalization.
///
/// The constrained Bowyer-Watson cavity of the midpoint can contain the whole star of another
/// recovered segment edge. The midpoint then lies strictly inside every circumsphere around
/// that edge, so the edge has no empty circumsphere once the midpoint exists and no
/// constrained-Delaunay complex contains both. That encroached edge is split first (bounded
/// cascade), and the original split is retried on the refined segments.
pub(super) fn split_recovered_segment_edge_in_place(
    working: &mut super::working::FacetRecoveryWorking,
    edge: [StableDigest; 2],
    protected: &[DelaunayRecoveredFacetTriangle],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<Vec<ProtectedFacetSplit>>, DelaunayFacetRecoveryError> {
    let protected = protected.to_vec();
    let protected_edges = protected_edges.to_vec();
    let mut trial_splits = Vec::new();
    // The cascade edits the working topology through committed splits; a declined cascade
    // leaves those committed encroachment splits in place, which are themselves valid
    // constrained-Delaunay refinements of recovered segments.
    let result = split_with_encroachment_cascade(
        working,
        EncroachmentCascade {
            edge,
            protected,
            protected_edges,
            depth: 0,
            parent: None,
        },
        constraint_index,
        work,
        &mut trial_splits,
    )?;
    Ok(if result || !trial_splits.is_empty() {
        Some(trial_splits)
    } else {
        None
    })
}

fn node_index_mutation(topology: &TopologyMutation, identity: StableDigest) -> Option<u32> {
    topology
        .nodes
        .iter()
        .position(|node| node.identity == identity)
        .map(|index| index as u32)
}

fn insertion_error(
    kind: DelaunayInsertionErrorKind,
    reason: String,
    constraint_index: u32,
) -> DelaunayFacetRecoveryError {
    let kind = match kind {
        DelaunayInsertionErrorKind::ResourceLimit => DelaunayFacetRecoveryErrorKind::ResourceLimit,
        DelaunayInsertionErrorKind::Cancelled => DelaunayFacetRecoveryErrorKind::Cancelled,
        _ => DelaunayFacetRecoveryErrorKind::InvalidTopology,
    };
    error(kind, Some(constraint_index), reason)
}

fn invalid(constraint_index: u32, reason: impl Into<String>) -> DelaunayFacetRecoveryError {
    error(
        DelaunayFacetRecoveryErrorKind::InvalidTopology,
        Some(constraint_index),
        reason,
    )
}
