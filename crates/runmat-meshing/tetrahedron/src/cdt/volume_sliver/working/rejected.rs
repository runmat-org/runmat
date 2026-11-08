//! Complete relocation searches that admitted nothing, with the neighborhood that decided them.
//!
//! The complete relocation handoff converts the working state to canonical form and builds every
//! candidate as a whole topology, so one search costs tens of full-mesh rebuilds. In coupled
//! convergence a rejected source stays the worst defect while refinement is accepted elsewhere,
//! and every accepted refinement batch used to rerun the same search with the same outcome.
//!
//! A search reads only the neighborhood of its source: the candidates are local edits, the
//! current topology is valid and Delaunay away from them, and the spectrum comparison of a
//! candidate with the current state does not depend on cells the two share. The outcome is
//! therefore fixed by the cells around every node a candidate changed, the source cell, the
//! objective and the worst violation priorities that set the comparison floor. A record keeps
//! those; the search reruns as soon as any of them differs.
use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_quality::QualityObjective;
use crate::cdt::DelaunayVolumeNode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Objective {
    Shape,
    Refinement,
}

impl From<QualityObjective> for Objective {
    fn from(objective: QualityObjective) -> Self {
        match objective {
            QualityObjective::Shape => Self::Shape,
            QualityObjective::Refinement => Self::Refinement,
        }
    }
}

/// What the search compared against: the source cell, the objective, and the worst shape and
/// refinement violation priorities of the whole state (the coupled comparison floor).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::cdt) struct CompleteSearchKey {
    source: [StableDigest; 4],
    objective: Objective,
    worst_priorities: [Option<u64>; 2],
}

impl CompleteSearchKey {
    pub(in crate::cdt) fn new(
        source: [StableDigest; 4],
        objective: QualityObjective,
        worst_shape_priority: Option<u64>,
        worst_refinement_priority: Option<u64>,
    ) -> Self {
        let mut source = source;
        source.sort_unstable();
        Self {
            source,
            objective: objective.into(),
            worst_priorities: [worst_shape_priority, worst_refinement_priority],
        }
    }
}

/// The closed star of the region nodes: every cell incident to one of them, with its nodes and
/// region. Equal neighborhoods give equal candidates and equal admission outcomes.
#[derive(Clone, Debug, PartialEq)]
struct Neighborhood {
    cells: BTreeSet<([StableDigest; 4], Option<PersistentEntityId>)>,
    nodes: BTreeMap<StableDigest, DelaunayVolumeNode>,
}

impl Neighborhood {
    /// `None` when a region node no longer exists.
    fn capture<T: TetrahedralTopology + ?Sized>(
        topology: &T,
        region: &BTreeSet<StableDigest>,
    ) -> Option<Self> {
        let mut cells = BTreeSet::new();
        let mut nodes = BTreeMap::new();
        for identity in region {
            let vertex = topology.node_index(*identity)?;
            for cell in topology.vertex_star(vertex) {
                let cell = topology.tetrahedron(cell)?;
                let corners = cell
                    .vertex_indices
                    .map(|index| topology.nodes()[index as usize]);
                let mut key = corners.map(|node| node.identity);
                key.sort_unstable();
                for node in corners {
                    nodes.insert(node.identity, node);
                }
                cells.insert((key, cell.region_id.clone()));
            }
        }
        Some(Self { cells, nodes })
    }
}

struct Rejection {
    region: BTreeSet<StableDigest>,
    neighborhood: Neighborhood,
}

/// Rejected complete relocation searches of one convergence run.
#[derive(Default)]
pub(in crate::cdt) struct RejectedCompleteRelocations {
    entries: BTreeMap<CompleteSearchKey, Rejection>,
}

impl RejectedCompleteRelocations {
    /// Whether a search with this key was rejected and nothing it read has changed since.
    /// A stale record is dropped.
    pub(in crate::cdt) fn still_rejected<T: TetrahedralTopology + ?Sized>(
        &mut self,
        topology: &T,
        key: &CompleteSearchKey,
    ) -> bool {
        let Some(rejection) = self.entries.get(key) else {
            return false;
        };
        let unchanged = Neighborhood::capture(topology, &rejection.region)
            .is_some_and(|current| current == rejection.neighborhood);
        if !unchanged {
            self.entries.remove(key);
        }
        unchanged
    }

    /// Records a search that admitted nothing. `changed` holds every node of a cell that some
    /// candidate added or removed; the source cell's nodes and their one-ring are added here, so
    /// candidates that failed before producing a topology are covered by the region too.
    pub(in crate::cdt) fn record<T: TetrahedralTopology + ?Sized>(
        &mut self,
        topology: &T,
        key: CompleteSearchKey,
        changed: BTreeSet<StableDigest>,
    ) {
        let mut region = changed;
        for identity in key.source {
            let Some(vertex) = topology.node_index(identity) else {
                return;
            };
            region.insert(identity);
            for cell in topology.vertex_star(vertex) {
                if let Some(cell) = topology.tetrahedron(cell) {
                    region.extend(
                        cell.vertex_indices
                            .map(|index| topology.nodes()[index as usize].identity),
                    );
                }
            }
        }
        // A changed node that the current topology lacks (a candidate's replacement node) is not
        // part of the current neighborhood; the cells around its neighbors cover its position.
        region.retain(|identity| topology.node_index(*identity).is_some());
        let Some(neighborhood) = Neighborhood::capture(topology, &region) else {
            return;
        };
        self.entries.insert(
            key,
            Rejection {
                region,
                neighborhood,
            },
        );
    }
}

#[cfg(test)]
mod tests;
