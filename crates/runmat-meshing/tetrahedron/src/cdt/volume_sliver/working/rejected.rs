//! Complete relocation searches that admitted nothing, with the neighborhood that decided them.
//!
//! The complete relocation handoff converts the working state to canonical form and builds every
//! candidate as a whole topology, so one search costs tens of full-mesh rebuilds. In coupled
//! convergence a rejected source stays the worst defect while refinement is accepted elsewhere,
//! and every accepted refinement batch used to rerun the same search with the same outcome.
//!
//! A search reads only the neighborhood of its source: the candidates are local edits, the
//! current topology is valid and Delaunay away from them. A candidate fails either for a reason
//! that the neighborhood alone decides (topology validation, embedding, refill), or because its
//! quality spectrum does not improve on the current one. That comparison is a multiset order
//! over violation priority keys: it is decided by the largest key whose count differs, so cells
//! the candidate shares with the current state cannot change it. The record therefore keeps the
//! source cell, the objective, the closed star of every node a candidate changed, and the
//! quality ratios of the cells each ranked-out candidate removed and added.
//!
//! The coupled comparison counts only keys at or above a floor, the worst violation of the whole
//! state, which moves as refinement proceeds elsewhere. A later pass with the same neighborhood
//! re-ranks the recorded changes against the current floor; the search reruns when the
//! neighborhood differs or a recorded candidate would now be admitted.
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

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

/// The search a record stands for: the source cell and the objective.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::cdt) struct CompleteSearchKey {
    source: [StableDigest; 4],
    objective: Objective,
}

impl CompleteSearchKey {
    pub(in crate::cdt) fn new(source: [StableDigest; 4], objective: QualityObjective) -> Self {
        let mut source = source;
        source.sort_unstable();
        Self {
            source,
            objective: objective.into(),
        }
    }
}

/// How a candidate's quality is ranked against the current state (see
/// `admission::candidate_quality_is_admissible`).
#[derive(Clone, Copy, Debug)]
pub(in crate::cdt) struct Ranking {
    /// Coupled convergence: both ratios of every cell count, at keys at or above `floor`.
    /// Otherwise the objective ratio counts at every key, and the refinement spectrum may not
    /// regress.
    pub(in crate::cdt) coupled: bool,
    pub(in crate::cdt) objective: QualityObjective,
    /// The violation priority key of the worst value of the current coupled spectrum, or 0.
    pub(in crate::cdt) floor: u64,
}

/// The (optimization, refinement) violation ratios of the cells one candidate removed and added.
/// The candidate passed every check that the neighborhood decides and lost the quality ranking.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::cdt) struct RankedOutCandidate {
    pub(in crate::cdt) removed: Vec<[f64; 2]>,
    pub(in crate::cdt) added: Vec<[f64; 2]>,
}

impl RankedOutCandidate {
    /// Whether the candidate would be admitted under `ranking`.
    pub(in crate::cdt) fn admissible(&self, ranking: Ranking) -> bool {
        if ranking.coupled {
            let both = |cell: &[f64; 2]| cell.to_vec();
            return self.compare(both, ranking.floor) == Ordering::Less;
        }
        let objective = |cell: &[f64; 2]| {
            vec![match ranking.objective {
                QualityObjective::Shape => cell[0],
                QualityObjective::Refinement => cell[1],
            }]
        };
        self.compare(objective, 0) == Ordering::Less
            && self.compare(|cell: &[f64; 2]| vec![cell[1]], 0) != Ordering::Greater
    }

    /// The spectrum order of the candidate against the current state, from the changed cells:
    /// the sign of the count difference at the largest priority key (at or above `floor`) whose
    /// counts differ. Only violating values (above 1) are in a spectrum.
    fn compare(&self, values: impl Fn(&[f64; 2]) -> Vec<f64>, floor: u64) -> Ordering {
        let mut counts = BTreeMap::<u64, i64>::new();
        for (cells, step) in [(&self.added, 1), (&self.removed, -1)] {
            for value in cells.iter().flat_map(&values) {
                if value > 1.0 {
                    let key = crate::cdt::volume_quality::violation_priority_key(value);
                    if key >= floor {
                        *counts.entry(key).or_default() += step;
                    }
                }
            }
        }
        counts
            .into_iter()
            .rev()
            .find(|(_, count)| *count != 0)
            .map_or(Ordering::Equal, |(_, count)| count.cmp(&0))
    }
}

/// What a complete search observed: the nodes of every cell a produced candidate changed, and
/// the ranked-out candidates.
#[derive(Default)]
pub(in crate::cdt) struct SearchRecording {
    pub(in crate::cdt) changed: BTreeSet<StableDigest>,
    pub(in crate::cdt) ranked_out: Vec<RankedOutCandidate>,
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
    ranked_out: Vec<RankedOutCandidate>,
}

/// Rejected complete relocation searches of one convergence run.
#[derive(Default)]
pub(in crate::cdt) struct RejectedCompleteRelocations {
    entries: BTreeMap<CompleteSearchKey, Rejection>,
}

impl RejectedCompleteRelocations {
    /// Whether a search with this key was rejected, nothing it read has changed since, and no
    /// ranked-out candidate would be admitted under `ranking`. A record that no longer holds is
    /// dropped.
    pub(in crate::cdt) fn still_rejected<T: TetrahedralTopology + ?Sized>(
        &mut self,
        topology: &T,
        key: &CompleteSearchKey,
        ranking: Ranking,
    ) -> bool {
        let Some(rejection) = self.entries.get(key) else {
            return false;
        };
        let holds = Neighborhood::capture(topology, &rejection.region)
            .is_some_and(|current| current == rejection.neighborhood)
            && !rejection
                .ranked_out
                .iter()
                .any(|candidate| candidate.admissible(ranking));
        if !holds {
            self.entries.remove(key);
        }
        holds
    }

    /// Records a search that admitted nothing. `recording.changed` holds every node of a cell
    /// that some candidate added or removed; the source cell's nodes and their one-ring are
    /// added here, so candidates that failed before producing a topology are covered by the
    /// region too.
    pub(in crate::cdt) fn record<T: TetrahedralTopology + ?Sized>(
        &mut self,
        topology: &T,
        key: CompleteSearchKey,
        recording: SearchRecording,
    ) {
        let SearchRecording {
            changed,
            ranked_out,
        } = recording;
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
                ranked_out,
            },
        );
    }
}

#[cfg(test)]
mod tests;
