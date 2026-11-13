//! One graph/store inventory for all sealed private edges in a candidate.
//! Index entries only locate existing owners; every edge is still checked
//! against the current graph bindings before its private binding is removed.

use std::collections::HashMap;

use crate::accel::graph::AccelStore;
use runmat_accelerate::graph::{AccelGraph, NodeId, ValueId};
use runmat_accelerate::{FusionGroup, FusionKind};

struct CoverageIndex {
    ranges: Vec<(usize, usize)>,
    prefix_max_end: Vec<usize>,
}

impl CoverageIndex {
    fn new(ranges: impl Iterator<Item = (usize, usize)>) -> Self {
        let mut ranges = ranges.collect::<Vec<_>>();
        ranges.sort_unstable();
        let mut max_end = 0;
        let prefix_max_end = ranges
            .iter()
            .map(|(_, end)| {
                max_end = max_end.max(*end);
                max_end
            })
            .collect();
        Self {
            ranges,
            prefix_max_end,
        }
    }

    fn covers(&self, producer_pc: usize, consumer_pc: usize) -> bool {
        let upper = self
            .ranges
            .partition_point(|(start, _)| *start <= producer_pc);
        // All ranges in this prefix begin before the producer. Its maximum
        // end therefore proves whether any single range covers both PCs.
        upper
            .checked_sub(1)
            .is_some_and(|last| self.prefix_max_end[last] >= consumer_pc)
    }
}

pub(super) struct ProjectionIndex {
    producers_by_end: HashMap<usize, Vec<usize>>,
    consumers_by_start: HashMap<usize, Vec<usize>>,
    stores_by_pc: HashMap<usize, Vec<usize>>,
    consumers_by_value: HashMap<ValueId, Vec<NodeId>>,
    regions: CoverageIndex,
    reductions: CoverageIndex,
}

impl ProjectionIndex {
    pub(super) fn new(graph: &AccelGraph, groups: &[FusionGroup], stores: &[AccelStore]) -> Self {
        let regions = CoverageIndex::new(
            groups
                .iter()
                .filter(|group| {
                    matches!(
                        group.kind,
                        FusionKind::Reduction
                            | FusionKind::ElementwiseChain
                            | FusionKind::ImageNormalize
                    )
                })
                .map(|group| (group.span.start, group.span.end)),
        );
        let mut index = Self {
            producers_by_end: HashMap::new(),
            consumers_by_start: HashMap::new(),
            stores_by_pc: HashMap::new(),
            consumers_by_value: HashMap::new(),
            regions,
            reductions: CoverageIndex::new(
                groups
                    .iter()
                    .filter(|group| {
                        matches!(
                            group.kind,
                            FusionKind::Reduction | FusionKind::ImageNormalize
                        )
                    })
                    .map(|group| (group.span.start, group.span.end)),
            ),
        };
        for (position, node) in graph.nodes.iter().enumerate() {
            index
                .producers_by_end
                .entry(node.span.end)
                .or_default()
                .push(position);
            index
                .consumers_by_start
                .entry(node.span.start)
                .or_default()
                .push(position);
            for &input in &node.inputs {
                index
                    .consumers_by_value
                    .entry(input)
                    .or_default()
                    .push(node.id);
            }
        }
        for (position, store) in stores.iter().enumerate() {
            index
                .stores_by_pc
                .entry(store.pc)
                .or_default()
                .push(position);
        }
        index
    }

    pub(super) fn producer_at(&self, pc: usize) -> Option<usize> {
        let [position] = self.producers_by_end.get(&pc)?.as_slice() else {
            return None;
        };
        Some(*position)
    }

    pub(super) fn reduction_covers(&self, producer_pc: usize, consumer_pc: usize) -> bool {
        self.reductions.covers(producer_pc, consumer_pc)
    }

    pub(super) fn region_covers(&self, producer_pc: usize, consumer_pc: usize) -> bool {
        self.regions.covers(producer_pc, consumer_pc)
    }

    pub(super) fn consumer_at(&self, pc: usize) -> Option<usize> {
        let [position] = self.consumers_by_start.get(&pc)?.as_slice() else {
            return None;
        };
        Some(*position)
    }

    pub(super) fn store_at(&self, pc: usize) -> Option<usize> {
        let [position] = self.stores_by_pc.get(&pc)?.as_slice() else {
            return None;
        };
        Some(*position)
    }

    pub(super) fn has_other_consumer(&self, value: ValueId, consumer: NodeId) -> bool {
        self.consumers_by_value
            .get(&value)
            .is_some_and(|nodes| nodes.iter().any(|node| *node != consumer))
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::CoverageIndex;

    #[test]
    fn indexed_coverage_matches_individual_intervals_including_nested_and_disjoint_ranges() {
        for ranges in [
            vec![],
            vec![(0, 0)],
            vec![(8, 9), (1, 7), (2, 3), (1, 5), (12, 14)],
        ] {
            let index = CoverageIndex::new(ranges.iter().copied());
            for producer in 0..=15 {
                for consumer in producer..=15 {
                    assert_eq!(
                        index.covers(producer, consumer),
                        ranges
                            .iter()
                            .any(|(start, end)| *start <= producer && consumer <= *end)
                    );
                }
            }
        }
    }
}
