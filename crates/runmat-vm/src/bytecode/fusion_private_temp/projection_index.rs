//! One graph/store inventory for all sealed private edges in a candidate.
//! Index entries only locate existing owners; every edge is still checked
//! against the current graph bindings before its private binding is removed.

use std::collections::HashMap;

use crate::accel::graph::AccelStore;
use runmat_accelerate::graph::{AccelGraph, NodeId, ValueId};
use runmat_accelerate::{FusionGroup, FusionKind};

pub(super) struct ProjectionIndex {
    producers_by_end: HashMap<usize, Vec<usize>>,
    consumers_by_start: HashMap<usize, Vec<usize>>,
    stores_by_pc: HashMap<usize, Vec<usize>>,
    consumers_by_value: HashMap<ValueId, Vec<NodeId>>,
    reduction_ranges: Vec<(usize, usize)>,
    reduction_prefix_max_end: Vec<usize>,
}

impl ProjectionIndex {
    pub(super) fn new(graph: &AccelGraph, groups: &[FusionGroup], stores: &[AccelStore]) -> Self {
        let mut reduction_ranges = groups
            .iter()
            .filter(|group| group.kind == FusionKind::Reduction)
            .map(|group| (group.span.start, group.span.end))
            .collect::<Vec<_>>();
        reduction_ranges.sort_unstable();
        let mut max_end = 0;
        let reduction_prefix_max_end = reduction_ranges
            .iter()
            .map(|(_, end)| {
                max_end = max_end.max(*end);
                max_end
            })
            .collect();
        let mut index = Self {
            producers_by_end: HashMap::new(),
            consumers_by_start: HashMap::new(),
            stores_by_pc: HashMap::new(),
            consumers_by_value: HashMap::new(),
            reduction_ranges,
            reduction_prefix_max_end,
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
        let upper = self
            .reduction_ranges
            .partition_point(|(start, _)| *start <= producer_pc);
        let lower =
            self.reduction_prefix_max_end[..upper].partition_point(|end| *end < consumer_pc);
        self.reduction_ranges[lower..upper]
            .iter()
            .any(|(_, end)| *end >= consumer_pc)
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
