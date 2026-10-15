//! Conservative window closure over compiler-certified private data edges.
//! This only proposes a span; graph dependency/fanout and typed execution
//! admission remain with the existing fusion plan and recipe owners.

use std::collections::{BTreeMap, HashMap, HashSet};

use runmat_accelerate::{FusionGroup, FusionKind};

use super::FusionPrivateTempBridge;

struct GroupIntervals {
    ordered: Vec<(usize, usize, usize)>,
    prefix_max_end: Vec<usize>,
}

impl GroupIntervals {
    fn new(groups: &[FusionGroup]) -> Self {
        let mut ordered = groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.span.start, group.span.end, index))
            .collect::<Vec<_>>();
        ordered.sort_unstable();
        let mut max_end = 0;
        let prefix_max_end = ordered
            .iter()
            .map(|(_, end, _)| {
                max_end = max_end.max(*end);
                max_end
            })
            .collect();
        Self {
            ordered,
            prefix_max_end,
        }
    }

    fn has_unselected_overlap(&self, start: usize, end: usize, selected: &HashSet<usize>) -> bool {
        let upper = self.ordered.partition_point(|(from, _, _)| *from <= end);
        let lower = self.prefix_max_end[..upper].partition_point(|max_end| *max_end < start);
        self.ordered[lower..upper]
            .iter()
            .any(|(_, to, index)| *to >= start && !selected.contains(index))
    }
}

pub(super) fn reduction_groups(groups: &mut Vec<FusionGroup>, bridges: &[FusionPrivateTempBridge]) {
    close_groups(groups, bridges, false);
}

pub(super) fn chain_groups(groups: &mut Vec<FusionGroup>, bridges: &[FusionPrivateTempBridge]) {
    close_groups(groups, bridges, true);
}

fn close_groups(groups: &mut Vec<FusionGroup>, bridges: &[FusionPrivateTempBridge], chains: bool) {
    let intervals = GroupIntervals::new(groups);
    let producers = groups
        .iter()
        .enumerate()
        .filter(|(_, group)| group.kind == FusionKind::ElementwiseChain)
        .map(|(index, group)| (group.span.end, index))
        .collect::<HashMap<_, _>>();
    if producers.len()
        != groups
            .iter()
            .filter(|group| group.kind == FusionKind::ElementwiseChain)
            .count()
    {
        return;
    }
    let mut incoming: BTreeMap<usize, Vec<&FusionPrivateTempBridge>> = BTreeMap::new();
    for bridge in bridges {
        incoming.entry(bridge.consumer_pc).or_default().push(bridge);
    }
    let mut removed = HashSet::new();
    let mut widened: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    // Visit sinks first so a partial prefix does not consume the producers
    // needed by a later, complete dependency closure.
    for root_index in (0..groups.len()).rev() {
        let root = &groups[root_index];
        if removed.contains(&root_index)
            || !(root.kind == FusionKind::Reduction
                || (chains && root.kind == FusionKind::ElementwiseChain))
            || (!chains && root.span.start != root.span.end)
        {
            continue;
        }
        let mut selected = HashSet::from([root_index]);
        let mut pending = vec![root_index];
        let mut valid = true;
        while let Some(index) = pending.pop() {
            let span = &groups[index].span;
            for (_, edges) in incoming.range(span.start..=span.end) {
                for bridge in edges {
                    let Some(&producer_index) = producers.get(&bridge.producer_pc) else {
                        valid = false;
                        break;
                    };
                    if producer_index >= root_index || removed.contains(&producer_index) {
                        valid = false;
                        break;
                    }
                    if selected.insert(producer_index) {
                        pending.push(producer_index);
                    }
                }
                if !valid {
                    break;
                }
            }
            if !valid {
                break;
            }
        }
        if !valid || selected.len() <= 1 {
            continue;
        }
        let start = selected
            .iter()
            .map(|index| groups[*index].span.start)
            .min()
            .expect("reduction root belongs to selected");
        let end = groups[root_index].span.end;
        // No independent signal window may be swallowed by the proposed
        // region. A later graph check also rejects unrelated contained nodes.
        let prior_widened_overlap =
            widened
                .range(..=end)
                .next_back()
                .is_some_and(|(_, (prior_end, index))| {
                    *prior_end >= start && !selected.contains(index)
                });
        if intervals.has_unselected_overlap(start, end, &selected) || prior_widened_overlap {
            continue;
        }
        groups[root_index].span.start = start;
        groups[root_index].stack_layout = None;
        widened.insert(start, (end, root_index));
        removed.extend(selected.into_iter().filter(|index| *index != root_index));
    }
    if !removed.is_empty() {
        *groups = groups
            .drain(..)
            .enumerate()
            .filter_map(|(index, group)| (!removed.contains(&index)).then_some(group))
            .collect();
    }
    for (id, group) in groups.iter_mut().enumerate() {
        group.id = id;
    }
}
