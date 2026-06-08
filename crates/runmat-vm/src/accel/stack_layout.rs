use std::collections::{HashMap, HashSet};

use runmat_accelerate::graph::{AccelGraph, ValueId, ValueOrigin};
use runmat_accelerate::{FusionGroup, FusionStackLayout, FusionStackValueBinding};

use crate::instr::Instr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackValue {
    Unknown,
    GraphValue(ValueId),
}

pub fn annotate_fusion_groups_with_stack_layout(
    instructions: &[Instr],
    graph: &AccelGraph,
    groups: &mut [FusionGroup],
) {
    if groups.is_empty() {
        return;
    }

    let mut groups_by_start: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, group) in groups.iter().enumerate() {
        groups_by_start
            .entry(group.span.start)
            .or_default()
            .push(idx);
    }

    let node_output_by_pc: HashMap<usize, ValueId> = graph
        .nodes
        .iter()
        .filter_map(|node| {
            node.outputs
                .first()
                .copied()
                .map(|value_id| (node.span.end, value_id))
        })
        .collect();

    let mut stack: Vec<StackValue> = Vec::new();
    for (pc, instr) in instructions.iter().enumerate() {
        if let Some(group_indices) = groups_by_start.get(&pc) {
            for &group_idx in group_indices {
                groups[group_idx].stack_layout =
                    build_group_stack_layout(instructions, graph, &groups[group_idx], &stack);
            }
        }

        let Some(effect) = instr.stack_effect() else {
            stack.clear();
            continue;
        };
        for _ in 0..effect.pops {
            let _ = stack.pop();
        }
        if effect.pushes == 0 {
            continue;
        }

        let pushed_value = if effect.pushes == 1 {
            node_output_by_pc
                .get(&pc)
                .copied()
                .map(StackValue::GraphValue)
                .unwrap_or(StackValue::Unknown)
        } else {
            StackValue::Unknown
        };
        for _ in 0..effect.pushes {
            stack.push(pushed_value);
        }
    }
}

fn build_group_stack_layout(
    instructions: &[Instr],
    graph: &AccelGraph,
    group: &FusionGroup,
    entry_stack: &[StackValue],
) -> Option<FusionStackLayout> {
    let required_stack_operands =
        required_stack_operands(instructions, group.span.start, group.span.end)?;
    if required_stack_operands > entry_stack.len() {
        return None;
    }

    let stack_value_ids = stack_backed_external_values(graph, group);
    let slice_start = entry_stack.len().saturating_sub(required_stack_operands);
    let mut seen = HashSet::new();
    let mut bindings = Vec::new();

    for (absolute_offset, value) in entry_stack.iter().enumerate().skip(slice_start) {
        let StackValue::GraphValue(value_id) = value else {
            continue;
        };
        if !stack_value_ids.contains(value_id) || !seen.insert(*value_id) {
            continue;
        }
        bindings.push(FusionStackValueBinding {
            value_id: *value_id,
            stack_offset: absolute_offset - slice_start,
        });
    }

    Some(FusionStackLayout {
        required_stack_operands,
        bindings,
    })
}

fn stack_backed_external_values(graph: &AccelGraph, group: &FusionGroup) -> HashSet<ValueId> {
    let node_set: HashSet<_> = group.nodes.iter().copied().collect();
    let mut values = HashSet::new();
    for node_id in &group.nodes {
        let Some(node) = graph.node(*node_id) else {
            continue;
        };
        for value_id in &node.inputs {
            let Some(info) = graph.value(*value_id) else {
                continue;
            };
            match info.origin {
                ValueOrigin::NodeOutput { node, .. } if !node_set.contains(&node) => {
                    if graph.var_binding(*value_id).is_none() {
                        values.insert(*value_id);
                    }
                }
                _ => {}
            }
        }
    }
    values
}

fn required_stack_operands(
    instructions: &[Instr],
    start_pc: usize,
    end_pc: usize,
) -> Option<usize> {
    if start_pc >= instructions.len() || end_pc >= instructions.len() || start_pc > end_pc {
        return None;
    }

    let mut current_depth = 0usize;
    let mut required_depth = 0usize;
    for instr in &instructions[start_pc..=end_pc] {
        let effect = instr.stack_effect()?;
        if current_depth < effect.pops {
            required_depth += effect.pops - current_depth;
            current_depth = effect.pops;
        }
        current_depth = current_depth - effect.pops + effect.pushes;
    }
    Some(required_depth)
}
