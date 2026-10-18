use std::collections::{HashMap, HashSet};

use runmat_accelerate::graph::{AccelGraph, ValueId, ValueOrigin, VarKind};
use runmat_accelerate::{FusionGroup, FusionStackLayout, FusionStackValueBinding};

use crate::bytecode::fusion_private_temp::FusionPrivateLiteralControl;
use crate::instr::Instr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackValue {
    Unknown,
    GraphValue(ValueId),
    ProjectedLiteral {
        value_id: ValueId,
        load_pc: usize,
        consumer_pc: usize,
    },
    Variable {
        kind: VarKind,
        index: usize,
    },
}

pub fn annotate_fusion_groups_with_stack_layout(
    instructions: &[Instr],
    graph: &AccelGraph,
    groups: &mut [FusionGroup],
) {
    annotate_fusion_groups_with_projected_literals(instructions, graph, groups, &[]);
}

/// The extra certificates are minted only by exact private binding erasure;
/// they are not a replacement for public graph variable bindings.
pub(crate) fn annotate_fusion_groups_with_projected_literals(
    instructions: &[Instr],
    graph: &AccelGraph,
    groups: &mut [FusionGroup],
    literals: &[FusionPrivateLiteralControl],
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

        if matches!(instr, Instr::StochasticEvolution { .. }) {
            // The selected path and the separately compiled original For
            // reconverge here; neither linear stack predecessor is a proof
            // about values after that control-flow join.
            stack.clear();
            continue;
        }

        if matches!(instr, Instr::Swap) {
            if stack.len() < 2 {
                stack.clear();
            } else {
                let last = stack.len() - 1;
                stack.swap(last, last - 1);
            }
            continue;
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
            match instr {
                Instr::LoadVar(index) => {
                    let certified = literals
                        .binary_search_by_key(&pc, |literal| literal.load_pc)
                        .ok()
                        .filter(|&position| {
                            (position == 0 || literals[position - 1].load_pc != pc)
                                && (position + 1 == literals.len()
                                    || literals[position + 1].load_pc != pc)
                        })
                        .and_then(|position| literals[position].projected_load_value(pc, *index));
                    certified
                        .map(|(value_id, consumer_pc)| StackValue::ProjectedLiteral {
                            value_id,
                            load_pc: pc,
                            consumer_pc,
                        })
                        .unwrap_or(StackValue::Variable {
                            kind: VarKind::Global,
                            index: *index,
                        })
                }
                Instr::LoadLocal(index) => StackValue::Variable {
                    kind: VarKind::Local,
                    index: *index,
                },
                _ => node_output_by_pc
                    .get(&pc)
                    .copied()
                    .map(StackValue::GraphValue)
                    .unwrap_or(StackValue::Unknown),
            }
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
    let variable_ids = external_variable_values(graph, group);
    let slice_start = entry_stack.len().saturating_sub(required_stack_operands);
    let mut seen = HashSet::new();
    let mut bindings = Vec::new();
    let mut compiler_variable_bindings = Vec::new();

    for (absolute_offset, value) in entry_stack.iter().enumerate().skip(slice_start) {
        let value_id = match value {
            StackValue::GraphValue(value_id)
                if stack_value_ids.contains(value_id) && seen.insert(*value_id) =>
            {
                *value_id
            }
            StackValue::ProjectedLiteral {
                value_id,
                load_pc,
                consumer_pc,
            } if *load_pc < group.span.start
                && group.span.end <= *consumer_pc
                && stack_value_ids.contains(value_id)
                && seen.insert(*value_id) =>
            {
                *value_id
            }
            StackValue::Variable { kind, index } => {
                // Retain each certified occurrence, including x+x; an ambiguous
                // same-slot graph source cannot authenticate any occurrence.
                let Some(Some(value_id)) = variable_ids.get(&(*kind, *index)) else {
                    continue;
                };
                *value_id
            }
            _ => continue,
        };
        let binding = FusionStackValueBinding {
            value_id,
            stack_offset: absolute_offset - slice_start,
        };
        if matches!(
            value,
            StackValue::Variable { .. } | StackValue::ProjectedLiteral { .. }
        ) {
            compiler_variable_bindings.push(binding.clone());
        }
        bindings.push(binding);
    }

    Some(FusionStackLayout {
        required_stack_operands,
        bindings,
        compiler_variable_bindings,
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

fn external_variable_values(
    graph: &AccelGraph,
    group: &FusionGroup,
) -> HashMap<(VarKind, usize), Option<ValueId>> {
    let mut values = HashMap::new();
    let node_set: HashSet<_> = group.nodes.iter().copied().collect();
    for node_id in &group.nodes {
        let Some(node) = graph.node(*node_id) else {
            continue;
        };
        for &value_id in &node.inputs {
            let Some(info) = graph.value(value_id) else {
                continue;
            };
            if matches!(info.origin, ValueOrigin::NodeOutput { node, .. } if node_set.contains(&node))
            {
                continue;
            }
            // A previous instruction-produced value can be the current
            // contents of a VM binding (for example x = gpuArray(...)). Its
            // graph origin remains NodeOutput; the exact binding map, not
            // origin syntax, authenticates each LoadVar occurrence at entry.
            let key = if let Some(binding) = graph.var_binding(value_id) {
                (binding.kind, binding.index)
            } else if let ValueOrigin::Variable { kind, index } = info.origin {
                (kind, index)
            } else {
                continue;
            };
            values
                .entry(key)
                .and_modify(|existing| {
                    if *existing != Some(value_id) {
                        *existing = None;
                    }
                })
                .or_insert(Some(value_id));
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate::fusion::FusionKind;
    use runmat_accelerate::graph::{
        AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, PrimitiveOp,
        ShapeInfo, ValueInfo,
    };

    fn source_layout(
        origins: Vec<ValueOrigin>,
        inputs: Vec<ValueId>,
        entry: Vec<StackValue>,
    ) -> FusionStackLayout {
        let values = origins
            .into_iter()
            .enumerate()
            .map(|(id, origin)| ValueInfo {
                id: id as ValueId,
                origin,
                ty: runmat_builtins::Type::Unknown,
                shape: ShapeInfo::Unknown,
                constant: None,
            })
            .collect();
        let graph = AccelGraph {
            nodes: vec![AccelNode {
                id: 0,
                label: AccelNodeLabel::Primitive(PrimitiveOp::Add),
                category: AccelOpCategory::Elementwise,
                inputs,
                outputs: Vec::new(),
                span: InstrSpan { start: 0, end: 0 },
                tags: Vec::<AccelGraphTag>::new(),
            }],
            values,
            var_bindings: HashMap::new(),
            node_bindings: HashMap::new(),
        };
        let group = FusionGroup {
            id: 0,
            kind: FusionKind::Reduction,
            nodes: vec![0],
            shape: ShapeInfo::Unknown,
            span: InstrSpan { start: 0, end: 0 },
            pattern: None,
            stack_layout: None,
        };
        build_group_stack_layout(&[Instr::Add], &graph, &group, &entry).expect("two entry operands")
    }

    #[test]
    fn swap_preserves_exact_operand_provenance() {
        let instructions = [
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::Swap,
            Instr::Add,
        ];
        let graph = crate::accel::graph::build_accel_graph(
            &instructions,
            &[runmat_builtins::Type::Num, runmat_builtins::Type::Num],
        );
        let node = graph.nodes.last().expect("swapped Add node");
        let mut groups = [FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![node.id],
            shape: ShapeInfo::Unknown,
            span: InstrSpan { start: 3, end: 3 },
            pattern: None,
            stack_layout: None,
        }];
        annotate_fusion_groups_with_stack_layout(&instructions, &graph, &mut groups);
        let layout = groups[0].stack_layout.as_ref().expect("exact stack layout");
        assert_eq!(
            layout.bindings,
            vec![
                FusionStackValueBinding {
                    value_id: node.inputs[0],
                    stack_offset: 0
                },
                FusionStackValueBinding {
                    value_id: node.inputs[1],
                    stack_offset: 1
                },
            ]
        );
    }

    #[test]
    fn typed_global_and_local_slots_cannot_alias() {
        let layout = source_layout(
            vec![
                ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                ValueOrigin::Variable {
                    kind: VarKind::Local,
                    index: 0,
                },
            ],
            vec![0, 1],
            vec![
                StackValue::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                StackValue::Variable {
                    kind: VarKind::Local,
                    index: 0,
                },
            ],
        );
        assert_eq!(
            layout.bindings,
            vec![
                FusionStackValueBinding {
                    value_id: 0,
                    stack_offset: 0,
                },
                FusionStackValueBinding {
                    value_id: 1,
                    stack_offset: 1,
                },
            ]
        );
    }

    #[test]
    fn same_slot_conflicting_source_versions_fail_closed() {
        let layout = source_layout(
            vec![
                ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                ValueOrigin::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
            ],
            vec![0, 1],
            vec![
                StackValue::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                StackValue::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
            ],
        );
        assert!(layout.bindings.is_empty());
    }

    #[test]
    fn repeated_operands_keep_each_certified_offset() {
        let layout = source_layout(
            vec![ValueOrigin::Variable {
                kind: VarKind::Global,
                index: 0,
            }],
            vec![0, 0],
            vec![
                StackValue::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
                StackValue::Variable {
                    kind: VarKind::Global,
                    index: 0,
                },
            ],
        );
        assert_eq!(
            layout.bindings,
            vec![
                FusionStackValueBinding {
                    value_id: 0,
                    stack_offset: 0,
                },
                FusionStackValueBinding {
                    value_id: 0,
                    stack_offset: 1,
                },
            ]
        );
    }
}
