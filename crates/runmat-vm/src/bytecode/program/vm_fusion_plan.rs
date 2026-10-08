//! VM-only fusion publication admission. The accelerate plan is a reusable
//! compute plan; the bytecode stream owns the order and location of stores.

use std::collections::HashSet;
use std::rc::Rc;

use runmat_accelerate::fusion::{FusionOp, FusionPlan, FusionStoreMaterialization};
use runmat_accelerate::graph::{AccelGraph, VarBinding, VarKind};
use runmat_accelerate::{prepare_fusion_plan, FusionPlanRef};

use crate::accel::graph::{build_accel_graph_with_stores, AccelStore};
use crate::bytecode::Instr;

use super::Bytecode;

impl Bytecode {
    /// Prepare one VM-owned projection of the cached compute plan. No store
    /// transcript is consulted again during instruction dispatch.
    pub(crate) fn prepare_vm_fusion_plan(&self) -> (Option<FusionPlanRef>, Option<AccelGraph>) {
        let groups = self.runtime_fusion_groups();
        if groups.is_empty() || self.fusion_metadata.mir_fusion_candidate_group_count == 0 {
            return (None, None);
        }

        let (graph, stores) = build_accel_graph_with_stores(&self.instructions, &self.var_types);
        let groups = self.runtime_fusion_groups_for_graph(&graph);
        let cached = prepare_fusion_plan(
            Some(&graph),
            &groups,
            self.fusion_metadata.mir_fusion_candidate_group_count,
        );
        let plan = cached.map(|cached| {
            let mut owned = (*cached).clone();
            reconcile_stores(&mut owned, &self.instructions, &stores);
            Rc::new(owned)
        });
        (plan, Some(graph))
    }
}

fn reconcile_stores(plan: &mut FusionPlan, instructions: &[Instr], stores: &[AccelStore]) {
    for group in &mut plan.groups {
        if !group.group.kind.is_elementwise() {
            continue;
        }
        // The graph's ValueId -> binding map retains only the last assignment.
        // Never publish a store merely because its value is an operation output:
        // its Store instruction must actually be skipped by this group.
        group.materialized_stores.clear();
        let span = &group.group.span;
        if span.start > span.end {
            group.kernel.supported = false;
            continue;
        }
        let Some(slice) = instructions.get(span.start..=span.end) else {
            group.kernel.supported = false;
            continue;
        };
        let produced: HashSet<_> = group
            .operations
            .iter()
            .filter_map(|op| match op {
                FusionOp::Primitive { output, .. } | FusionOp::Builtin { output, .. } => *output,
            })
            .collect();
        let mut transcript = stores_in_span(stores, span.start, span.end)
            .iter()
            .peekable();
        let mut admitted = Vec::new();
        let mut valid = true;
        for (offset, instruction) in slice.iter().enumerate() {
            let pc = span.start + offset;
            let binding = match instruction {
                Instr::StoreVar(index) => Some(VarBinding {
                    kind: VarKind::Global,
                    index: *index,
                }),
                Instr::StoreLocal(index) => Some(VarBinding {
                    kind: VarKind::Local,
                    index: *index,
                }),
                _ => None,
            };
            let Some(binding) = binding else {
                if transcript.peek().is_some_and(|store| store.pc == pc) {
                    valid = false;
                    break;
                }
                continue;
            };
            let Some(store) = transcript.next_if(|store| store.pc == pc) else {
                valid = false;
                break;
            };
            if store.binding != binding || !produced.contains(&store.value_id) {
                valid = false;
                break;
            }
            admitted.push(FusionStoreMaterialization {
                value_id: store.value_id,
                binding,
            });
        }
        if !valid || transcript.next().is_some() {
            group.kernel.supported = false;
            continue;
        }
        group.materialized_stores = admitted;
    }
}

/// GraphBuilder emits stores in PC order. Bound each window with two binary
/// searches rather than rescan the whole transcript for every fusion group.
fn stores_in_span(stores: &[AccelStore], start: usize, end: usize) -> &[AccelStore] {
    let first = stores.partition_point(|store| store.pc < start);
    let after_last = stores.partition_point(|store| store.pc <= end);
    &stores[first..after_last]
}

#[cfg(test)]
#[path = "vm_fusion_plan/tests.rs"]
mod tests;
