//! VM-only fusion publication admission. The accelerate plan is a reusable
//! compute plan; the bytecode stream owns the order and location of stores.

use std::collections::HashSet;
use std::rc::Rc;

use runmat_accelerate::fusion::{FusionOp, FusionPlan, FusionStoreMaterialization};
use runmat_accelerate::graph::{AccelGraph, VarBinding, VarKind};
use runmat_accelerate::{prepare_fusion_plan, FusionPlanRef};

use crate::accel::graph::{build_accel_graph_with_transcript, AccelLoad, AccelStore};
use crate::accel::stack_layout::annotate_fusion_groups_with_projected_literals;
use crate::bytecode::Instr;

use runmat_runtime::context::{
    RuntimeContext, VmFusionCompletionRight, VmFusionDeclineRight, VmFusionInstalledRegion,
};

#[path = "vm_fusion_plan/semantic_identity.rs"]
mod semantic_identity;
#[path = "vm_fusion_plan/host_input_uses.rs"]
mod host_input_uses;
use crate::accel::fusion::input_transport::VmFusionInputTransport;
pub(crate) use semantic_identity::{vm_fusion_executable_identity, vm_fusion_semantic_identity};

/// VM-private owner retained beside the exact store-reconciled plan. Runtime
/// can certify session/generation but only this module can check actual
/// bytecode preparation and `Rc<FusionPlan>` identity.
pub(crate) struct VmFusionPlanInstallation {
    plan: FusionPlanRef,
    regions: Vec<(usize, VmFusionInstalledRegion, Option<usize>)>,
    input_uses: Vec<(usize, runmat_accelerate::fusion_exec::VmHostInputUses)>,
    input_transports: Vec<(usize, VmFusionInputTransport)>,
}

impl std::fmt::Debug for VmFusionPlanInstallation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VmFusionPlanInstallation")
            .field("groups", &self.regions.len())
            .finish_non_exhaustive()
    }
}

impl VmFusionPlanInstallation {
    pub(crate) fn input_transport<'a>(
        &'a self,
        active_plan: &FusionPlanRef,
        selected: &runmat_accelerate::FusionGroupPlan,
    ) -> Option<&'a VmFusionInputTransport> {
        if !Rc::ptr_eq(&self.plan, active_plan) || !selected.kernel.supported {
            return None;
        }
        let original = self.plan.groups.iter().find(|group| {
            group.index == selected.index
                && group.group.id == selected.group.id
                && group.group.span == selected.group.span
                && group.output == selected.output
                && group.inputs == selected.inputs
        })?;
        self.input_transports
            .iter()
            .find_map(|(index, transport)| (*index == original.index).then_some(transport))
    }

    /// Exact graph-value IDs and load PCs originate in the same instruction
    /// walk. Neither serialized hints nor ordinal merged stack patterns grant
    /// host substitution authority.
    pub(crate) fn host_input_uses<'a>(
        &'a self,
        active_plan: &FusionPlanRef,
        selected: &runmat_accelerate::FusionGroupPlan,
    ) -> Option<&'a runmat_accelerate::fusion_exec::VmHostInputUses> {
        if !Rc::ptr_eq(&self.plan, active_plan) || !selected.kernel.supported {
            return None;
        }
        let original = self.plan.groups.iter().find(|group| {
            std::ptr::eq(*group, selected)
                || (group.index == selected.index
                    && group.group.id == selected.group.id
                    && group.group.span == selected.group.span
                    && group.output == selected.output)
        })?;
        self.input_uses
            .iter()
            .find_map(|(index, uses)| (*index == original.index).then_some(uses))
    }

    pub(crate) fn right_for(
        &self,
        active_plan: &FusionPlanRef,
        selected: &runmat_accelerate::FusionGroupPlan,
        runtime: &RuntimeContext,
    ) -> Option<VmFusionCompletionRight> {
        if !Rc::ptr_eq(&self.plan, active_plan) || !selected.kernel.supported {
            return None;
        }
        let original = self.plan.groups.iter().find(|group| {
            group.index == selected.index
                && group.group.id == selected.group.id
                && group.group.span == selected.group.span
                && group.output == selected.output
        })?;
        let installed = self
            .regions
            .iter()
            .find_map(|(index, installed, _)| (*index == original.index).then_some(installed))?;
        installed
            .matches_group(
                selected.group.id,
                selected.group.span.start,
                selected.group.span.end,
            )
            .then(|| installed.right(runtime))
            .flatten()
    }

    /// Only a selected installed region can decline before effects. This is
    /// deliberately separate from the FullCall completion/host-route right.
    pub(crate) fn decline_for(
        &self,
        active_plan: &FusionPlanRef,
        selected: &runmat_accelerate::FusionGroupPlan,
        runtime: &RuntimeContext,
    ) -> Option<VmFusionDeclineRight> {
        if !Rc::ptr_eq(&self.plan, active_plan) || !selected.kernel.supported {
            return None;
        }
        let original = self.plan.groups.iter().find(|group| {
            group.index == selected.index
                && group.group.id == selected.group.id
                && group.group.span == selected.group.span
                && group.output == selected.output
        })?;
        let installed = self
            .regions
            .iter()
            .find_map(|(index, installed, _)| (*index == original.index).then_some(installed))?;
        installed
            .matches_group(
                selected.group.id,
                selected.group.span.start,
                selected.group.span.end,
            )
            .then(|| installed.decline_right(runtime))
            .flatten()
    }

    pub(crate) fn publication_pc_for(
        &self,
        active_plan: &FusionPlanRef,
        selected: &runmat_accelerate::FusionGroupPlan,
    ) -> Option<usize> {
        if !Rc::ptr_eq(&self.plan, active_plan) || !selected.kernel.supported {
            return None;
        }
        let original = self.plan.groups.iter().find(|group| {
            group.index == selected.index
                && group.group.id == selected.group.id
                && group.group.span == selected.group.span
                && group.output == selected.output
        })?;
        self.regions.iter().find_map(|(index, _, publication)| {
            (*index == original.index).then_some(*publication).flatten()
        })
    }
}

use super::Bytecode;

impl Bytecode {
    /// Read-only diagnostic projection of the same graph and sanitized groups
    /// used by VM preparation. This does not grant execution, binding, session,
    /// capability, or publication authority.
    pub fn reconciled_runtime_fusion_graph(
        &self,
    ) -> Option<(AccelGraph, Vec<runmat_accelerate::FusionGroup>)> {
        let (plan, graph) = self.prepare_vm_fusion_plan();
        Some((
            graph?,
            plan?
                .groups
                .iter()
                .map(|selected| selected.group.clone())
                .collect(),
        ))
    }

    /// Prepare one VM-owned projection of the cached compute plan. No store
    /// transcript is consulted again during instruction dispatch.
    pub(crate) fn prepare_vm_fusion_plan(&self) -> (Option<FusionPlanRef>, Option<AccelGraph>) {
        let (plan, graph, _, _) = self.prepare_vm_fusion_plan_with_stores();
        (plan, graph)
    }

    fn prepare_vm_fusion_plan_with_stores(
        &self,
    ) -> (
        Option<FusionPlanRef>,
        Option<AccelGraph>,
        Vec<AccelStore>,
        Vec<AccelLoad>,
    ) {
        let groups = self.runtime_fusion_groups();
        if groups.is_empty() || self.fusion_metadata.mir_fusion_candidate_group_count == 0 {
            return (None, None, Vec::new(), Vec::new());
        }

        let (mut graph, stores, loads) =
            build_accel_graph_with_transcript(&self.instructions, &self.var_types);
        // Scalar-vs-matrix normalization belongs to this exact graph. Close
        // certified private edges only after that distinction is available.
        let groups = self.runtime_fusion_groups_for_graph(&graph);
        let mut literal_controls = crate::bytecode::fusion_private_temp::literal_controls(self);
        if !crate::bytecode::fusion_private_temp::project_private_bindings(
            &mut graph,
            &groups,
            &crate::bytecode::fusion_private_temp::bridges(self),
            &crate::bytecode::fusion_private_temp::empty_controls(self),
            &mut literal_controls,
            &stores,
            &loads,
        ) {
            return (None, Some(graph), stores, loads);
        }
        let groups = self.runtime_fusion_groups_for_graph(&graph);
        let cached = prepare_fusion_plan(
            Some(&graph),
            &groups,
            self.fusion_metadata.mir_fusion_candidate_group_count,
        );
        let plan = cached.map(|cached| {
            let mut owned = (*cached).clone();
            // Semantic windows are annotated before canonical graph
            // reconciliation, while their node sets are still empty. Rebuild
            // the compiler-owned stack certificate from the exact sanitized
            // nodes selected for this VM plan; never infer variable authority
            // from an ordinal stack pattern.
            let mut reconciled_groups = owned
                .groups
                .iter()
                .map(|selected| selected.group.clone())
                .collect::<Vec<_>>();
            literal_controls.sort_by_key(|control| control.load_pc);
            annotate_fusion_groups_with_projected_literals(
                &self.instructions,
                &graph,
                &mut reconciled_groups,
                &literal_controls,
            );
            for (selected, reconciled) in owned.groups.iter_mut().zip(reconciled_groups) {
                selected.group.stack_layout = reconciled.stack_layout;
            }
            reconcile_stores(&mut owned, &self.instructions, &stores);
            let bridges = crate::bytecode::fusion_private_temp::bridges(self);
            for selected in &mut owned.groups {
                let private_pcs = crate::bytecode::fusion_private_temp::private_chain_store_pcs(
                    &selected.group,
                    &bridges,
                );
                // Reconciliation has already authenticated every skipped
                // Store. Only sealed, graph-checked private stores disappear.
                let private_bindings = stores
                    .iter()
                    .filter(|store| private_pcs.contains(&store.pc))
                    .map(|store| (store.value_id, store.binding.clone()))
                    .collect::<Vec<_>>();
                selected.materialized_stores.retain(|store| {
                    !private_bindings.iter().any(|(value, binding)| {
                        *value == store.value_id && binding == &store.binding
                    })
                });
                if VmFusionInputTransport::project(selected, &graph, &loads, &stores).is_none() {
                    selected.kernel.supported = false;
                }
            }
            Rc::new(owned)
        });
        (plan, Some(graph), stores, loads)
    }

    /// Install only groups from the actual reconciled bytecode projection.
    /// This VM-owned path never accepts a serialized recipe or caller span.
    pub(crate) fn prepare_vm_fusion_plan_with_installation(
        &self,
        runtime: &RuntimeContext,
    ) -> (
        Option<FusionPlanRef>,
        Option<AccelGraph>,
        Option<VmFusionPlanInstallation>,
    ) {
        let (plan, graph, stores, loads) = self.prepare_vm_fusion_plan_with_stores();
        let installation = plan.as_ref().map(|plan| {
            let executable = vm_fusion_executable_identity(self);
            let input_uses = plan
                .groups
                .iter()
                .filter_map(|group| {
                    host_input_uses::project(group, &loads).map(|uses| (group.index, uses))
                })
                .collect();
            let input_transports = plan
                .groups
                .iter()
                .filter(|group| group.kernel.supported)
                .filter_map(|group| {
                    VmFusionInputTransport::project(group, graph.as_ref()?, &loads, &stores)
                        .map(|transport| (group.index, transport))
                })
                .collect();
            let regions = plan
                .groups
                .iter()
                .filter(|group| group.kernel.supported)
                .filter_map(|group| {
                    let semantic = vm_fusion_semantic_identity(executable?, group)?;
                    VmFusionInstalledRegion::install(
                        runtime,
                        semantic,
                        group.group.id,
                        group.group.span.start,
                        group.group.span.end,
                    )
                    .map(|installed| {
                        let publication = group
                            .group
                            .span
                            .end
                            .checked_add(1)
                            .and_then(|pc| {
                                let output = group.output?;
                                stores.iter().find(|store| {
                                    store.pc == pc
                                        && store.value_id == output
                                        && match self.instructions.get(pc) {
                                            Some(Instr::StoreVar(slot)) => {
                                                store.binding.kind == VarKind::Global
                                                    && store.binding.index == *slot
                                            }
                                            Some(Instr::StoreLocal(slot)) => {
                                                store.binding.kind == VarKind::Local
                                                    && store.binding.index == *slot
                                            }
                                            _ => false,
                                        }
                                })
                            })
                            .map(|store| store.pc);
                        (group.index, installed, publication)
                    })
                })
                .collect();
            VmFusionPlanInstallation {
                plan: Rc::clone(plan),
                regions,
                input_uses,
                input_transports,
            }
        });
        (plan, graph, installation)
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
