//! Exact VM input sources, selected before a fusion region is executable.
//!
//! An absent source is an admission decline, not a language exception. In
//! particular, a materialized member/call/index result is never reconstructed
//! by looking up an unrelated variable or by guessing an operand ordinal.

use runmat_accelerate::graph::{AccelGraph, ValueId, ValueOrigin, VarBinding, VarKind};
use runmat_accelerate::{FusionGroupPlan, FusionStackLayout};
use runmat_value::Value;

use crate::accel::graph::{AccelLoad, AccelStore};

#[derive(Debug)]
enum InputSource {
    Stack(usize),
    Literal(Value),
    Frame(VarBinding),
}

#[derive(Debug)]
pub(crate) struct VmFusionInputTransport {
    inputs: Vec<ValueId>,
    sources: Vec<InputSource>,
    layout: FusionStackLayout,
    skipped_reads: Vec<VarBinding>,
}

impl VmFusionInputTransport {
    pub(crate) fn project(
        plan: &FusionGroupPlan,
        graph: &AccelGraph,
        loads: &[AccelLoad],
        stores: &[AccelStore],
    ) -> Option<Self> {
        let layout = plan.group.stack_layout.as_ref()?;
        for binding in &layout.bindings {
            if binding.stack_offset >= layout.required_stack_operands
                || layout.bindings.iter().any(|other| {
                    other.stack_offset == binding.stack_offset && other.value_id != binding.value_id
                })
            {
                return None;
            }
        }
        let span = &plan.group.span;
        let first_load = loads.partition_point(|load| load.pc < span.start);
        let after_load = loads.partition_point(|load| load.pc <= span.end);
        let loads = &loads[first_load..after_load];
        let first_store = stores.partition_point(|store| store.pc < span.start);
        let after_store = stores.partition_point(|store| store.pc <= span.end);
        let stores = &stores[first_store..after_store];
        let mut sources = Vec::with_capacity(plan.inputs.len());
        let mut skipped_reads = Vec::new();
        // Cover every skipped semantic load, including a leaf of a folded
        // array rather than only the plan's top-level external inputs.
        for read in loads {
            let internal = stores
                .iter()
                .rev()
                .find(|store| store.pc < read.pc && store.binding == read.binding)
                .is_some_and(|store| {
                    if store.value_id != read.value_id || graph.var_binding(read.value_id).is_some()
                    {
                        return false;
                    }
                    let Some(info) = graph.value(read.value_id) else {
                        return false;
                    };
                    let ValueOrigin::NodeOutput { node, .. } = info.origin else {
                        return false;
                    };
                    let Some(producer) = graph.node(node) else {
                        return false;
                    };
                    graph.node_binding(node).is_none()
                        && span.start <= producer.span.start
                        && producer.span.end < store.pc
                        && (plan.group.nodes.contains(&node)
                            || graph.source_literal_array(read.value_id).is_some()
                            || graph.exact_folded_unary_scalar(read.value_id).is_some())
                });
            if !internal && !skipped_reads.contains(&read.binding) {
                skipped_reads.push(read.binding.clone());
            }
        }
        for value_id in &plan.inputs {
            graph.value(*value_id)?;
            let reads = loads
                .iter()
                .filter(|load| load.value_id == *value_id)
                .collect::<Vec<_>>();
            // A frame input must remain the same binding for the entire
            // region. Internal private stores are handled as graph producers,
            // never as external frame reads.
            for read in &reads {
                if stores.iter().any(|store| store.binding == read.binding) {
                    return None;
                }
            }
            let source = if let Some(binding) = layout
                .bindings
                .iter()
                .find(|binding| binding.value_id == *value_id)
            {
                if binding.stack_offset >= layout.required_stack_operands {
                    return None;
                }
                InputSource::Stack(binding.stack_offset)
            } else if let Some(value) = graph
                .source_literal_constant(*value_id)
                .or_else(|| graph.source_literal_array(*value_id))
            {
                InputSource::Literal(value.clone())
            } else {
                let first = reads.first()?;
                if reads.iter().any(|read| read.binding != first.binding) {
                    return None;
                }
                InputSource::Frame(first.binding.clone())
            };
            sources.push(source);
        }
        Some(Self {
            inputs: plan.inputs.clone(),
            sources,
            layout: layout.clone(),
            skipped_reads,
        })
    }

    /// A skipped semantic load may use an ordinary, assigned frame slot only.
    /// Shared/global/persistent reads and absent arguments stay with ordinary
    /// dispatch, which owns refresh, authority failures and exception order.
    pub(crate) fn permits_frame_reads(
        &self,
        context: &crate::bytecode::program::ExecutionContext,
        missing: &std::collections::HashSet<usize>,
        globals: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
        persistent: &crate::interpreter::PersistentAliases,
    ) -> bool {
        self.skipped_reads.iter().all(|binding| {
            // Legacy local-stack frames have no assigned-slot authority.
            (binding.kind != VarKind::Local || context.call_stack.is_empty())
                && !missing.contains(&binding.index)
                && !persistent.contains_key(&binding.index)
                && !globals.has_binding_for_slot(binding.index)
                && !context
                    .lexical_binding_slots
                    .iter()
                    .any(|slot| slot.slot == binding.index)
                && crate::runtime::workspace::workspace_slot_assigned(binding.index) == Some(true)
        })
    }

    pub(crate) fn matches(&self, plan: &FusionGroupPlan) -> bool {
        self.inputs == plan.inputs && plan.group.stack_layout.as_ref() == Some(&self.layout)
    }

    pub(crate) fn required_stack_operands(&self) -> usize {
        self.layout.required_stack_operands
    }

    pub(crate) fn gather(
        &self,
        stack: &[Value],
        vars: &crate::interpreter::vars::InterpreterVars,
        context: &crate::bytecode::program::ExecutionContext,
    ) -> Option<Vec<Value>> {
        self.sources
            .iter()
            .map(|source| match source {
                InputSource::Stack(offset) => stack.get(*offset).cloned(),
                InputSource::Literal(value) => Some(value.clone()),
                InputSource::Frame(binding) => match binding.kind {
                    VarKind::Global => {
                        crate::runtime::workspace::workspace_slot_value(binding.index)
                            .or_else(|| vars.get(binding.index).map(|value| value.clone()))
                    }
                    VarKind::Local => match context.call_stack.last() {
                        Some(frame) => context
                            .locals
                            .get(frame.locals_start + binding.index)
                            .cloned(),
                        None => crate::runtime::workspace::workspace_slot_value(binding.index)
                            .or_else(|| vars.get(binding.index).map(|value| value.clone())),
                    },
                },
            })
            .collect()
    }
}
