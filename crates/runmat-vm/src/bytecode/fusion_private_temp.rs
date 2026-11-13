//! A private MIR temporary bridge, derived from the selected executable
//! stream and its sealed statement facts. No serialized hint mints a bridge.

use crate::accel::graph::{AccelLoad, AccelStore};
use crate::bytecode::Bytecode;
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNodeLabel, AccelOpCategory, PrimitiveOp, ValueId, VarBinding,
    VarKind,
};
use runmat_accelerate::FusionGroup;
#[cfg(test)]
use runmat_accelerate::FusionKind;
use runmat_value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FusionPrivateTempBridge {
    pub slot: usize,
    pub store_pc: usize,
    pub load_pc: usize,
    pub producer_pc: usize,
    pub consumer_pc: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FusionPrivateEmptyControl {
    pub slot: usize,
    pub store_pc: usize,
    pub producer_pc: usize,
    pub consumer_pc: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FusionPrivateLiteralControl {
    pub slot: usize,
    pub store_pc: usize,
    pub load_pc: usize,
    pub producer_pc: usize,
    pub consumer_pc: usize,
    // Minted only when the exact graph binding is actually erased below.
    projected_value: Option<ValueId>,
}

impl FusionPrivateLiteralControl {
    pub(crate) fn projected_load_value(&self, pc: usize, slot: usize) -> Option<(ValueId, usize)> {
        (self.load_pc == pc && self.slot == slot)
            .then_some(self.projected_value.map(|id| (id, self.consumer_pc)))
            .flatten()
    }
}

#[path = "fusion_private_temp/coalesce.rs"]
mod coalesce;
#[path = "fusion_private_temp/projection_index.rs"]
mod projection_index;
#[path = "fusion_private_temp/proof.rs"]
mod proof;

pub(crate) fn bridges(bytecode: &Bytecode) -> Vec<FusionPrivateTempBridge> {
    proof::bridges(bytecode)
}

pub(crate) fn empty_controls(bytecode: &Bytecode) -> Vec<FusionPrivateEmptyControl> {
    proof::empty_controls(bytecode)
}

pub(crate) fn literal_controls(bytecode: &Bytecode) -> Vec<FusionPrivateLiteralControl> {
    proof::literal_controls(bytecode)
}

/// Only sealed, group-contained stores may be omitted by a final-only
/// reduction. The ordinary VM store gate remains authoritative for every
/// other StoreVar/StoreLocal in the selected span.
pub(crate) fn private_reduction_store_pcs(
    group: &FusionGroup,
    bridges: &[FusionPrivateTempBridge],
    controls: &[FusionPrivateEmptyControl],
) -> Vec<usize> {
    if !group.kind.is_reduction() {
        return Vec::new();
    }
    bridges
        .iter()
        .filter(|bridge| group.span.start <= bridge.store_pc && bridge.load_pc < group.span.end)
        .map(|bridge| bridge.store_pc)
        .chain(
            controls
                .iter()
                .filter(|control| {
                    group.span.start <= control.producer_pc && control.consumer_pc <= group.span.end
                })
                .map(|control| control.store_pc),
        )
        .collect()
}
pub(crate) fn coalesce_reduction_groups(
    groups: &mut Vec<FusionGroup>,
    bridges: &[FusionPrivateTempBridge],
) {
    coalesce::reduction_groups(groups, bridges);
}

pub(crate) fn coalesce_chain_groups(
    groups: &mut Vec<FusionGroup>,
    bridges: &[FusionPrivateTempBridge],
) {
    coalesce::chain_groups(groups, bridges);
}

pub(crate) fn private_chain_store_pcs(
    group: &FusionGroup,
    bridges: &[FusionPrivateTempBridge],
) -> Vec<usize> {
    if !group.kind.is_elementwise() {
        return Vec::new();
    }
    bridges
        .iter()
        .filter(|bridge| {
            group.span.start <= bridge.producer_pc && bridge.consumer_pc <= group.span.end
        })
        .map(|bridge| bridge.store_pc)
        .collect()
}

pub(crate) fn project_private_bindings(
    graph: &mut AccelGraph,
    groups: &[FusionGroup],
    bridges: &[FusionPrivateTempBridge],
    controls: &[FusionPrivateEmptyControl],
    literals: &mut [FusionPrivateLiteralControl],
    stores: &[AccelStore],
    loads: &[AccelLoad],
) -> bool {
    let index = projection_index::ProjectionIndex::new(graph, groups, stores);
    for bridge in bridges {
        if !index.region_covers(bridge.producer_pc, bridge.consumer_pc) {
            continue;
        }
        let (Some(producer_position), Some(consumer_position), Some(store_position)) = (
            index.producer_at(bridge.producer_pc),
            index.consumer_at(bridge.consumer_pc),
            index.store_at(bridge.store_pc),
        ) else {
            return false;
        };
        let (producer, consumer, store) = (
            &graph.nodes[producer_position],
            &graph.nodes[consumer_position],
            &stores[store_position],
        );
        let [output] = producer.outputs.as_slice() else {
            return false;
        };
        let expected = VarBinding {
            kind: VarKind::Global,
            index: bridge.slot,
        };
        if !consumer.inputs.contains(output)
            || store.value_id != *output
            || store.binding != expected
            || graph.var_binding(*output) != Some(&expected)
            || graph.node_binding(producer.id) != Some(&expected)
            || index.has_other_consumer(*output, consumer.id)
        {
            return false;
        }
        let producer_id = producer.id;
        graph.var_bindings.remove(output);
        graph.node_bindings.remove(&producer_id);
    }
    for control in controls {
        if !index.reduction_covers(control.producer_pc, control.consumer_pc) {
            continue;
        }
        let (Some(producer_position), Some(consumer_position), Some(store_position)) = (
            index.producer_at(control.producer_pc),
            index.consumer_at(control.consumer_pc),
            index.store_at(control.store_pc),
        ) else {
            return false;
        };
        let (producer, consumer, store) = (
            &graph.nodes[producer_position],
            &graph.nodes[consumer_position],
            &stores[store_position],
        );
        let [output] = producer.outputs.as_slice() else {
            return false;
        };
        let expected = VarBinding {
            kind: VarKind::Global,
            index: control.slot,
        };
        let exact_empty = graph.value(*output).is_some_and(|value| {
            matches!(&value.constant, Some(Value::Tensor(tensor))
                if tensor.shape() == [0, 0] && tensor.is_empty())
        });
        if producer.category != AccelOpCategory::Other
            || !matches!(
                &producer.label,
                AccelNodeLabel::Primitive(PrimitiveOp::UPlus)
            )
            || producer.tags.as_slice() != [AccelGraphTag::FoldedLiteralArray]
            || !producer.inputs.is_empty()
            || !exact_empty
            || !consumer.is_reduction()
            || !consumer.inputs.contains(output)
            || store.value_id != *output
            || store.binding != expected
            || graph.var_binding(*output) != Some(&expected)
            || graph.node_binding(producer.id) != Some(&expected)
            || index.has_other_consumer(*output, consumer.id)
        {
            return false;
        }
        graph.var_bindings.remove(output);
        graph.node_bindings.remove(&producer.id);
    }
    for literal in literals {
        // The source-use certificate is intentionally independent of the
        // producer's graph operation. Existing private-edge/empty-control
        // owners may already have projected this exact store.
        if bridges.iter().any(|bridge| {
            bridge.slot == literal.slot
                && bridge.store_pc == literal.store_pc
                && bridge.load_pc == literal.load_pc
                && bridge.producer_pc == literal.producer_pc
                // The bridge ends at the unique graph operation while the
                // literal certificate ends at its containing statement.
                // The same store and load establish the same private value;
                // only that contained, source-ordered use may share proof.
                && literal.load_pc < bridge.consumer_pc
                && bridge.consumer_pc <= literal.consumer_pc
                && index.region_covers(bridge.producer_pc, bridge.consumer_pc)
        }) || controls.iter().any(|control| {
            control.slot == literal.slot
                && control.store_pc == literal.store_pc
                && control.producer_pc == literal.producer_pc
                // The empty-control certificate ends at the unique reduction
                // call. The literal-use certificate ends at its containing
                // statement. Their exact store, producer, and private slot
                // identify the same selected value; the reduction call must
                // still be inside that sealed use, after its unique load.
                && literal.load_pc < control.consumer_pc
                && control.consumer_pc <= literal.consumer_pc
                && index.reduction_covers(control.producer_pc, control.consumer_pc)
        }) {
            continue;
        }
        let (Some(producer_position), Some(store_position)) = (
            index.producer_at(literal.producer_pc),
            index.store_at(literal.store_pc),
        ) else {
            continue;
        };
        let (producer, store) = (&graph.nodes[producer_position], &stores[store_position]);
        let [output] = producer.outputs.as_slice() else {
            return false;
        };
        let expected = VarBinding {
            kind: VarKind::Global,
            index: literal.slot,
        };
        let exact_immutable = graph.exact_folded_literal_array(*output).is_some()
            || graph.exact_folded_unary_scalar(*output).is_some();
        if !exact_immutable {
            // A sealed single-use temporary need not be a literal control.
            // Leave its binding intact; it has not acquired fold authority.
            continue;
        }
        let Some(consumer_position) = graph.nodes.iter().position(|node| {
            node.span.start >= literal.load_pc
                && node.span.end <= literal.consumer_pc
                && node.inputs.contains(output)
        }) else {
            return false;
        };
        let consumer = &graph.nodes[consumer_position];
        let matching_load = loads
            .binary_search_by_key(&literal.load_pc, |load| load.pc)
            .ok()
            .filter(|&position| {
                (position == 0 || loads[position - 1].pc != literal.load_pc)
                    && (position + 1 == loads.len() || loads[position + 1].pc != literal.load_pc)
            })
            .map(|position| &loads[position]);
        if !consumer.inputs.contains(output)
            || store.value_id != *output
            || store.binding != expected
            || graph.var_binding(*output) != Some(&expected)
            || graph.node_binding(producer.id) != Some(&expected)
            || index.has_other_consumer(*output, consumer.id)
            || !matching_load
                .is_some_and(|load| load.value_id == *output && load.binding == expected)
        {
            return false;
        }
        graph.var_bindings.remove(output);
        graph.node_bindings.remove(&producer.id);
        literal.projected_value = Some(*output);
    }
    true
}

#[cfg(test)]
#[path = "fusion_private_temp/dag_tests.rs"]
mod dag_tests;
#[cfg(test)]
#[path = "fusion_private_temp/tests.rs"]
mod tests;
