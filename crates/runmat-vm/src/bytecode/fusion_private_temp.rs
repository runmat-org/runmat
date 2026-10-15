//! A private MIR temporary bridge, derived from the selected executable
//! stream and its sealed statement facts. No serialized hint mints a bridge.

use crate::accel::graph::AccelStore;
use crate::bytecode::Bytecode;
use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNodeLabel, AccelOpCategory, PrimitiveOp, VarBinding, VarKind,
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
    stores: &[AccelStore],
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
    true
}

#[cfg(test)]
#[path = "fusion_private_temp/dag_tests.rs"]
mod dag_tests;
#[cfg(test)]
#[path = "fusion_private_temp/tests.rs"]
mod tests;
