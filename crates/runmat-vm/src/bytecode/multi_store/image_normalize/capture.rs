//! Read only the exact external leaves at the selected statement boundary.
//! A stack value produced later inside the region is not an entry input.

use super::{
    ImageEntryInput, ImageNormalizePublicationJoin, SelectedImageRegion, SelectedImageRegionSet,
};
use crate::accel::graph::AccelStore;
use crate::bytecode::program::ExecutionContext;
use runmat_accelerate::fusion::{
    image_reduction::{ImageSourceLeaf, ImageSourceTranscript},
    FusionGroupPlan,
};
use runmat_accelerate::graph::{AccelGraph, ValueOrigin, VarBinding, VarKind};
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_value::Value;

/// The owned VM entry projection and its pre-clone managed reservation.
/// Field order releases cloned values before refunding their capacity.
pub(crate) struct CapturedImageInputs {
    pub(crate) values: Vec<Value>,
    _preparation: Option<HostAllocationReceipt>,
}

/// Seal the source of every entry input against the same ordered bytecode
/// store transcript used to reconcile selected publications. A graph value
/// produced by `rand` or `single` may still be the current value of a named
/// slot; its `NodeOutput` origin is not evidence of a frozen constant.
pub(super) fn sealed_entry_inputs(
    graph: &AccelGraph,
    stores: &[AccelStore],
    plan: &FusionGroupPlan,
    publication: &ImageNormalizePublicationJoin,
    source: &ImageSourceTranscript,
) -> Option<Vec<ImageEntryInput>> {
    let start = publication.pcs().start;
    let mut sources = Vec::new();
    sources.try_reserve_exact(plan.inputs.len()).ok()?;
    for &id in &plan.inputs {
        let info = graph.value(id)?;
        let mut prior_binding: Option<&VarBinding> = None;
        for store in stores
            .iter()
            .filter(|store| store.pc < start && store.value_id == id)
        {
            if prior_binding.is_some_and(|binding| binding != &store.binding) {
                // One graph value was assigned to multiple slots. Its ValueId
                // does not identify which later Load supplied this operand.
                return None;
            }
            prior_binding = Some(&store.binding);
        }
        let entry = if let Some(binding) = prior_binding {
            if let ValueOrigin::Variable { kind, index } = &info.origin {
                if binding.kind != *kind || binding.index != *index {
                    return None;
                }
            }
            if stores
                .iter()
                .rev()
                .find(|store| store.pc < start && &store.binding == binding)?
                .value_id
                != id
            {
                // A later write replaced this value before the region began.
                return None;
            }
            if let ValueOrigin::NodeOutput { node, .. } = &info.origin {
                if graph.node(*node)?.span.end >= start {
                    return None;
                }
            }
            ImageEntryInput::Binding(binding.clone())
        } else {
            match &info.origin {
                ValueOrigin::Variable { kind, index } => {
                    let binding = VarBinding {
                        kind: *kind,
                        index: *index,
                    };
                    if stores
                        .iter()
                        .rev()
                        .any(|store| store.pc < start && store.binding == binding)
                    {
                        // This external load was replaced before the region.
                        return None;
                    }
                    ImageEntryInput::Binding(binding)
                }
                ValueOrigin::Constant | ValueOrigin::NodeOutput { .. }
                    if graph.var_binding(id).is_none() && plan.const_values.contains_key(&id) =>
                {
                    if let ValueOrigin::NodeOutput { node, .. } = &info.origin {
                        if graph.node(*node)?.span.end >= start {
                            return None;
                        }
                    }
                    ImageEntryInput::Constant
                }
                ValueOrigin::NodeOutput { node, .. }
                    if graph.var_binding(id).is_none()
                        && graph.node_binding(*node).is_none()
                        && source.leaves.iter().any(|leaf| {
                            matches!(leaf, ImageSourceLeaf::Constant { id: folded, .. } if *folded == id)
                        })
                        && (graph.exact_folded_literal_array(id).is_some()
                            || graph.exact_folded_unary_scalar(id).is_some()) =>
                {
                    // The sealed source transcript proves this executable
                    // producer precedes its selected use and records its
                    // exact represented bits. It is not a live unbound leaf.
                    ImageEntryInput::FoldedConstant
                }
                _ => return None,
            }
        };
        if let ImageEntryInput::Binding(binding) = &entry {
            if prior_binding.is_some()
                && graph
                    .var_binding(id)
                    .is_some_and(|indexed| indexed != binding)
            {
                // The ordered store is authority; the graph's binding index
                // can be absent for a projected compiler-private slot. If it
                // survives projection, it must agree, but cannot establish
                // order by itself.
                return None;
            }
            if publication
                .ordered_stores()
                .iter()
                .any(|store| &store.store.binding == binding)
            {
                // The entry slot is not stable through the skipped VM region.
                return None;
            }
        }
        sources.push(entry);
    }
    Some(sources)
}

impl SelectedImageRegionSet {
    fn entry_value<'a>(
        &'a self,
        region: &'a SelectedImageRegion,
        id: &runmat_accelerate::graph::ValueId,
        source: &ImageEntryInput,
        vars: &'a [Value],
        context: &'a ExecutionContext,
    ) -> Option<&'a Value> {
        self.graph.value(*id)?;
        match source {
            ImageEntryInput::Constant => region.plan.const_values.get(id),
            ImageEntryInput::FoldedConstant => self.graph.value(*id)?.constant.as_ref(),
            ImageEntryInput::Binding(binding) => match binding.kind {
                VarKind::Global => vars.get(binding.index),
                VarKind::Local => {
                    if let Some(frame) = context.call_stack.last() {
                        context
                            .locals
                            .get(frame.locals_start.checked_add(binding.index)?)
                    } else {
                        vars.get(binding.index)
                    }
                }
            },
        }
    }

    pub(crate) fn capture_inputs(
        &self,
        region: &SelectedImageRegion,
        vars: &[Value],
        context: &ExecutionContext,
        runtime: &RuntimeContext,
    ) -> Result<Option<CapturedImageInputs>, HostAllocationError> {
        if region.entry_inputs.len() != region.plan.inputs.len() {
            return Ok(None);
        }
        let finite = runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        });
        let mut requested = region
            .plan
            .inputs
            .len()
            .checked_mul(std::mem::size_of::<Value>())
            .ok_or(HostAllocationError::Overflow)?;
        // Value::Tensor clones only its COW handle and rank-sized shape.
        // Resident handles and scalar controls clone without heap payloads.
        for (id, source) in region.plan.inputs.iter().zip(&region.entry_inputs) {
            let Some(value) = self.entry_value(region, id, source, vars, context) else {
                return Ok(None);
            };
            let additional = match value {
                Value::Tensor(tensor) => tensor
                    .shape()
                    .len()
                    .checked_mul(std::mem::size_of::<usize>()),
                Value::String(text) => Some(text.len()),
                Value::StringArray(array) => {
                    let mut bytes = array
                        .shape
                        .len()
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|shape| {
                            array
                                .data
                                .len()
                                .checked_mul(std::mem::size_of::<String>())
                                .and_then(|data| shape.checked_add(data))
                        })
                        .ok_or(HostAllocationError::Overflow)?;
                    for text in &array.data {
                        bytes = bytes
                            .checked_add(text.len())
                            .ok_or(HostAllocationError::Overflow)?;
                    }
                    Some(bytes)
                }
                Value::CharArray(array) => array
                    .shape()
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|shape| {
                        array
                            .data
                            .len()
                            .checked_mul(std::mem::size_of::<char>())
                            .and_then(|data| shape.checked_add(data))
                    }),
                Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::GpuTensor(_) => Some(0),
                _ if finite => return Ok(None),
                _ => Some(0),
            }
            .ok_or(HostAllocationError::Overflow)?;
            requested = requested
                .checked_add(additional)
                .ok_or(HostAllocationError::Overflow)?;
        }
        let receipt = if finite {
            let requested = u64::try_from(requested).map_err(|_| HostAllocationError::Overflow)?;
            let plan = HostAllocationPlan::new(requested, 0)?;
            let receipt = HostAllocationReceipt::admit(Some(runtime), plan)?;
            if receipt.is_none() {
                return Err(HostAllocationError::StaleAuthority);
            }
            receipt
        } else {
            None
        };
        let mut values = Vec::new();
        if values.try_reserve_exact(region.plan.inputs.len()).is_err() {
            return Ok(None);
        }
        for (id, source) in region.plan.inputs.iter().zip(&region.entry_inputs) {
            let Some(value) = self.entry_value(region, id, source, vars, context) else {
                return Ok(None);
            };
            values.push(value.clone());
        }
        Ok(Some(CapturedImageInputs {
            values,
            _preparation: receipt,
        }))
    }
}
