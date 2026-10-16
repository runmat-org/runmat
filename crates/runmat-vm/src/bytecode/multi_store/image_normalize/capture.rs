//! Read only the exact external leaves at the selected statement boundary.
//! A stack value produced later inside the region is not an entry input.

use super::{SelectedImageRegion, SelectedImageRegionSet};
use crate::bytecode::program::ExecutionContext;
use runmat_accelerate::graph::{ValueOrigin, VarKind};
use runmat_value::Value;

impl SelectedImageRegionSet {
    pub(crate) fn capture_inputs(
        &self,
        region: &SelectedImageRegion,
        vars: &[Value],
        context: &ExecutionContext,
    ) -> Option<Vec<Value>> {
        let mut values = Vec::with_capacity(region.plan.inputs.len());
        for id in &region.plan.inputs {
            let info = self.graph.value(*id)?;
            let value = match &info.origin {
                ValueOrigin::Variable { kind, index } => {
                    // A source leaf changed by this selected region cannot be
                    // captured earlier than its ordinary Load instruction.
                    if region.publication.ordered_stores().iter().any(|store| {
                        store.store.binding.kind == *kind && store.store.binding.index == *index
                    }) {
                        return None;
                    }
                    match kind {
                        VarKind::Global => vars.get(*index)?,
                        VarKind::Local => {
                            if let Some(frame) = context.call_stack.last() {
                                context
                                    .locals
                                    .get(frame.locals_start.checked_add(*index)?)?
                            } else {
                                vars.get(*index)?
                            }
                        }
                    }
                }
                ValueOrigin::Constant | ValueOrigin::NodeOutput { .. } => {
                    region.plan.const_values.get(id)?
                }
                ValueOrigin::Unknown => return None,
            };
            values.push(value.clone());
        }
        Some(values)
    }
}
