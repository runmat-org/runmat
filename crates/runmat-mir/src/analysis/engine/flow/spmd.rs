//! Project worker-local facts into their client-side SPMD result owners.
use runmat_types::{DynamicReason, LiteralValue, ValueFact, ValueKindFact};

use crate::analysis::parallel::SpmdExit;
use crate::BasicBlockId;

use super::FlowState;

pub(super) fn project_exit(
    exits: &[SpmdExit],
    predecessor: BasicBlockId,
    successor: BasicBlockId,
    state: &mut FlowState,
) {
    for exit in exits
        .iter()
        .filter(|exit| exit.crosses(predecessor, successor))
    {
        for local in &exit.outputs {
            let slot = &mut state.locals[local.0];
            let value = slot
                .fact
                .take()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
            // Entry availability is checked by Composite access at runtime;
            // conditional worker writes do not make the client owner undefined.
            let projected = if matches!(value.kind, ValueKindFact::Distributed(_)) {
                value
            } else if let Some(labs) = exit.labs {
                runmat_types::composite_fact(exit.region, labs, value, None)
            } else {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            };
            slot.set(projected, LiteralValue::Unknown);
        }
    }
}
