pub(super) mod observation;

use super::state::HostState;
use crate::region::RegionPlacement;
use crate::NativeExecutorResult;
use runmat_runtime::native::{
    NativeCancellation, NativeCancellationReason, NativeExit, NativeSiteOutcome,
};

pub(super) fn checkpoint(
    state: &mut HostState,
    request: runmat_runtime::native::NativeSiteRequest,
    exit: &mut NativeExit,
) -> NativeExecutorResult<Option<NativeSiteOutcome>> {
    // An enclosing generic region owns this execution interval. Do not time
    // overlapping nested alternatives or replace its pending observation.
    if state.region_observation.is_some() {
        return Ok(None);
    }
    let Some(plan) = state.optimized_region(request) else {
        return Ok(None);
    };
    let inputs = state.optimized_region_inputs(&plan)?;
    let Some(workload) = runmat_runtime::numeric_region::workload(&inputs) else {
        return Ok(None);
    };
    let placement = match crate::region::choose_region(&state.runtime, &plan, &inputs, workload) {
        RegionPlacement::Unavailable => return Ok(None),
        RegionPlacement::Generic(observation) => {
            state.region_observation = Some(observation::GenericRegionObservation::new(
                &plan,
                observation,
            ));
            return Ok(None);
        }
        RegionPlacement::Vectorized(observation) => observation,
    };
    // The observation guard records failure on execution/publication errors,
    // cancellation or unwinding, and success only after publication completes.
    match runmat_runtime::numeric_region::execute(
        &plan.program,
        &inputs,
        &state.runtime.cancellation(),
    )
    .map_err(|error| crate::NativeExecutorError::Host(error.into()))?
    {
        runmat_runtime::numeric_region::NumericRegionExecution::Completed(outputs) => {
            state.publish_optimized_region(&plan, outputs)?;
            placement.complete();
            Ok(Some(NativeSiteOutcome::continue_execution()))
        }
        runmat_runtime::numeric_region::NumericRegionExecution::Ineligible => Ok(None),
        runmat_runtime::numeric_region::NumericRegionExecution::Cancelled => {
            *exit = NativeExit::cancelled(NativeCancellation {
                reason: NativeCancellationReason::REQUESTED,
                reserved: 0,
                generation: 1,
            });
            Ok(Some(NativeSiteOutcome::exit()))
        }
    }
}
