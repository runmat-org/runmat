use runmat_runtime::native::{
    NativeCancellation, NativeCancellationReason, NativeExit, NativeSiteOutcome,
};

use crate::JitResult;

use super::state::HostState;

pub(super) fn checkpoint(
    state: &mut HostState,
    request: runmat_runtime::native::NativeSiteRequest,
    exit: &mut NativeExit,
) -> JitResult<Option<NativeSiteOutcome>> {
    let Some(plan) = state.optimized_region(request) else {
        return Ok(None);
    };
    let inputs = state.optimized_region_inputs(&plan)?;
    let Some(workload) = runmat_runtime::numeric_region::workload(&inputs) else {
        state.disable_optimized_region(plan.region);
        return Ok(None);
    };
    let Some(placement) =
        crate::region::choose_vectorized(&state.runtime, &plan, &inputs, workload)
    else {
        state.disable_optimized_region(plan.region);
        return Ok(None);
    };
    let started = std::time::Instant::now();
    match runmat_runtime::numeric_region::execute(
        &plan.program,
        &inputs,
        &state.runtime.cancellation(),
    )
    .map_err(|error| crate::JitError::Host(error.into()))?
    {
        runmat_runtime::numeric_region::NumericRegionExecution::Completed(outputs) => {
            state.publish_optimized_region(&plan, outputs)?;
            placement.observe(runmat_time::duration_ns_saturating(started.elapsed()), true);
            Ok(Some(NativeSiteOutcome::continue_execution()))
        }
        runmat_runtime::numeric_region::NumericRegionExecution::Ineligible => {
            placement.observe(
                runmat_time::duration_ns_saturating(started.elapsed()),
                false,
            );
            state.disable_optimized_region(plan.region);
            Ok(None)
        }
        runmat_runtime::numeric_region::NumericRegionExecution::Cancelled => {
            placement.observe(
                runmat_time::duration_ns_saturating(started.elapsed()),
                false,
            );
            *exit = NativeExit::cancelled(NativeCancellation {
                reason: NativeCancellationReason::REQUESTED,
                reserved: 0,
                generation: 1,
            });
            Ok(Some(NativeSiteOutcome::exit()))
        }
    }
}
