use runmat_native_codegen::{
    NativeFrameState, NativeMirSite, NativeRegionBoundary, NativeRegionBoundaryKind,
    NativeSafepointId,
};
use runmat_runtime::native::{NativeCall, NativeDeoptReason, NativeDeoptimization, NativeExit};

use crate::specialization::{GuardFailure, GuardFailureKind};
use crate::{JitError, JitResult};

use super::state::HostState;

pub(super) fn checkpoint(
    state: &mut HostState,
    call: &mut NativeCall,
    site: &NativeMirSite,
    frame_state: &NativeFrameState,
    boundaries: &[NativeRegionBoundary],
    safepoint: Option<NativeSafepointId>,
    exit: &mut NativeExit,
) -> JitResult<bool> {
    for boundary in boundaries.iter().filter(|boundary| {
        boundary.kind == NativeRegionBoundaryKind::Entry && boundary.point == site.point
    }) {
        for guard in &boundary.guards {
            let failure = if state.should_inject_guard(guard.contract.id) {
                Some(GuardFailure {
                    guard: guard.contract.id,
                    deopt: guard.contract.deopt,
                    kind: GuardFailureKind::for_condition(&guard.contract.condition),
                })
            } else {
                state.evaluate_guard(guard).err()
            };
            if let Some(failure) = failure {
                state.retire_guard(failure.guard);
                install_exit(
                    state,
                    call,
                    site,
                    &boundary.frame_state,
                    HostState::deoptimization_reason(failure.kind),
                    guard_token(failure.guard),
                    exit,
                )?;
                return Ok(true);
            }
        }
    }

    if safepoint.is_some_and(|safepoint| state.should_inject_safepoint(safepoint)) {
        install_exit(
            state,
            call,
            site,
            frame_state,
            NativeDeoptReason::EXPLICIT_SLOW_PATH,
            safepoint_token(safepoint.expect("checked safepoint")),
            exit,
        )?;
        return Ok(true);
    }
    Ok(false)
}

fn install_exit(
    state: &mut HostState,
    call: &mut NativeCall,
    site: &NativeMirSite,
    frame_state: &NativeFrameState,
    reason: NativeDeoptReason,
    identity: u64,
    exit: &mut NativeExit,
) -> JitResult<()> {
    if call.frame.is_null() {
        return Err(JitError::Host(
            "native call has no frame for deoptimization".into(),
        ));
    }
    // SAFETY: NativeCall validation and invocation ownership keep the frame and
    // resume state live until generated code returns to GenericInvocation.
    let resume = unsafe { (*call.frame).resume };
    if resume.is_null() {
        return Err(JitError::Host(
            "native frame has no exact deoptimization resume state".into(),
        ));
    }
    let materialized = state.materialize_deoptimization(frame_state, site)?;
    let operand_depth = u32::try_from(materialized.operands.len())
        .map_err(|_| JitError::Host("deoptimization operand depth exceeds native ABI".into()))?;
    let local_count = u32::try_from(materialized.locals.len())
        .map_err(|_| JitError::Host("deoptimization local count exceeds native ABI".into()))?;
    // SAFETY: resume points at invocation-owned writable storage validated above.
    unsafe {
        (*resume).bytecode_pc = materialized.site.bytecode_pc.unwrap_or(0);
        (*resume).operand_depth = operand_depth;
        (*resume).local_count = local_count;
        (*resume).side_effect_epoch = materialized.site.side_effect_epoch;
    }
    *exit = NativeExit::deoptimized(NativeDeoptimization {
        reason,
        target: state.effective_deoptimization_target(&materialized),
        guard: identity,
        resume,
    });
    Ok(())
}

fn guard_token(guard: runmat_types::RegionGuardId) -> u64 {
    (u64::from(guard.region.ordinal) << 32) | u64::from(guard.ordinal)
}

fn safepoint_token(safepoint: NativeSafepointId) -> u64 {
    (1_u64 << 63) | (u64::from(safepoint.0) + 1)
}
