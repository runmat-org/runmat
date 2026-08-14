use runmat_native_codegen::{NativeFrameState, NativeMirSite};
use runmat_runtime::native::{NativeCall, NativeDeoptReason, NativeExit, NativeResumeKind};

use crate::JitResult;

use super::state::HostState;

pub(super) fn checkpoint(
    state: &mut HostState,
    call: &mut NativeCall,
    site: &NativeMirSite,
    frame_state: &NativeFrameState,
    exit: &mut NativeExit,
) -> JitResult<bool> {
    if !state.take_osr_request(site.point) {
        return Ok(false);
    }
    super::deoptimization::install_exit_for_target(
        state,
        call,
        site,
        frame_state,
        super::deoptimization::NativeTransfer {
            reason: NativeDeoptReason::EXPLICIT_SLOW_PATH,
            target: NativeResumeKind::OPTIMIZED_NATIVE,
            identity: osr_token(site.point),
        },
        exit,
    )?;
    Ok(true)
}

fn osr_token(point: runmat_types::ProgramPointId) -> u64 {
    (1_u64 << 62) | (u64::from(point.block) << 31) | u64::from(point.position)
}
