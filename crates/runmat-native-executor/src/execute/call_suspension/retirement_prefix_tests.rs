use std::rc::Rc;

use runmat_native_codegen::NativeSitePhase as CodegenPhase;
use runmat_runtime::native::{
    NativeCall, NativeExit, NativeResumeState, NativeSiteOutcome, NativeSiteOutcomeKind,
    NativeSitePhase, NativeSiteRequest,
};

use crate::execute::{region::observation::test_support::observed_state, site};
use crate::region::placement_test_support::Probe;

use super::{CompletedCall, CompletedPayload};

#[test]
fn completed_retirement_skips_block_prefix_before_exact_site() {
    let mut state = observed_state(&Rc::new(Probe::default()));
    let mut prefix_site = state.function.blocks[0].terminator.site.clone();
    prefix_site.point.position = 0;
    prefix_site.phase = CodegenPhase::Rvalue;
    let mut retired_site = prefix_site.clone();
    retired_site.point.position = 1;
    retired_site.phase = CodegenPhase::Statement;
    state.function.expected_sites = vec![prefix_site, retired_site];

    let prefix = NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    };
    let retired = NativeSiteRequest {
        position: 1,
        phase: NativeSitePhase::STATEMENT,
        ..prefix
    };
    let decision = NativeSiteOutcome::continue_execution();
    state.completed_call = Some(
        CompletedCall::new(
            retired,
            None,
            CompletedPayload::ValueRetirement(decision),
            None,
            None,
        )
        .unwrap(),
    );
    state
        .prepare_resume(NativeResumeState {
            function: retired.function,
            block: retired.block,
            position: retired.position,
            phase: retired.phase.0,
            ordinal: retired.ordinal,
            ..NativeResumeState::default()
        })
        .unwrap();

    let mut call = NativeCall::default();
    let mut exit = NativeExit::completed(0);
    // Model Cranelift's block-entry callback sequence: the prefix callback
    // must not consume or reject the exact retired site's completion.
    let before = site::execute(&mut state, &mut call, prefix, &mut exit).unwrap();
    assert_eq!(before.kind, NativeSiteOutcomeKind::CONTINUE);
    assert!(state.completed_call.is_some());
    let resumed = site::execute(&mut state, &mut call, retired, &mut exit).unwrap();
    assert_eq!(resumed.kind, NativeSiteOutcomeKind::CONTINUE);
    assert!(state.completed_call.is_none());
}
