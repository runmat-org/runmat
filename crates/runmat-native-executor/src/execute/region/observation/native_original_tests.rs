use super::*;
use crate::region::placement_test_support::{plan, Probe};
use runmat_runtime::numeric_region::service::{
    NativeNumericRegionRequest, NativeNumericRegionRight, NativeOriginalRegionObservation,
    SourceRegionRouteAudit,
};
use std::{cell::Cell, rc::Rc};

struct Completion(Rc<Cell<usize>>);
impl NativeOriginalRegionObservation for Completion {
    fn audit(&self) -> Option<SourceRegionRouteAudit> {
        None
    }
    fn complete(
        self: Box<Self>,
        runtime: runmat_runtime::context::RuntimeContext,
        right: NativeNumericRegionRight,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<'static, ()>,
        runmat_runtime::RuntimeError,
    > {
        let admission = runtime.clone();
        runmat_runtime::execution::admit_native_continuation_with(&admission, || async move {
            assert!(right.is_current(&runtime));
            self.0.set(self.0.get() + 1);
            Ok(())
        })
    }
}

fn request(site: SiteIdentity) -> NativeSiteRequest {
    NativeSiteRequest {
        function: site.point.function.0,
        block: site.point.block,
        position: site.point.position,
        phase: if site.phase == runmat_native_codegen::NativeSitePhase::Terminator {
            NativeSitePhase::TERMINATOR
        } else {
            NativeSitePhase::RVALUE
        },
        ordinal: site.ordinal,
        reserved: 0,
    }
}

#[test]
fn original_native_completion_suspends_only_after_all_source_publications_and_consumes_once() {
    runmat_gc::gc_test_context(|| {
        let probe = Rc::new(Probe::default());
        let mut state = super::test_support::observed_state(&probe);
        state.region_observation = None;
        let plan = plan();
        let entry = request(plan.entry);
        let (_program_owner, _, right) = NativeNumericRegionRequest::capture_verified(
            &state.runtime,
            plan.region,
            entry,
            &plan.program,
            &[&runmat_value::Value::Num(1.0)],
        )
        .unwrap()
        .unwrap()
        .into_parts();
        let completed = Rc::new(Cell::new(0));
        let observation = RegionObservation::original(
            state.runtime.clone(),
            right,
            Some(Box::new(Completion(completed.clone()))),
        );
        state.region_observation = Some(GenericRegionObservation::new(
            OptimizedRegionPlanLease::from_test_plan(plan.clone()),
            observation,
        ));
        after_site(
            &mut state,
            entry,
            &Ok(NativeSiteOutcome::continue_execution()),
        )
        .unwrap();
        assert_eq!(completed.get(), 0);
        let mut exit_site = plan.entry;
        exit_site.point = plan.exit_site.point;
        exit_site.phase = runmat_native_codegen::NativeSitePhase::Terminator;
        let exit = request(exit_site);
        assert!(matches!(
            before_site(&mut state, exit),
            Err(crate::NativeExecutorError::CallSuspended)
        ));
        assert_eq!(completed.get(), 0, "no poll at source publication boundary");
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        futures::executor::block_on(crate::execute::call_suspension::complete(
            &mut state,
            continuation,
            generation,
        ))
        .unwrap();
        assert_eq!(completed.get(), 1);
        before_site(&mut state, exit).unwrap();
        before_site(&mut state, exit).unwrap();
        assert!(state.completed_call.is_none());
        assert_eq!(completed.get(), 1);
    });
}

#[test]
fn exact_hostcall_suspension_retains_source_cursor_without_publishing_a_sample() {
    runmat_gc::gc_test_context(|| {
        let probe = Rc::new(Probe::default());
        let mut state = super::test_support::observed_state(&probe);
        let entry = request(plan().entry);
        state.pending_call = Some(crate::execute::call_suspension::PendingCall {
            continuation: 1,
            generation: 1,
            request: entry,
            embedded: None,
            retained_roots: Vec::new(),
            future: crate::execute::call_suspension::admit_completion_with(&state, || async {
                Ok(crate::execute::call_suspension::CallCompletion::StatementOutput)
            })
            .unwrap(),
        });
        after_site(&mut state, entry, &Ok(NativeSiteOutcome::exit())).unwrap();
        assert_eq!(state.region_observation.as_ref().unwrap().cursor, 0);
        assert!(probe.feedback.borrow().is_empty());
        state.pending_call = None;
        before_site(&mut state, entry).unwrap();
        after_site(
            &mut state,
            entry,
            &Ok(NativeSiteOutcome::continue_execution()),
        )
        .unwrap();
        assert_eq!(state.region_observation.as_ref().unwrap().cursor, 1);
    });
}
