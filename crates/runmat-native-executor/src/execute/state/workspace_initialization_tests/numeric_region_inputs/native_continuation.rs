use super::*;
use runmat_runtime::context::{RuntimeAccelerationService, RuntimeServicePorts};
use runmat_runtime::numeric_region::service::*;
use std::cell::Cell;

struct Adapter {
    calls: Rc<Cell<usize>>,
    publications: Rc<Cell<usize>>,
    fail: bool,
}
struct Publication(Rc<Cell<usize>>);
impl NativeNumericRegionPublication for Publication {
    fn complete(self: Box<Self>, runtime: &RuntimeContext, right: NativeNumericRegionRight) {
        assert!(right.is_current(runtime));
        self.0.set(self.0.get() + 1);
    }
}
impl RuntimeAccelerationService for Adapter {
    fn numeric_region_capability(&self) -> Option<NumericRegionCapability> {
        Some(NumericRegionCapability::DenseF64)
    }
    fn execute_numeric_region(
        &self,
        runtime: RuntimeContext,
        request: NativeNumericRegionRequest,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<'static, NativeNumericRegionOutcome>,
        runmat_runtime::RuntimeError,
    > {
        let calls = self.calls.clone();
        let publications = self.publications.clone();
        let fail = self.fail;
        let admission = runtime.clone();
        runmat_runtime::execution::admit_native_continuation_with(&admission, || async move {
            calls.set(calls.get() + 1);
            let (_program_owner, _, right) = request.into_parts();
            if fail {
                return Err(runmat_runtime::runtime_error::semantic_error(
                    "NativeRegionInjected",
                    "injected provider failure",
                ));
            }
            let outputs = root_values(&runtime, 1, |_| Ok(Value::Num(42.0)))?;
            Ok(NativeNumericRegionOutcome::Completed {
                outputs,
                right,
                publication: Some(NativeNumericRegionPublicationOwner::new(
                    Box::new(Publication(publications)),
                    None,
                )),
            })
        })
    }
}

fn site() -> runmat_runtime::native::NativeSiteRequest {
    runmat_runtime::native::NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: runmat_runtime::native::NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    }
}

fn start(
    fail: bool,
) -> (
    HostState,
    Rc<Cell<usize>>,
    Rc<Cell<usize>>,
    runmat_execution::cancellation::CancellationSource,
) {
    let calls = Rc::new(Cell::new(0));
    let publications = Rc::new(Cell::new(0));
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(Adapter {
            calls: calls.clone(),
            publications: publications.clone(),
            fail,
        })),
    );
    let mut state = omitted_input_state_with(runtime, None);
    let input = state.arena.insert(Value::Num(7.0));
    state.set_local(0, input).unwrap();
    let mut plan = crate::region::placement_test_support::plan();
    plan.program
        .nodes
        .push(runmat_runtime::numeric_region::NumericRegionNode::Unary {
            operation: runmat_runtime::numeric_region::NumericUnaryOperation::Minus,
            input: 0,
        });
    plan.program.outputs[0] = 1;
    // The appended unary operation executes at this actual native rvalue
    // occurrence. Keep the verified site's transcript complete.
    plan.operation_sites.push(plan.entry);
    plan.skipped_sites.insert(crate::region::SiteIdentity {
        point: runmat_types::ProgramPointId {
            function: plan.region.function,
            block: 0,
            position: 1,
        },
        phase: runmat_native_codegen::NativeSitePhase::Rvalue,
        ordinal: 0,
    });
    state.optimized_regions.insert(
        plan.entry,
        crate::region::OptimizedRegionPlanLease::from_test_plan(plan),
    );
    state.current_request = Some(site());
    (state, calls, publications, cancellation)
}

#[test]
fn native_region_suspends_before_poll_and_publishes_once_after_completion() {
    runmat_gc::gc_test_context(|| {
        let (mut state, calls, publications, _cancellation) = start(false);
        let mut exit = runmat_runtime::native::NativeExit::completed(0);
        assert!(matches!(
            crate::execute::region::checkpoint(&mut state, site(), &mut exit),
            Err(NativeExecutorError::CallSuspended)
        ));
        assert_eq!(calls.get(), 0);
        assert_eq!(publications.get(), 0);
        assert!(state.skipped_optimized_sites.is_empty());
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        futures::executor::block_on(crate::execute::call_suspension::complete(
            &mut state,
            continuation,
            generation,
        ))
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(publications.get(), 0);
        assert!(
            crate::execute::region::checkpoint(&mut state, site(), &mut exit)
                .unwrap()
                .is_some()
        );
        assert_eq!(publications.get(), 1);
        assert_eq!(state.vectorized_regions(), 1);
        assert!(!state.skipped_optimized_sites.is_empty());
        assert!(
            crate::execute::call_suspension::numeric_region::take(&mut state, site())
                .unwrap()
                .is_none()
        );
    });
}

#[test]
fn native_region_error_or_unpolled_drop_never_publishes_or_skips() {
    runmat_gc::gc_test_context(|| {
        for poll in [false, true] {
            let (mut state, calls, publications, _cancellation) = start(true);
            let mut exit = runmat_runtime::native::NativeExit::completed(0);
            assert!(matches!(
                crate::execute::region::checkpoint(&mut state, site(), &mut exit),
                Err(NativeExecutorError::CallSuspended)
            ));
            if poll {
                let pending = state.pending_call.as_ref().unwrap();
                let (continuation, generation) = (pending.continuation, pending.generation);
                assert!(
                    futures::executor::block_on(crate::execute::call_suspension::complete(
                        &mut state,
                        continuation,
                        generation
                    ))
                    .is_err()
                );
            }
            assert_eq!(calls.get(), usize::from(poll));
            assert_eq!(publications.get(), 0);
            assert!(state.skipped_optimized_sites.is_empty());
            drop(state);
            assert_eq!(publications.get(), 0);
        }

        // Actual Managed completion abandonment, after the provider has
        // returned but before the native source site takes/publishes it.
        use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
        use runmat_runtime::context::{
            ParallelCapability, RuntimeParallelResources, RuntimeParallelService,
        };
        struct Parallel(ManagedHostEffectLedger);
        impl RuntimeParallelService for Parallel {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let snapshot = self.0.snapshot();
                RuntimeParallelResources {
                    memory_available_bytes: Some(snapshot.free_bytes()),
                    epoch: snapshot.epoch,
                    ..RuntimeParallelResources::default()
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }
        let (mut state, calls, publications, _cancellation) = start(false);
        let ledger = ManagedHostEffectLedger::new(8 * 1024 * 1024);
        let ports = state
            .runtime
            .service_ports()
            .clone()
            .with_parallel(Rc::new(Parallel(ledger.clone())));
        state.runtime = state.runtime.clone().with_service_ports(ports);
        let mut exit = runmat_runtime::native::NativeExit::completed(0);
        assert!(matches!(
            crate::execute::region::checkpoint(&mut state, site(), &mut exit),
            Err(NativeExecutorError::CallSuspended)
        ));
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        futures::executor::block_on(crate::execute::call_suspension::complete(
            &mut state,
            continuation,
            generation,
        ))
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert!(ledger.snapshot().live_reserved_bytes > 0);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        assert!(state.completed_call.is_some());
        assert_eq!(publications.get(), 0);
        assert!(state.skipped_optimized_sites.is_empty());
        drop(state);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(publications.get(), 0);
    });
}

#[test]
fn native_pass_through_publication_uses_original_generation_not_refreshed_local() {
    runmat_gc::gc_test_context(|| {
        let (mut state, _, _, _cancellation) = start(false);
        let mut plan = crate::region::placement_test_support::plan();
        let original = state.local_reference(0).unwrap();
        plan.outputs[0].source = crate::region::RegionOutputSource::Input(0);
        let changed = state.arena.insert(Value::Num(99.0));
        state.set_local(0, changed).unwrap();
        state
            .publish_optimized_region_from_inputs(&plan, &[original], vec![Value::Num(42.0)])
            .unwrap();
        assert_eq!(
            state.arena.get(state.values[&plan.outputs[0].ssa]).unwrap(),
            &Value::Num(7.0)
        );
    });
}

#[test]
fn cancelled_native_completion_never_publishes_or_skips() {
    runmat_gc::gc_test_context(|| {
        let (mut state, _, publications, cancellation) = start(false);
        let mut exit = runmat_runtime::native::NativeExit::completed(0);
        assert!(matches!(
            crate::execute::region::checkpoint(&mut state, site(), &mut exit),
            Err(NativeExecutorError::CallSuspended)
        ));
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        futures::executor::block_on(crate::execute::call_suspension::complete(
            &mut state,
            continuation,
            generation,
        ))
        .unwrap();
        assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
        assert!(state.runtime.is_cancelled());
        assert!(
            crate::execute::region::checkpoint(&mut state, site(), &mut exit)
                .unwrap()
                .is_some()
        );
        assert_eq!(publications.get(), 0);
        assert!(state.skipped_optimized_sites.is_empty());
    });
}
