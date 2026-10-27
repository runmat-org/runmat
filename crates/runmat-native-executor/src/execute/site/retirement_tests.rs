use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::execute::state::{HostStateInput, NativeFunctionInputs};
use crate::region::placement_test_support::Probe;
use runmat_native_codegen::{NativeLocalId, NativeLocalKind, NativeLocalMetadata, NativeValueId};
use runmat_runtime::context::{
    host_allocation::{HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt},
    host_effect::ManagedHostEffectLedger,
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use runmat_runtime::native::*;
use runmat_value::{ExactHostTensor2dAllocation, NumericDType, Value};
use std::{rc::Rc, sync::Arc};

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
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn assert_values(state: &HostState, reference: NativeValueRef, sign: f64, width: usize) -> u64 {
    let Value::Tensor(value) = state
        .arena
        .get(reference)
        .expect("live numerical generation")
    else {
        panic!("numeric tensor output")
    };
    assert_eq!(value.shape(), &[width, 1]);
    let data = value.materialize_f64();
    for (index, actual) in data.iter().enumerate() {
        assert_eq!(*actual, sign * (index + 1) as f64);
    }
    value.host_buffer().checked_byte_len().unwrap() as u64
}

#[test]
fn numerical_site_loop_retires_payloads_before_invocation_teardown() {
    runmat_gc::gc_test_context(|| {
        // One long fixed-geometry loop and one bounded large-payload stress.
        // A 1 MiB output must not wait for 64 insertions under a 16 MiB budget.
        for (width, iterations, steady_start, steady_end) in [
            (1024usize, 4096usize, 256usize, 512usize),
            (131072, 96, 32, 64),
        ] {
            let ledger = ManagedHostEffectLedger::new(1 << 24);
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            );
            let template = observed_state(&Rc::new(Probe::default()));
            let mut function = template.function.clone();
            drop(template);
            // Match ActiveInvocation::enter_once: a raw VTable callback does
            // not itself activate the selected runtime for nested allocations.
            let runtime_guard = runtime.enter();
            assert_eq!(
                HostAllocationReceipt::policy(Some(&runtime)).unwrap(),
                HostAllocationPolicy::Managed,
            );
            function.abi.fixed_inputs = vec![NativeLocalId(0)];
            function.binding_locals = vec![runmat_native_codegen::NativeBindingLocal {
                binding: runmat_types::BindingId(0),
                local: NativeLocalId(0),
            }];
            function.locals = vec![NativeLocalMetadata {
                id: NativeLocalId(0),
                binding: Some(runmat_types::BindingId(0)),
                name: Some("x".into()),
                kind: NativeLocalKind::Binding,
            }];
            let terminator = function.blocks[0].terminator.clone();
            let mut source_site = terminator.site.clone();
            source_site.phase = runmat_native_codegen::NativeSitePhase::Rvalue;
            source_site.construct = runmat_mir::MirConstructKind::Unary;
            function.expected_sites.push(source_site.clone());
            function.blocks[0].instructions.push(NativeInstruction {
                id: runmat_native_codegen::NativeInstructionId(0),
                site: source_site,
                source: terminator.source,
                class: runmat_mir::NativeLoweringClass::NativeOperation,
                effects: Default::default(),
                capabilities: Default::default(),
                inputs: vec![],
                outputs: vec![runmat_native_codegen::NativeOutput {
                    value: NativeValueId(0),
                    value_type: runmat_native_codegen::NativeValueType::Generic,
                    local: Some(NativeLocalId(0)),
                }],
                effect_epoch_output: None,
                embedded_constructs: vec![],
                operation: NativeOperation::Rvalue {
                    value: runmat_mir::MirRvalue::Unary(
                        runmat_types::OperatorKind::UnaryMinus,
                        runmat_mir::MirOperand::Local(runmat_mir::MirLocalId(0)),
                    ),
                    result: runmat_native_codegen::NativeRvalueResult::Assignment,
                },
                safepoint: None,
                frame_state: Some(terminator.frame_state),
            });
            // Tensor::new adopts an ordinary Vec; it does not attach a claim
            // merely because a runtime is active. Fund this original alias
            // through the existing exact numeric owner before construction.
            let physical =
                ExactHostTensor2dAllocation::requested_bytes(NumericDType::F64, width, 1).unwrap();
            let retained =
                HostAllocationPlan::retained_exact_output_bytes(physical as u64, 2).unwrap();
            let receipt = HostAllocationReceipt::admit(
                Some(&runtime),
                HostAllocationPlan::new(0, retained).unwrap(),
            )
            .unwrap()
            .expect("managed original numerical payload");
            let input = receipt
                .with_output_allocation(&runtime, |permit| {
                    let mut tensor =
                        ExactHostTensor2dAllocation::new(NumericDType::F64, width, 1, permit)?;
                    for (index, value) in tensor.as_f64_slice_mut().unwrap().iter_mut().enumerate()
                    {
                        *value = (index + 1) as f64;
                    }
                    tensor.into_tensor().map(Value::Tensor)
                })
                .unwrap();
            drop(receipt);
            let seed_claimed_bytes = ledger.snapshot().live_claimed_bytes;
            assert!(seed_claimed_bytes >= (width * 8) as u64);
            let (mut state, _) = HostState::new(HostStateInput {
                functions: Arc::new(vec![function.clone()]),
                function,
                inputs: NativeFunctionInputs::legacy(vec![input]),
                requested_outputs: 0,
                runtime: runtime.clone(),
                program_capture: None,
                captures: vec![],
                lexical_environment: None,
                deoptimization: Default::default(),
                interpreter_resume_points: Default::default(),
                coverage_sites: Default::default(),
                osr_point: None,
                optimized_regions: Default::default(),
                workspace: None,
            })
            .unwrap();
            state.reconcile_external_workspace().unwrap();
            let alias = state.local_reference(0).unwrap();
            state.arena.retain(alias).unwrap();
            let mut resume = NativeResumeState::default();
            let mut frame = NativeFrame {
                resume: &mut resume,
                ..Default::default()
            };
            let mut call = NativeCall {
                frame: &mut frame,
                ..Default::default()
            };
            let host = crate::execute::callbacks::table(&mut state);
            let execute = host.execute_site.unwrap();
            let request = NativeSiteRequest {
                function: 0,
                block: 0,
                position: 0,
                phase: NativeSitePhase::RVALUE,
                ordinal: 0,
                reserved: 0,
            };
            let mut steady = (0, 0, 0, 0);
            let mut warm = (0, 0, 0);
            // This is one retained invocation state driven through its real host
            // ABI, not a claim about generated machine-loop code. The existing
            // compiled generic-for witness independently covers native backedges.
            for iteration in 1..=(iterations + 64) {
                let mut outcome = NativeSiteOutcome::continue_execution();
                let mut exit = NativeExit::completed(0);
                // SAFETY: all callback/frame/resume/state backing is retained for
                // this synchronous entry and the exact continuation below.
                assert_eq!(
                    unsafe { execute(host.context, &mut call, &request, &mut outcome, &mut exit) },
                    NativeHostStatus::OK
                );
                assert_eq!(outcome.kind, NativeSiteOutcomeKind::EXIT);
                assert_eq!(exit.kind, NativeExitKind::SUSPENDED);
                assert_eq!(
                    futures::executor::block_on(crate::execute::call_suspension::complete(
                        &mut state,
                        exit.suspension.continuation,
                        exit.suspension.generation,
                    ))
                    .unwrap(),
                    request
                );
                state.prepare_resume(resume).unwrap();
                // SAFETY: exact reentry uses the original live ABI records.
                assert_eq!(
                    unsafe { execute(host.context, &mut call, &request, &mut outcome, &mut exit) },
                    NativeHostStatus::OK
                );
                assert_eq!(outcome.kind, NativeSiteOutcomeKind::CONTINUE);
                let output = state.values[&NativeValueId(0)];
                let current_payload_bytes = assert_values(
                    &state,
                    output,
                    if iteration % 2 == 0 { 1. } else { -1. },
                    width,
                );
                state.set_local(0, output).unwrap();
                state.reconcile_external_workspace().unwrap();
                let (live, slots, capacity) = state.arena.retirement_stats();
                let snapshot = ledger.snapshot();
                let bytes = snapshot.live_claimed_bytes + snapshot.live_reserved_bytes;
                // UnaryMinus produces a distinct nonzero numeric buffer. A
                // managed seed alone must never satisfy the output oracle.
                let expected_claimed = current_payload_bytes
                    + if iteration <= iterations {
                        seed_claimed_bytes
                    } else {
                        0
                    };
                assert!(
                    snapshot.live_claimed_bytes >= expected_claimed,
                    "managed seed/output payload absent: claimed={}, reserved={}, expected={expected_claimed}, iteration={iteration}, width={width}",
                    snapshot.live_claimed_bytes,
                    snapshot.live_reserved_bytes,
                );
                if iteration == 32 {
                    warm = (live, slots, capacity);
                }
                // Full natural sweep windows, including byte-pressure retirement.
                // No test-only collection is forced.
                if (steady_start..=steady_end).contains(&iteration) {
                    steady.0 = steady.0.max(live);
                    steady.1 = steady.1.max(slots);
                    steady.2 = steady.2.max(capacity);
                    steady.3 = steady.3.max(bytes);
                }
                if iteration > steady_end {
                    assert!(live <= steady.0, "live entries grew with loop length");
                    assert!(slots <= steady.1, "arena slots grew with loop length");
                    assert!(capacity <= steady.2, "arena capacity grew with loop length");
                    assert!(
                        bytes <= steady.3,
                        "retained managed payload grew with loop length"
                    );
                }
                if iteration <= iterations {
                    assert_values(&state, alias, 1., width);
                }
                if iteration == iterations {
                    assert!(warm.0 > 0 && warm.1 > 0 && warm.2 > 0);
                    state.arena.release(alias).unwrap();
                }
            }
            assert!(matches!(
                state.arena.get(alias),
                Err(NativeExecutorError::StaleValue)
            ));
            assert!(state.pending_call.is_none());
            assert!(state.completed_call.is_none());
            drop(state);
            drop(runtime_guard);
            drop(runtime);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    });
}
