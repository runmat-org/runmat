//! Production-runner placement observation witnesses using the deterministic provider.
use super::*;

fn capturing_fusion_placement(
    session: Rc<runmat_accelerate::placement::PlacementSession>,
) -> Rc<placement_feedback::FeedbackPlacement<()>> {
    Rc::new(placement_feedback::FeedbackPlacement {
        session,
        reduction_selection: Default::default(),
        delivered: Default::default(),
        invalidate_selected_provider: Default::default(),
        revoked_before_publication: Default::default(),
    })
}

#[test]
fn automatic_resident_general_chain_uses_one_private_host_mirror_through_full_call() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        let provider = registered_provider().expect("test provider");
        provider.reject_fusion.store(true, Ordering::Relaxed);
        let _reject = RejectFusion(provider);
        for single in [false, true] {
            // A fresh process with RUNMAT_ALLOW_PRECISION_DOWNCAST=1 exercises
            // provider-effective Single facts while the CPU exp intermediate
            // below must retain the original Double host allocation contract.
            provider.report_f32.store(
                single
                    && runmat_accelerate::precision::downcast_permitted(
                        runmat_value::NumericDType::F64,
                    ),
                Ordering::Relaxed,
            );
            for decoded in [false, true] {
                runmat_accelerate::placement::reset();
                let expression = "exp(-h).*sin(4*x)+tanh(x).*tanh(x)";
                let source = if decoded {
                    format!("function y = resident_chain(x,h)\ny = {expression};\nend\nx = [0.25, 0.5, 0.75]; h = [0.25, 0.5, 0.75]; y = resident_chain(x,h);")
                } else {
                    format!("x = [0.25, 0.5, 0.75]; h = [0.25, 0.5, 0.75]; y = {expression};")
                };
                let mut bytecode = compile_semantic(&source);
                if decoded {
                    let encoded =
                        runmat_vm::encode_interpreter_program_v2(&bytecode.function_registry())
                            .unwrap();
                    let registry = runmat_vm::decode_interpreter_program_v2(&encoded).unwrap();
                    bytecode.bound_functions = registry.clone().into_functions();
                    bytecode.function_registry = registry;
                }
                let values = [0.25, 0.5, 0.75];
                let single_values = values.map(|value| value as f32);
                let data = if single {
                    HostNumericDataView::F32(&single_values)
                } else {
                    HostNumericDataView::F64(&values)
                };
                let handle = runmat_accelerate_api::upload_numeric_protecting(
                    &**provider,
                    &HostNumericTensorView {
                        data,
                        shape: &[1, 3],
                        storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    },
                    &[],
                )
                .unwrap()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap();
                let resume = resume_after_input_store(&bytecode, Value::GpuTensor(handle.clone()));
                let capture = capturing_fusion_placement(Rc::new(
                    runmat_accelerate::placement::PlacementSession::default(),
                ));
                let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                ))
                .with_service_ports(
                    runmat_runtime::context::RuntimeServicePorts::default()
                        .with_placement(capture.clone()),
                );
                let downloads = provider.numeric_downloads.load(Ordering::Relaxed);
                let allocations = provider.next_id.load(Ordering::Relaxed);
                let dispatches = provider.fused_dispatches.load(Ordering::Relaxed);
                let runmat_vm::InterpreterOutcome::Completed(completion) = block_on(
                    runmat_vm::interpret_resume_in_context(&bytecode, resume, None, runtime),
                )
                .expect("ordinary compiled resident-to-host FullCall route");
                let vars = &completion.values;
                let output = vars
                    .get(fusion_var_index(&bytecode, "y"))
                    .expect("published host result");
                let Value::Tensor(output) = &*output else {
                    panic!("automatic CPU selection must publish a host result");
                };
                assert_eq!(
                    output.numeric_dtype(),
                    if single {
                        runmat_value::NumericDType::F32
                    } else {
                        runmat_value::NumericDType::F64
                    }
                );
                for (actual, x) in output.materialize_f64().iter().zip(values) {
                    let expected = (-x).exp() * (4.0 * x).sin() + x.tanh() * x.tanh();
                    assert!((actual - expected).abs() < if single { 2e-6 } else { 1e-12 });
                }
                assert!(
                    matches!(vars.get(fusion_var_index(&bytecode, "x")).as_deref(), Some(Value::GpuTensor(actual))
                    if actual.view_identity() == handle.view_identity()
                        && actual.wire_descriptor() == handle.wire_descriptor()),
                    "workspace keeps its exact original resident source"
                );
                let host = vars
                    .get(fusion_var_index(&bytecode, "h"))
                    .expect("ordinary host source");
                let Value::Tensor(host) = &*host else {
                    panic!("ordinary host source stays host-resident");
                };
                assert_eq!(host.numeric_dtype(), runmat_value::NumericDType::F64);
                assert_eq!(
                    provider.numeric_downloads.load(Ordering::Relaxed) - downloads,
                    1,
                    "all repeated ordinary loads share one exact logical-view readback: single={single}, decoded={decoded}, precision={:?}",
                    provider.precision(),
                );
                assert_eq!(provider.next_id.load(Ordering::Relaxed), allocations);
                assert_eq!(
                    provider.fused_dispatches.load(Ordering::Relaxed),
                    dispatches
                );
                assert_one_cpu_full_call(&capture);
            }
        }
    });
}

fn assert_one_cpu_full_call(capture: &placement_feedback::FeedbackPlacement<()>) {
    let feedback = capture.delivered.lock().unwrap();
    let cpu = feedback
        .iter()
        .filter(|feedback| feedback.candidate == "cpu.shared.fusion.elementwise")
        .collect::<Vec<_>>();
    assert_eq!(cpu.len(), 1, "one exact completed CPU FullCall receipt");
    assert!(cpu[0].succeeded);
    assert!(cpu[0].total_elapsed_ns > 0);
    assert!(matches!(
        cpu[0].signature.revision.authority,
        runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    ));
    assert!(
        capture
            .session
            .profile_snapshot()
            .feedback
            .iter()
            .all(|series| series.signature != cpu[0].signature),
        "session-only FullCall identity must not be profile exported"
    );
}

fn fusion_test_runtime(
    placement: Rc<runmat_accelerate::placement::PlacementSession>,
) -> runmat_runtime::context::RuntimeContext {
    runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        runmat_runtime::context::RuntimeServicePorts::default().with_placement(placement),
    )
}

fn fusion_var_index(bytecode: &runmat_vm::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(index, candidate)| (candidate == name).then_some(*index))
        .expect("named fixture variable")
}

pub(super) fn resume_after_input_store(
    bytecode: &runmat_vm::Bytecode,
    input: Value,
) -> runmat_vm::InterpreterResumeState {
    let slot = fusion_var_index(bytecode, "x");
    let store = bytecode
        .instructions
        .iter()
        .position(|instruction| matches!(instruction, Instr::StoreVar(index) if *index == slot))
        .expect("compiler input store");
    // The prefix is a literal matrix/range construction, not user code. Never
    // rewrite instructions or fusion windows to manufacture an observable span.
    for instruction in &bytecode.instructions[..store] {
        assert!(
            matches!(
                instruction,
                Instr::LoadConst(_)
                    | Instr::LoadInt(_)
                    | Instr::CreateRange(_)
                    | Instr::CreateMatrix(_, _)
                    | Instr::EnterScope(_)
            ) || matches!(instruction, Instr::CreateMatrixFromSequences { elements, .. } if elements.iter().all(|element| element.stack_operand_count() == 1)),
            "unexpected input-prefix instruction: {instruction:?}"
        );
    }
    let pc = store + 1;
    let groups = bytecode.runtime_fusion_groups();
    let function_region = bytecode.layout.as_ref().is_some_and(|layout| {
        let registry = bytecode.function_registry();
        bytecode.bound_functions.keys().any(|id| {
            registry.get(*id).is_some_and(|function| {
                !runmat_vm::Bytecode::for_function(function, registry.clone(), layout.clone())
                    .runtime_fusion_groups()
                    .is_empty()
            })
        })
    });
    assert!(
        groups.iter().any(|group| group.span.start >= pc) || function_region,
        "fixture must retain a script or admitted function fusion region"
    );
    assert!(
        groups
            .iter()
            .all(|group| !(group.span.start < pc && pc <= group.span.end)),
        "resume boundary must not split a fusion region"
    );
    let mut vars = vec![Value::Num(0.0); bytecode.var_count];
    vars[slot] = input;
    runmat_vm::InterpreterResumeState {
        pc,
        completion_boundary: None,
        vars: runmat_vm::InterpreterVars::from_owned(vars),
        assigned_slots: std::collections::HashSet::from([slot]),
        supplied_inputs: 0,
        requested_outputs: 0,
        missing_input_slots: Default::default(),
        global_aliases: Default::default(),
        persistent_aliases: Default::default(),
        side_effect_epoch: 0,
    }
}

fn assert_fusion_numeric_output(value: &Value, inputs: &[f64]) {
    let value = gather_if_needed(value).expect("gather result");
    let Value::Tensor(tensor) = value else {
        panic!("expected numeric tensor, got {value:?}");
    };
    assert_eq!(tensor.shape(), vec![1, inputs.len()]);
    for (actual, input) in tensor.materialize_f64().iter().zip(inputs) {
        let expected = input.sin() * input + 2.0;
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }
}

fn assert_provider_fusion_was_submitted() {
    use runmat_accelerate::placement::{PlacementEventKind, PlacementVariant};
    let report = runmat_accelerate::placement_report();
    assert!(
        report
            .traces
            .iter()
            .any(|trace| trace.operation == "fusion.elementwise"
                && trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Selected
                        && event.variant == Some(PlacementVariant::ProviderFusion))
                && trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Queue
                        && event.variant == Some(PlacementVariant::ProviderFusion)
                        && event.reason.as_deref() == Some("provider_submission"))
                && !trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Complete)),
        "expected selected and submitted provider fusion, not a host fallback"
    );
}

#[test]
fn fusion_execution_records_one_correlated_provider_lifecycle() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();

        let bytecode = compile_semantic(
            r#"
            x = 1:4096;
            y = sin(x) .* x + 2;
            "#,
        );
        let inputs = (1..=4096).map(f64::from).collect::<Vec<_>>();
        let resume = resume_after_input_store(
            &bytecode,
            Value::Tensor(Tensor::new(inputs.clone(), vec![1, 4096]).expect("host fixture")),
        );
        let runtime = fusion_test_runtime(Rc::new(
            runmat_accelerate::placement::PlacementSession::default(),
        ));
        let runmat_vm::InterpreterOutcome::Completed(completion) = block_on(
            runmat_vm::interpret_resume_in_context(&bytecode, resume, None, runtime),
        )
        .expect("fused execution from host input");
        let vars = &completion.values;
        let output = vars
            .get(fusion_var_index(&bytecode, "y"))
            .expect("completed fusion output");
        assert_fusion_numeric_output(&output, &(1..=4096).map(f64::from).collect::<Vec<_>>());
        assert_provider_fusion_was_submitted();

        let report = runmat_accelerate::placement_report();
        let trace = report
            .traces
            .iter()
            .find(|trace| trace.operation == "fusion.elementwise")
            .expect("elementwise fusion placement trace");
        assert!(trace.complete);
        assert_eq!(trace.dropped_events, 0);
        let kinds = trace
            .events
            .iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>();
        for required in [
            runmat_accelerate::placement::PlacementEventKind::Candidate,
            runmat_accelerate::placement::PlacementEventKind::Prepare,
            runmat_accelerate::placement::PlacementEventKind::Selected,
            runmat_accelerate::placement::PlacementEventKind::Upload,
            runmat_accelerate::placement::PlacementEventKind::Queue,
            runmat_accelerate::placement::PlacementEventKind::Kernel,
        ] {
            assert!(kinds.contains(&required), "missing {required:?}: {kinds:?}");
        }
        assert!(trace.events.iter().any(|event| {
            event.kind == runmat_accelerate::placement::PlacementEventKind::Prepare
                && event.reason.as_deref() == Some("generate_wgsl")
                && event.duration_ns.is_some()
        }));
        // This fixture submits source through the provider facade. It does not
        // expose a separate device pipeline compilation measurement.
        assert!(!kinds.contains(&runmat_accelerate::placement::PlacementEventKind::Compile));
        assert!(!kinds.contains(&runmat_accelerate::placement::PlacementEventKind::Complete));
        assert!(trace
            .events
            .iter()
            .any(|event| event.reason.as_deref() == Some("provider_submission")));
        let queue = trace
            .events
            .iter()
            .find(|event| event.kind == runmat_accelerate::placement::PlacementEventKind::Queue)
            .expect("queue event");
        assert_eq!(queue.reason.as_deref(), Some("host_dispatch"));
        assert!(queue.duration_ns.is_some());
        let kernel = trace
            .events
            .iter()
            .find(|event| event.kind == runmat_accelerate::placement::PlacementEventKind::Kernel)
            .expect("kernel event");
        assert_eq!(
            kernel.reason.as_deref(),
            Some("provider_timing_unavailable")
        );
        assert_eq!(kernel.duration_ns, None);
    });
}

#[test]
fn fusion_submission_does_not_train_completed_provider_execution() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_placement(placement.clone()),
        );
        let bytecode = compile_semantic(
            r#"
            x = 1:4096;
            y = sin(x) .* x + 2;
            "#,
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime,
                None,
                None,
            ))
            .outcome
            .expect("session-owned fusion execution");
        let vars = &completion.values;
        assert_fusion_numeric_output(
            &vars
                .get(fusion_var_index(&bytecode, "y"))
                .expect("completed fusion output"),
            &(1..=4096).map(f64::from).collect::<Vec<_>>(),
        );
        assert_provider_fusion_was_submitted();

        let profile = placement.profile_snapshot();
        assert!(
            !profile.feedback.iter().any(|series| {
                series.candidate.starts_with("provider.")
                    && series.candidate.contains("fusion.elementwise")
                    && series.samples > 0
            }),
            "provider enqueue must not become a completed-execution sample"
        );
    });
}

#[test]
fn fusion_complete_cost_falls_back_before_transfer_for_small_host_inputs() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();

        let bytecode = compile_semantic(
            r#"
            x = [0.25, 0.5];
            y = sin(x) .* x + 2;
            "#,
        );
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let capture = capturing_fusion_placement(placement.clone());
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_placement(capture.clone()),
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime,
                None,
                None,
            ))
            .outcome
            .expect("shared-runtime selection");
        let vars = &completion.values;
        assert_fusion_numeric_output(
            &vars
                .get(fusion_var_index(&bytecode, "y"))
                .expect("completed fusion output"),
            &[0.25, 0.5],
        );
        assert_one_cpu_full_call(&capture);

        let report = runmat_accelerate::placement_report();
        let cpu_traces = report
            .traces
            .iter()
            .filter(|trace| {
                trace.operation == "fusion.elementwise"
                    && trace.events.iter().any(|event| {
                        event.kind == runmat_accelerate::placement::PlacementEventKind::Selected
                            && event.variant
                                == Some(
                                    runmat_accelerate::placement::PlacementVariant::SharedRuntime,
                                )
                            && event.reason.as_deref() == Some("complete_cost")
                    })
            })
            .collect::<Vec<_>>();
        assert!(
            !cpu_traces.is_empty(),
            "expected a cost-selected CPU fusion region"
        );
        for trace in cpu_traces {
            assert!(!trace.events.iter().any(|event| {
                event.kind == runmat_accelerate::placement::PlacementEventKind::Fallback
            }));
            assert!(trace.events.iter().any(|event| {
                event.kind == runmat_accelerate::placement::PlacementEventKind::Complete
                    && event.variant
                        == Some(runmat_accelerate::placement::PlacementVariant::SharedRuntime)
                    && event.duration_ns.is_some()
            }));
            assert!(!trace.events.iter().any(|event| {
                matches!(
                    event.kind,
                    runmat_accelerate::placement::PlacementEventKind::Upload
                        | runmat_accelerate::placement::PlacementEventKind::Compile
                        | runmat_accelerate::placement::PlacementEventKind::Queue
                        | runmat_accelerate::placement::PlacementEventKind::Kernel
                )
            }));
        }
    });
}

#[test]
fn compiled_general_private_chain_completes_on_shared_cpu_with_one_receipt() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        let bytecode = compile_semantic("x = [0.25, 0.5]; y = exp(-x).*sin(4*x)+tanh(x).*tanh(x);");
        let (_, groups) = bytecode
            .reconciled_runtime_fusion_graph()
            .expect("checked private chain");
        assert_eq!(groups.iter().filter(|group| group.kind == FusionKind::ElementwiseChain && group.nodes.len() >= 8).count(), 1);
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let capture = capturing_fusion_placement(placement.clone());
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_placement(capture.clone()),
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime.clone(),
                None,
                None,
            ))
            .outcome
            .expect("ordinary VM shared CPU general chain");
        let output = completion
            .values
            .get(fusion_var_index(&bytecode, "y"))
            .expect("shared CPU output");
        let Value::Tensor(output) = &*output else {
            panic!("shared CPU output must remain a host tensor");
        };
        assert_eq!(output.shape(), [1, 2]);
        for (actual, x) in output.materialize_f64().iter().zip([0.25f64, 0.5]) {
            let expected = (-x).exp() * (4.0 * x).sin() + x.tanh() * x.tanh();
            assert!((actual - expected).abs() < 1e-12);
        }
        block_on(runtime.scope(async {}));
        assert_one_cpu_full_call(&capture);
        let report = runmat_accelerate::placement_report();
        let traces = report
            .traces
            .iter()
            .filter(|trace| trace.operation == "fusion.elementwise")
            .collect::<Vec<_>>();
        assert_eq!(
            traces.len(),
            1,
            "one completed chain, not singleton fallback execution"
        );
        let trace = traces[0];
        assert!(trace.events.iter().any(|event| event.kind
            == runmat_accelerate::placement::PlacementEventKind::Complete
            && event.variant
                == Some(runmat_accelerate::placement::PlacementVariant::SharedRuntime)
            && event.duration_ns.is_some()));
        assert!(!trace.events.iter().any(|event| matches!(
            event.kind,
            runmat_accelerate::placement::PlacementEventKind::Fallback
                | runmat_accelerate::placement::PlacementEventKind::Upload
                | runmat_accelerate::placement::PlacementEventKind::Queue
        )));
    });
}

fn assert_cpu_fusion_was_selected_without_completion(reason: &str) {
    use runmat_accelerate::placement::{PlacementEventKind, PlacementVariant};
    let report = runmat_accelerate::placement_report();
    assert!(
        report
            .traces
            .iter()
            .any(|trace| trace.operation == "fusion.elementwise"
                && trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Selected
                        && event.variant == Some(PlacementVariant::SharedRuntime)
                        && event.reason.as_deref() == Some(reason))
                && !trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Complete)),
        "expected an incomplete CPU-selected region: {:?}",
        report.traces
    );
}

#[test]
fn fusion_compiler_nested_expressions_and_branches_have_disjoint_regions() {
    gc_test_context(|| {
        for source in [
            "x = [0.25, 0.5]; y = sin(x) .* (x + cos(x));",
            "x = [0.25, 0.5]; for k = 1:2; y = sin(x) .* x + 2; end",
            "x = [0.25, 0.5]; if true; y = sin(x) .* x + 2; else; y = cos(x) .* x + 3; end",
        ] {
            let bytecode = compile_semantic(source);
            let groups = bytecode.runtime_fusion_groups();
            assert!(
                !groups.is_empty(),
                "fixture must produce semantic fusion regions"
            );
            for (index, left) in groups.iter().enumerate() {
                for right in &groups[index + 1..] {
                    assert!(
                        left.span.end < right.span.start || right.span.end < left.span.start,
                        "compiler-produced regions overlap: {:?} and {:?}",
                        left.span,
                        right.span
                    );
                }
            }
        }
    });
}

/// Delegates all placement and feedback to the real session. Cancellation is
/// requested only after the planner actually selects shared fusion execution;
/// no instruction, span, candidate, decision, or execution result is replaced.
struct CancelOnCpuFusionSelection {
    placement: Rc<runmat_accelerate::placement::PlacementSession>,
    cancellation: runmat_execution::cancellation::CancellationSource,
    selected: std::cell::Cell<bool>,
    vm_full_call: bool,
    observed: std::cell::RefCell<Vec<runmat_execution::PlacementFeedback>>,
}

impl runmat_runtime::context::RuntimePlacementService for CancelOnCpuFusionSelection {
    fn whole_work_vm_fusion_region_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        region: runmat_runtime::context::VmFusionRegionRegistration,
        liveness: std::sync::Weak<()>,
    ) -> Option<runmat_runtime::context::VmFusionSessionRegistration> {
        if !self.vm_full_call {
            return None;
        }
        runmat_runtime::context::RuntimePlacementService::whole_work_vm_fusion_region_session_revision(self.placement.as_ref(), program, region, liveness)
    }
    fn plan(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError> {
        self.plan_with_observation_permit(request)
            .map(|selected| selected.decision)
    }

    fn plan_with_observation_permit(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, RuntimeError> {
        let selection =
            runmat_runtime::context::RuntimePlacementService::plan_with_observation_permit(
                self.placement.as_ref(),
                request,
            )?;
        let decision = &selection.decision;
        if decision.signature.operation == "fusion.elementwise"
            && decision.selections.iter().any(|selection| {
                selection.kind == runmat_execution::ExecutionCandidateKind::SharedRuntime
                    && selection.candidate == "cpu.shared.fusion.elementwise"
            })
        {
            if self.vm_full_call {
                assert!(matches!(decision.signature.revision.authority, runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)), "cancel the actual installed FullCall route");
            }
            self.selected.set(true);
            self.cancellation
                .cancel(runmat_execution::CancellationReason::User);
        }
        Ok(selection)
    }

    fn observe(&self, feedback: runmat_execution::PlacementFeedback) -> Result<(), RuntimeError> {
        self.observed.borrow_mut().push(feedback.clone());
        runmat_runtime::context::RuntimePlacementService::observe(self.placement.as_ref(), feedback)
    }

    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation) {
        runmat_runtime::context::RuntimePlacementService::invalidate(
            self.placement.as_ref(),
            invalidation,
        );
    }
}

#[test]
fn missing_vm_full_call_right_cannot_inject_or_train_fusion_cancellation() {
    cancellation_after_cpu_fusion_selection(false, false);
    cancellation_after_cpu_fusion_selection(true, false);
}

#[test]
fn installed_full_call_cpu_cancellation_retires_without_positive_receipt() {
    cancellation_after_cpu_fusion_selection(false, true);
    cancellation_after_cpu_fusion_selection(true, true);
}

fn cancellation_after_cpu_fusion_selection(reject_provider: bool, vm_full_call: bool) {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        let provider = registered_provider().expect("test provider");
        provider
            .reject_fusion
            .store(reject_provider, Ordering::Relaxed);
        let _reject = reject_provider.then(|| RejectFusion(provider));
        // The same host fixture as the successful CPU witness. Injecting a
        // pending resident download instead would suspend before any fusion
        // trace, not inside an observed CPU region. This test proves runner
        // cancellation after selection, not provider-future drop or caught-error
        // coverage; ticket RAII drop remains separately unit-tested.
        let bytecode = compile_semantic("x = [0.25, 0.5]; y = sin(x) .* x + 2;");
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let service = Rc::new(CancelOnCpuFusionSelection {
            placement: placement.clone(),
            cancellation: cancellation.clone(),
            selected: std::cell::Cell::new(false),
            vm_full_call,
            observed: Default::default(),
        });
        let runtime = runmat_runtime::context::RuntimeContext::with_cancellation(
            Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        )
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default().with_placement(service.clone()),
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let outcome = block_on(runmat_vm::interpret_workspace(
            &bytecode,
            runmat_vm::InterpreterVars::from_owned(vars),
            None,
            runtime,
            None,
            None,
        ))
        .outcome;
        if !vm_full_call {
            let _ =
                outcome.expect("without an installed VM FullCall right, original instructions run");
            assert!(!service.selected.get(), "no synthetic fusion selection");
            assert!(service.observed.borrow().is_empty(), "no fusion receipt");
            assert!(placement.profile_snapshot().feedback.is_empty());
            return;
        }
        let error = outcome.expect_err("cancellation after installed CPU fusion selection");
        assert!(
            service.selected.get(),
            "real planner must select CPU fusion"
        );
        assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(
            service.observed.borrow().is_empty(),
            "unfinished VM FullCall must not manufacture a receipt"
        );
        assert!(placement.profile_snapshot().feedback.is_empty());
        assert_cpu_fusion_was_selected_without_completion(if reject_provider {
            "provider_operation_unsupported"
        } else {
            "complete_cost"
        });
    });
}
