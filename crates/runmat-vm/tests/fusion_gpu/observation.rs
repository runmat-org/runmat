//! Production-runner placement observation witnesses using the deterministic provider.
use super::*;

fn fusion_var_index(bytecode: &runmat_vm::Bytecode, name: &str) -> usize {
    bytecode
        .var_names
        .iter()
        .find_map(|(index, candidate)| (candidate == name).then_some(*index))
        .expect("named fixture variable")
}

fn resume_after_input_store(
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
                    | Instr::CreateMatrixDynamic(_)
                    | Instr::EnterScope(_)
            ),
            "unexpected input-prefix instruction: {instruction:?}"
        );
    }
    let pc = store + 1;
    let groups = bytecode.runtime_fusion_groups();
    assert!(groups.iter().any(|group| group.span.start >= pc));
    assert!(
        groups
            .iter()
            .all(|group| !(group.span.start < pc && pc <= group.span.end)),
        "resume boundary must not split a fusion region"
    );
    let mut vars = vec![None; bytecode.var_count];
    vars[slot] = Some(input);
    runmat_vm::InterpreterResumeState {
        pc,
        completion_boundary: None,
        vars,
        supplied_inputs: 0,
        requested_outputs: 0,
        missing_input_slots: Default::default(),
        global_aliases: Default::default(),
        persistent_aliases: Default::default(),
        side_effect_epoch: 0,
    }
}

fn resident_small_input() -> Value {
    let mut handle = PROVIDER
        .get()
        .expect("test provider")
        .push(vec![0.25, 0.5], vec![1, 2]);
    // Model an automatically resident input at a real resume boundary, not an
    // explicit gpuArray whose placement intent may not be silently discarded.
    runmat_accelerate_api::mark_handle_automatic(&mut handle);
    Value::GpuTensor(handle)
}

fn assert_fusion_numeric_output(bytecode: &runmat_vm::Bytecode, vars: &[Value], inputs: &[f64]) {
    let value = gather_if_needed(&vars[fusion_var_index(bytecode, "y")]).expect("gather result");
    let Value::Tensor(tensor) = value else {
        panic!("expected numeric tensor, got {value:?}");
    };
    assert_eq!(tensor.shape, vec![1, inputs.len()]);
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
        let vars = completion.values;
        assert_fusion_numeric_output(
            &bytecode,
            &vars,
            &(1..=4096).map(f64::from).collect::<Vec<_>>(),
        );
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
            runmat_accelerate::placement::PlacementEventKind::Compile,
            runmat_accelerate::placement::PlacementEventKind::Queue,
            runmat_accelerate::placement::PlacementEventKind::Kernel,
        ] {
            assert!(kinds.contains(&required), "missing {required:?}: {kinds:?}");
        }
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
        let mut vars = vec![Value::Num(0.0); bytecode.var_count];
        block_on(runmat_vm::interpret_with_vars_in_context(
            &bytecode, &mut vars, None, runtime,
        ))
        .expect("session-owned fusion execution");
        assert_fusion_numeric_output(
            &bytecode,
            &vars,
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
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_placement(placement.clone()),
        );
        let mut vars = vec![Value::Num(0.0); bytecode.var_count];
        block_on(runmat_vm::interpret_with_vars_in_context(
            &bytecode, &mut vars, None, runtime,
        ))
        .expect("shared-runtime selection");
        assert_fusion_numeric_output(&bytecode, &vars, &[0.25, 0.5]);
        let profile = placement.profile_snapshot();
        let cpu = profile
            .feedback
            .iter()
            .find(|series| series.candidate == "cpu.shared.fusion.elementwise")
            .expect("completed CPU-region feedback");
        assert_eq!(cpu.samples, 1);
        assert_eq!(cpu.failures, 0);

        let report = runmat_accelerate::placement_report();
        let trace = report
            .traces
            .iter()
            .find(|trace| trace.operation == "fusion.elementwise")
            .expect("elementwise fusion placement trace");
        assert!(trace.events.iter().any(|event| {
            event.kind == runmat_accelerate::placement::PlacementEventKind::Selected
                && event.variant
                    == Some(runmat_accelerate::placement::PlacementVariant::SharedRuntime)
                && event.reason.as_deref() == Some("complete_cost")
        }));
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
    });
}

struct ScopedDownloadMode(&'static TestProvider);

impl ScopedDownloadMode {
    fn new(mode: DownloadMode) -> Self {
        let provider = PROVIDER.get().expect("registered test provider");
        provider.download_started.store(false, Ordering::SeqCst);
        *provider
            .download_test_mode
            .lock()
            .expect("download mode lock") = mode;
        Self(provider)
    }
}

impl Drop for ScopedDownloadMode {
    fn drop(&mut self) {
        *self
            .0
            .download_test_mode
            .lock()
            .expect("download mode lock") = DownloadMode::Normal;
        self.0.download_started.store(false, Ordering::SeqCst);
    }
}

fn assert_cpu_fusion_was_selected_without_completion() {
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
                        && event.reason.as_deref() == Some("complete_cost"))
                && !trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Complete)),
        "expected an incomplete CPU-selected region: {:?}",
        report.traces
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

fn assert_cpu_region_failed_without_success(
    placement: &runmat_accelerate::placement::PlacementSession,
) {
    let profile = placement.profile_snapshot();
    let cpu = profile
        .feedback
        .iter()
        .find(|series| series.candidate == "cpu.shared.fusion.elementwise")
        .expect("CPU observation failure");
    assert_eq!(cpu.samples, 0);
    assert_eq!(cpu.failures, 1);
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

#[test]
fn fusion_cpu_region_polled_drop_records_failure_not_completion() {
    gc_test_context(|| {
        use std::future::Future;
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        // This is an ordinary compiler-produced region. A tiny resident input
        // selects shared bytecode; unsupported unary_sin then downloads it.
        let bytecode = compile_semantic("x = [0.25, 0.5]; y = sin(x) .* x + 2;");
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let runtime = fusion_test_runtime(placement.clone());
        let resume = resume_after_input_store(&bytecode, resident_small_input());
        let mode = ScopedDownloadMode::new(DownloadMode::Pending);
        let mut future = Box::pin(runmat_vm::interpret_resume_in_context(
            &bytecode, resume, None, runtime,
        ));
        let waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert!(mode.0.download_started.load(Ordering::SeqCst));
        assert_cpu_fusion_was_selected_without_completion();
        assert!(!placement
            .profile_snapshot()
            .feedback
            .iter()
            .any(|series| series.candidate == "cpu.shared.fusion.elementwise"));
        drop(future);
        assert_cpu_region_failed_without_success(&placement);
        assert_cpu_fusion_was_selected_without_completion();
    });
}

#[test]
fn fusion_cpu_region_caught_error_records_failure_not_completion() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        runmat_accelerate::placement::reset();
        let bytecode = compile_semantic(
            "x = [0.25, 0.5]; caught = false; try; y = sin(x) .* x + 2; catch; caught = true; end",
        );
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let resume = resume_after_input_store(&bytecode, resident_small_input());
        let mode = ScopedDownloadMode::new(DownloadMode::Fail);
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_resume_in_context(
                &bytecode,
                resume,
                None,
                fusion_test_runtime(placement.clone()),
            ))
            .expect("handler resumes actual bytecode");
        assert!(mode.0.download_started.load(Ordering::SeqCst));
        assert_eq!(
            completion.values[fusion_var_index(&bytecode, "caught")],
            Value::Bool(true)
        );
        assert_cpu_region_failed_without_success(&placement);
        assert_cpu_fusion_was_selected_without_completion();
    });
}
