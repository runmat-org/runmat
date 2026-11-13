#![cfg(feature = "native-accel")]

#[path = "support/admitted_logarithms.rs"]
mod admitted_logarithms;
#[path = "support/mod.rs"]
mod support;

use futures::executor::block_on;
use runmat_accelerate::placement::{PlacementEventKind, PlacementVariant};
use runmat_execution::{
    Digest, ExecutionCandidateKind, PlacementDecision, PlacementFeedback, PlacementPlanRequest,
    WholeWorkDecision, WholeWorkPlacementRequestRef,
};
use runmat_runtime::{
    context::{
        RuntimeContext, RuntimePlacementFeedbackSource, RuntimePlacementService,
        RuntimeServicePorts, RuntimeWholeWorkSelection,
    },
    RuntimeError,
};
use runmat_value::Value;
use std::{cell::RefCell, rc::Rc};

struct Selection {
    session: runmat_accelerate::placement::PlacementSession,
    operation: &'static str,
    kind: ExecutionCandidateKind,
    requests: RefCell<Vec<WholeWorkDecision>>,
}

impl RuntimePlacementService for Selection {
    fn whole_work_runtime_binding_session_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        binding: runmat_builtins::BuiltinBindingIdentity,
        registration: Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        RuntimePlacementService::whole_work_runtime_binding_session_revision(
            &self.session,
            program,
            binding,
            registration,
        )
    }

    fn select_whole_work_borrowed(
        &self,
        request: WholeWorkPlacementRequestRef<'_>,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        // This fixture covers both real routes, not profitability policy. The
        // candidate must originate in this invocation's admitted offers; the
        // actual session still validates its resources and issues the permit.
        let candidates = if request.signature.operation == self.operation {
            std::slice::from_ref(
                request
                    .candidates
                    .iter()
                    .find(|candidate| candidate.kind == self.kind)
                    .expect("intended route must be offered by the actual operation"),
            )
        } else {
            request.candidates
        };
        let selected = RuntimePlacementService::select_whole_work_borrowed(
            &self.session,
            WholeWorkPlacementRequestRef {
                candidates,
                ..request
            },
        )?;
        assert_eq!(selected.decision.signature, request.signature.to_owned());
        assert!(candidates.iter().any(|candidate| {
            candidate.selected(request.signature.operation) == selected.decision.selected
        }));
        assert!(selected.observation_is_current());
        self.requests.borrow_mut().push(selected.decision.clone());
        Ok(selected)
    }

    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        RuntimePlacementService::plan(&self.session, request)
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe(&self.session, feedback)
    }
    fn observe_selected_whole_work(
        &self,
        selected: &RuntimeWholeWorkSelection,
        elapsed_ns: u64,
        outcome: runmat_execution::WholeWorkObservationOutcome,
    ) -> Result<(), RuntimeError> {
        RuntimePlacementService::observe_selected_whole_work(
            &self.session,
            selected,
            elapsed_ns,
            outcome,
        )
    }
    fn defer_feedback(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        RuntimePlacementService::defer_feedback(&self.session, source)
    }
    fn drain_feedback(&self) {
        RuntimePlacementService::drain_feedback(&self.session);
    }
    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.session, invalidation);
    }
}

fn context(
    operation: &'static str,
    kind: ExecutionCandidateKind,
) -> (RuntimeContext, Rc<Selection>) {
    let selection = Rc::new(Selection {
        session: Default::default(),
        operation,
        kind,
        requests: RefCell::new(vec![]),
    });
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(selection.clone()));
    (context, selection)
}

fn execute(
    bytecode: &runmat_vm::Bytecode,
    context: RuntimeContext,
) -> Result<runmat_runtime::execution::RootedValues, Box<RuntimeError>> {
    runmat_vm::reset_thread_state_for_tests();
    let variables = vec![];
    let runmat_vm::InterpreterOutcome::Completed(completion) =
        block_on(runmat_vm::interpret_workspace(
            bytecode,
            runmat_vm::InterpreterVars::from_owned(variables),
            None,
            context,
            None,
            None,
        ))
        .outcome
        .map_err(Box::new)?;
    completion
        .with_owned(|completion| {
            runmat_runtime::execution::RootedValues::new(
                completion
                    .values
                    .into_owned()
                    .expect("ordinary logical execution frame is owned"),
            )
        })
        .map_err(|error| Box::new(error.into_runtime_error()))
}

fn output<'a>(bytecode: &runmat_vm::Bytecode, values: &'a [Value]) -> &'a Value {
    let index = bytecode
        .var_names
        .iter()
        .find_map(|(index, name)| (name == "result").then_some(*index))
        .unwrap();
    &values[index]
}

fn retire_output_after_frame(
    values: runmat_runtime::execution::RootedValues,
    resident: Option<runmat_accelerate_api::GpuTensorHandle>,
) {
    drop(values);
    if let Some(handle) = resident {
        assert_eq!(
            handle.release().expect("completed output retirement"),
            runmat_accelerate_api::allocation::AllocationReleaseOutcome::Retired,
            "completed frame must retain no hidden allocation aliases"
        );
    }
}

fn assert_output(value: &Value, expected: &[u8]) {
    let gathered = block_on(runmat_runtime::gather_if_needed_async(value)).unwrap();
    match gathered {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, [2, 2]);
            assert_eq!(array.data, expected);
        }
        other => panic!("expected logical matrix, got {other:?}"),
    }
}

fn assert_execution(selection: &Selection, provider: bool, operands: usize) {
    let requests = selection.requests.borrow();
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.signature.operation == selection.operation)
            .count(),
        1
    );
    let report = runmat_accelerate::placement_report();
    let traces = report
        .traces
        .iter()
        .filter(|trace| trace.operation == selection.operation)
        .collect::<Vec<_>>();
    assert_eq!(traces.len(), 1, "{} trace", selection.operation);
    let trace = traces[0];
    let uploads = trace
        .events
        .iter()
        .filter(|event| event.kind == PlacementEventKind::Upload)
        .count();
    let decision = requests
        .iter()
        .find(|request| request.signature.operation == selection.operation)
        .unwrap();
    // Completed observations are drained by the owning session, not replayed
    // through this facade's observe method. Query the actual pricing evidence.
    let evidence = selection.session.observed_candidate_duration(
        &decision.signature,
        &decision.selected.candidate,
        decision.selected.kind,
    );
    if provider {
        assert_eq!(uploads, operands);
        assert!(trace
            .events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Queue
                && event.variant == Some(PlacementVariant::ProviderOperation)
                && event.reason.as_deref() == Some("provider_submission")));
        assert!(!trace
            .events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Complete));
        assert_eq!(
            evidence,
            runmat_execution::CandidateObservationEvidence::Unavailable(
                runmat_execution::CandidateCostUnavailable::NoObservations
            ),
            "{}: enqueue is not a measured device completion",
            selection.operation
        );
    } else {
        assert_eq!(uploads, 0);
        assert!(trace
            .events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Complete
                && event.variant == Some(PlacementVariant::SharedRuntime)));
        assert_eq!(
            evidence,
            runmat_execution::CandidateObservationEvidence::Unavailable(
                runmat_execution::CandidateCostUnavailable::InsufficientObservations {
                    successful: 1,
                    failed: 0
                }
            ),
            "{}: completed host route must retain its observation",
            selection.operation
        );
    }
}

#[test]
fn canonical_logical_forms_execute_through_vm_with_real_provider_feasibility() {
    runmat_gc::gc_test_context(|| {
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("in-process provider");
        let _provider_scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        // One long-lived in-process provider serves this entire test binary;
        // the automatic provider cache never changes identity between cases.
        runmat_accelerate::configure_auto_offload(runmat_accelerate::AutoOffloadOptions {
            enabled: true,
            calibrate: false,
            log_level: runmat_accelerate::AutoOffloadLogLevel::Trace,
        });
        let cases: [(&str, &[u8]); 10] = [
            ("logical", &[0, 1, 1, 0]),
            ("and", &[0, 0, 1, 0]),
            ("or", &[1, 1, 1, 1]),
            ("xor", &[1, 1, 0, 1]),
            ("eq", &[0, 0, 1, 0]),
            ("ne", &[1, 1, 0, 1]),
            ("ge", &[0, 0, 1, 1]),
            ("gt", &[0, 0, 0, 1]),
            ("le", &[1, 1, 1, 0]),
            ("lt", &[1, 1, 0, 0]),
        ];
        for (name, expected) in cases {
            let operands = if name == "logical" { "a" } else { "a, b" };
            // feval retains the language call boundary; fusion cannot replace
            // the individual canonical operation whose preparation we inspect.
            let source = format!("a = [0 2; -1 0]; b = [1 2; 0 -3]; name = '{name}'; result = feval(name, {operands});");
            let bytecode = support::compile_source(&source).unwrap();
            for gpu in [false, true] {
                runmat_accelerate::placement::reset();
                let kind = if gpu {
                    ExecutionCandidateKind::ProviderOperation
                } else {
                    ExecutionCandidateKind::SharedRuntime
                };
                let (context, selection) = context(name, kind);
                let values = execute(&bytecode, context).unwrap();
                let result = output(&bytecode, &values);
                assert_eq!(matches!(result, Value::GpuTensor(_)), gpu, "{name}");
                assert_output(result, expected);
                assert_execution(&selection, gpu, if name == "logical" { 1 } else { 2 });
                let resident = match result {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                };
                retire_output_after_frame(values, resident);
            }
        }
        assert_invalid_calls_before_preparation();
        assert_direct_call();
        assert_power_execution();
        admitted_logarithms::run();
    });
}

fn assert_power_execution() {
    // Keep the provider/cache lifetime shared with the logical witnesses above.
    // feval prevents fusion from replacing the admitted runtime boundary.
    let cases = [
        (
            "a = [-4 4]; b = [2 0.5];",
            vec![1, 2],
            false,
            vec![(16.0, 0.0), (2.0, 0.0)],
        ),
        (
            "a = [-4 4]; b = [2; 0.5];",
            vec![2, 2],
            true,
            vec![(16.0, 0.0), (0.0, 2.0), (16.0, 0.0), (2.0, 0.0)],
        ),
        (
            "a = [-1e-300 1 NaN]; b = [0.5 NaN 0];",
            vec![1, 3],
            true,
            vec![(0.0, 1e-150), (f64::NAN, 0.0), (f64::NAN, 0.0)],
        ),
    ];
    for extensions in [false, true] {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(extensions);
        for (source, shape, complex, expected) in &cases {
            let source = format!("{source} result = feval('power', a, b);");
            let bytecode = support::compile_source(&source).unwrap();
            for gpu in [false, true] {
                runmat_accelerate::placement::reset();
                let kind = if gpu {
                    ExecutionCandidateKind::ProviderOperation
                } else {
                    ExecutionCandidateKind::SharedRuntime
                };
                let (context, selection) = context("power", kind);
                let values = execute(&bytecode, context).unwrap();
                let result = output(&bytecode, &values);
                assert_eq!(matches!(result, Value::GpuTensor(_)), gpu);
                let gathered = block_on(runmat_runtime::gather_if_needed_async(result)).unwrap();
                let actual = match gathered {
                    Value::Tensor(tensor) if !complex => {
                        assert_eq!(tensor.shape(), shape);
                        tensor
                            .as_f64_slice()
                            .unwrap()
                            .iter()
                            .map(|&x| (x, 0.0))
                            .collect::<Vec<_>>()
                    }
                    Value::ComplexTensor(tensor) if *complex => {
                        assert_eq!(tensor.shape(), shape);
                        tensor
                            .as_f64_slice()
                            .unwrap()
                            .iter()
                            .map(|value| (value.0, value.1))
                            .collect::<Vec<_>>()
                    }
                    other => panic!("power representation mismatch: {other:?}"),
                };
                assert_eq!(actual.len(), expected.len());
                for ((re, im), &(expected_re, expected_im)) in actual.into_iter().zip(expected) {
                    assert_power_component(re, expected_re);
                    assert_power_component(im, expected_im);
                }
                assert_execution(&selection, gpu, 2);
                let resident = match result {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                };
                retire_output_after_frame(values, resident);
            }
        }
    }
}

fn assert_power_component(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else if expected == 0.0 {
        // Principal complex powers retain floating trigonometric residuals.
        assert!(actual.abs() < 1e-14, "{actual} versus {expected}");
    } else {
        assert!(
            (actual / expected - 1.0).abs() < 1e-12,
            "{actual} versus {expected}"
        );
    }
}

fn assert_invalid_calls_before_preparation() {
    for (name, source) in [
        ("logical", "a = [NaN 1]; result = feval('logical', a);"),
        ("logical", "a = [0 1]; result = feval('logical', a, 1);"),
        (
            "logical",
            "a = [0 1]; [result, extra] = feval('logical', a);",
        ),
        ("and", "a = [0 1]; result = feval('and', a);"),
        ("eq", "a = [0 1]; b = [1 2 3]; result = feval('eq', a, b);"),
        ("power", "a = [1 2]; result = feval('power', a);"),
        (
            "power",
            "a = [1 2]; b = [1 2 3]; result = feval('power', a, b);",
        ),
        (
            "power",
            "a = [1 2]; b = [2 3]; [result, extra] = feval('power', a, b);",
        ),
    ] {
        runmat_accelerate::placement::reset();
        let bytecode = support::compile_source(source).unwrap();
        let (context, selection) = context(name, ExecutionCandidateKind::ProviderOperation);
        assert!(execute(&bytecode, context).is_err(), "{name} invalid input");
        assert!(!selection
            .requests
            .borrow()
            .iter()
            .any(|request| request.signature.operation == name));
        assert!(!runmat_accelerate::placement_report()
            .traces
            .iter()
            .any(|trace| trace.operation == name
                && trace
                    .events
                    .iter()
                    .any(|event| event.kind == PlacementEventKind::Upload)));
    }
}

fn assert_direct_call() {
    runmat_accelerate::placement::reset();
    let bytecode = support::compile_source("a = [0 2; -1 0]; result = logical(a);").unwrap();
    let (context, selection) = context("logical", ExecutionCandidateKind::ProviderOperation);
    let values = execute(&bytecode, context).unwrap();
    let result = output(&bytecode, &values);
    assert_output(result, &[0, 1, 1, 0]);
    assert_execution(&selection, true, 1);
    let Value::GpuTensor(handle) = result else {
        panic!("direct logical did not retain provider result");
    };
    let resident = handle.clone();
    retire_output_after_frame(values, Some(resident));
}
