#![cfg(feature = "native-accel")]

#[path = "support/mod.rs"]
mod support;

use futures::executor::block_on;
use runmat_accelerate::placement::{PlacementEventKind, PlacementVariant};
use runmat_execution::{
    ExecutionCandidateKind, PlacementDecision, PlacementFeedback, PlacementPlanRequest,
    SelectedExecutionCandidate,
};
use runmat_runtime::{
    context::{RuntimeContext, RuntimePlacementService, RuntimeServicePorts},
    RuntimeError,
};
use runmat_value::Value;
use std::{cell::RefCell, rc::Rc};

struct Selection {
    operation: &'static str,
    kind: ExecutionCandidateKind,
    requests: RefCell<Vec<PlacementPlanRequest>>,
    feedback: RefCell<Vec<PlacementFeedback>>,
}

impl RuntimePlacementService for Selection {
    fn plan(&self, request: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        let kind = if request.signature.operation == self.operation {
            self.kind
        } else {
            ExecutionCandidateKind::SharedRuntime
        };
        let node = &request.graph.nodes[0];
        let candidate = &node
            .candidates
            .iter()
            .find(|candidate| candidate.descriptor.kind == kind)
            .unwrap_or_else(|| {
                panic!(
                    "{} lacks admitted {kind:?} candidate",
                    request.signature.operation
                )
            })
            .descriptor;
        let decision = PlacementDecision {
            signature: request.signature.clone(),
            selections: vec![SelectedExecutionCandidate {
                node: node.identity.clone(),
                candidate: candidate.identity.clone(),
                kind,
            }],
            predicted_total_ns: 1,
            from_cache: false,
            used_local_fallback: false,
            explored_states: node.candidates.len() as u32,
            pruned_states: 0,
        };
        self.requests.borrow_mut().push(request);
        Ok(decision)
    }
    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }
    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
}

fn context(
    operation: &'static str,
    kind: ExecutionCandidateKind,
) -> (RuntimeContext, Rc<Selection>) {
    let selection = Rc::new(Selection {
        operation,
        kind,
        requests: RefCell::new(vec![]),
        feedback: RefCell::new(vec![]),
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
) -> Result<Vec<Value>, Box<RuntimeError>> {
    runmat_vm::reset_thread_state_for_tests();
    let mut variables = vec![];
    let runmat_vm::InterpreterOutcome::Completed(completion) = block_on(
        runmat_vm::interpret_with_vars_in_context(bytecode, &mut variables, None, context),
    )
    .map_err(Box::new)?;
    Ok(completion.values)
}

fn output<'a>(bytecode: &runmat_vm::Bytecode, values: &'a [Value]) -> &'a Value {
    let index = bytecode
        .var_names
        .iter()
        .find_map(|(index, name)| (name == "result").then_some(*index))
        .unwrap();
    &values[index]
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
    let samples = selection.feedback.borrow();
    let samples = samples
        .iter()
        .filter(|sample| sample.signature.operation == selection.operation)
        .collect::<Vec<_>>();
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
        assert!(
            samples.is_empty(),
            "enqueue is not a measured device completion"
        );
    } else {
        assert_eq!(uploads, 0);
        assert!(trace
            .events
            .iter()
            .any(|event| event.kind == PlacementEventKind::Complete
                && event.variant == Some(PlacementVariant::SharedRuntime)));
        assert_eq!(samples.len(), 1);
        assert!(samples[0].succeeded);
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
            profile_path: None,
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
                if let Value::GpuTensor(handle) = result {
                    provider.free(handle).unwrap();
                    runmat_accelerate_api::clear_handle_metadata(handle);
                }
            }
        }
        assert_invalid_calls_before_preparation();
        assert_direct_call(provider);
    });
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

fn assert_direct_call(provider: &dyn runmat_accelerate_api::AccelProvider) {
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
    provider.free(handle).unwrap();
    runmat_accelerate_api::clear_handle_metadata(handle);
}
