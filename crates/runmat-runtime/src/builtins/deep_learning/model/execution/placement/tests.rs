use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelProvider, DeepLearningForwardBufferBound,
    DeepLearningForwardCapability, DeepLearningForwardResources, DeepLearningForwardRoute,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorView,
};
use runmat_execution::{
    Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind, WholeWorkPlacementRequest,
};
use runmat_value::{CellArray, NumericDType, Tensor, Value};

use super::super::{execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout};
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{
    layers::{feature_input_layer_builtin, relu_layer_builtin},
    model::{dlnetwork_builtin, output},
};
use crate::context::{
    RuntimeContext, RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkSelection,
};
use crate::RuntimeError;

#[path = "tests/conv_cancellation.rs"]
mod conv_cancellation;

#[derive(Default)]
struct SelectedRoute {
    feedback: RefCell<Vec<PlacementFeedback>>,
    prefer_provider: bool,
}

impl RuntimePlacementService for SelectedRoute {
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        Some(PlacementRevision {
            program,
            catalog: Digest::sha256(b"test-catalog"),
            compiler: Digest::sha256(b"test-compiler"),
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(provider),
            policy: Digest::sha256(b"test-policy"),
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, RuntimeError> {
        request
            .validate()
            .expect("well-formed whole-network request");
        let selected = request
            .candidates
            .iter()
            .find(|candidate| candidate.kind.is_provider() == self.prefer_provider)
            .expect("requested route is legal");
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: selected.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
            observation_permit: None,
        })
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, RuntimeError> {
        panic!("whole-work test must not invoke numeric graph planning")
    }

    fn observe(&self, feedback: PlacementFeedback) -> Result<(), RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }

    fn invalidate(&self, _: PlacementInvalidation) {}
}

fn network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

fn runtime(service: Rc<SelectedRoute>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_placement(service))
}

fn assert_half_score(value: &Value) {
    let Value::Tensor(score) = value else {
        panic!("a host numerical prediction is required")
    };
    assert_eq!(score.shape(), &[1, 1]);
    assert_eq!(score.numeric_dtype(), NumericDType::F64);
    assert_eq!(score.materialize_f64(), vec![0.5]);
}

struct RacingReluProvider {
    owner: TestAllocationOwner,
    cancellation: runmat_execution::cancellation::CancellationSource,
    stage_calls: AtomicUsize,
}

impl AccelProvider for RacingReluProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("selected route must preserve native numeric input transfer")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let data = match host.data {
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
            _ => {
                return Err(
                    anyhow::anyhow!("the selected ReLU fixture requires floating input").into(),
                )
            }
        };
        Ok(self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data,
                shape: host.shape.to_vec(),
                storage: GpuTensorStorage::Real,
            })?
            .handle)
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("provider failure must not gather")
    }

    fn activation_relu<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.stage_calls.fetch_add(1, Ordering::SeqCst);
            self.cancellation
                .cancel(runmat_execution::CancellationReason::User);
            anyhow::bail!("provider ReLU failed after cancellation raced the result")
        })
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "racing-relu-placement-fixture".into()
    }

    fn deep_learning_forward_route_capability(
        &self,
        route: &DeepLearningForwardRoute<'_>,
    ) -> anyhow::Result<Option<DeepLearningForwardCapability>> {
        let geometry = route
            .analyze()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        Ok(Some(DeepLearningForwardCapability::Ready(
            DeepLearningForwardResources {
                output_bytes: geometry.output_bytes,
                largest_stage_value_bytes: geometry.largest_stage_value_bytes,
                known_normalization_scratch_bytes: 0,
                upload_bytes: 0,
                resident_bytes: 0,
                new_buffer_bytes_upper_bound: DeepLearningForwardBufferBound::Unavailable,
            },
        )))
    }
}

fn relu_network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, relu], 2, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn publication_completes_one_selected_host_observation() {
    let service = Rc::new(SelectedRoute::default());
    let runtime = runtime(service.clone());
    let network = network();
    let Value::Object(object) = network.value() else {
        panic!("network object expected")
    };
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let outcome = block_on(execute_network(NetworkExecutionRequest {
        runtime: Some(&runtime),
        network: object,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .unwrap();
    assert!(service.feedback.borrow().is_empty());
    let output = output::present(
        object,
        &input,
        outcome,
        output::OutputPresentation::Forward,
        false,
    )
    .unwrap();
    assert_eq!(output.len(), 1);
    assert_half_score(&output[0]);
    let feedback = service.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(feedback[0].succeeded);
    assert_eq!(
        feedback[0].kind,
        runmat_execution::ExecutionCandidateKind::SharedRuntime
    );
}

#[test]
fn registered_predict_uses_explicit_context_and_output_count_for_placement() {
    let service = Rc::new(SelectedRoute::default());
    let runtime = runtime(service.clone());
    let network_owner = network();
    let args = [
        network_owner.value().clone(),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    ];
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("predict");
    assert_eq!(bindings.len(), 1);
    let _ambient_outputs =
        crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let output = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(1)),
    ))
    .expect("canonical predict must use the invoking context and requested outputs");
    assert_eq!(
        output.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    let values = output.as_slice();
    assert_eq!(values.len(), 1);
    assert_half_score(&values[0]);
    let feedback = service.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert_eq!(feedback[0].signature.operation, "predict");
    assert!(feedback[0].succeeded);
    drop(feedback);

    let state_output = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(2)),
    ))
    .expect("dlnetwork prediction can publish its state as the second output");
    assert_eq!(
        state_output.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    assert_eq!(state_output.len(), 2);
    let feedback = service.feedback.borrow();
    assert_eq!(feedback.len(), 2);
    assert_ne!(
        feedback[0].signature.runtime_facts,
        feedback[1].signature.runtime_facts
    );
    drop(feedback);

    runtime.set_error_namespace("PredictInvocation");
    let source = runmat_execution::cancellation::CancellationSource::new();
    source.cancel(runmat_execution::CancellationReason::User);
    let cancelled = runtime.with_cancellation_token(source.token());
    let error = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(cancelled), Some(1)),
    ))
    .expect_err("the explicitly selected runtime must control cancellation");
    assert_eq!(error.identifier(), Some("PredictInvocation:Cancelled"));
    assert_eq!(service.feedback.borrow().len(), 2);
}

#[test]
fn abandoning_unpublished_result_releases_without_failure_feedback() {
    let service = Rc::new(SelectedRoute::default());
    let runtime = runtime(service.clone());
    let network = network();
    let Value::Object(object) = network.value() else {
        panic!("network object expected")
    };
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let outcome = block_on(execute_network(NetworkExecutionRequest {
        runtime: Some(&runtime),
        network: object,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .unwrap();
    drop(outcome);
    assert!(service.feedback.borrow().is_empty());
}

#[test]
fn cancellation_before_presentation_abandons_without_success_feedback() {
    let service = Rc::new(SelectedRoute::default());
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = runtime(service.clone()).with_cancellation_token(cancellation.token());
    runtime.set_error_namespace("PublicationInvocation");
    let network = network();
    let Value::Object(object) = network.value() else {
        panic!("network object expected")
    };
    let ((error, source_nodes), recorded_nodes) =
        crate::builtins::deep_learning::autodiff::with_test_tape(|| {
            let input = block_on(crate::builtins::deep_learning::training::dlarray_builtin(
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "CB".into(),
                )]),
            ))
            .expect("an annotated dlarray input is required to prove tape publication");
            let source_nodes = crate::builtins::deep_learning::autodiff::test_tape_node_count();
            assert!(
                source_nodes > 0,
                "the annotated input must own a source node"
            );
            let outcome = block_on(execute_network(NetworkExecutionRequest {
                runtime: Some(&runtime),
                network: object,
                input: &input,
                caller: NetworkCaller::Forward,
                output_layout: NetworkOutputLayout::Preserve,
                requested_outputs: Some(2),
            }))
            .unwrap();
            assert!(
                outcome.trace.is_some(),
                "this invocation has a recordable network trace"
            );
            cancellation.cancel(runmat_execution::CancellationReason::User);
            let error = output::present(
                object,
                &input,
                outcome,
                output::OutputPresentation::Forward,
                true,
            )
            .expect_err("a canceled invocation must not publish state or tape results");
            (error, source_nodes)
        });
    assert_eq!(error.identifier(), Some("PublicationInvocation:Cancelled"));
    assert_eq!(
        recorded_nodes, source_nodes,
        "cancellation must preserve only the input's pre-existing source node"
    );
    assert!(service.feedback.borrow().is_empty());
}

#[test]
fn uncancelled_annotated_network_publishes_tape_node() {
    let service = Rc::new(SelectedRoute::default());
    let runtime = runtime(service.clone());
    let network = network();
    let Value::Object(object) = network.value() else {
        panic!("network object expected")
    };
    let ((output, source_nodes, input_node, output_node), recorded_nodes) =
        crate::builtins::deep_learning::autodiff::with_test_tape(|| {
            let input = block_on(crate::builtins::deep_learning::training::dlarray_builtin(
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "CB".into(),
                )]),
            ))
            .expect("an annotated dlarray input is required to prove tape publication");
            let source_nodes = crate::builtins::deep_learning::autodiff::test_tape_node_count();
            let input_node = crate::builtins::deep_learning::autodiff::dlarray_node_id(&input)
                .expect("annotated input must own a tape node");
            let outcome = block_on(execute_network(NetworkExecutionRequest {
                runtime: Some(&runtime),
                network: object,
                input: &input,
                caller: NetworkCaller::Forward,
                output_layout: NetworkOutputLayout::Preserve,
                requested_outputs: Some(2),
            }))
            .unwrap();
            assert!(outcome.trace.is_some());
            let output = output::present(
                object,
                &input,
                outcome,
                output::OutputPresentation::Forward,
                true,
            )
            .expect("an uncancelled annotated input must publish its tape node");
            let output_node = crate::builtins::deep_learning::autodiff::dlarray_node_id(&output[0])
                .expect("the published prediction must own a network node");
            (output, source_nodes, input_node, output_node)
        });
    assert_eq!(output.len(), 2);
    assert!(
        recorded_nodes > source_nodes,
        "the positive control must record a network result beyond the input source"
    );
    assert_ne!(output_node, input_node);
    let feedback = service.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(feedback[0].succeeded);
}

#[test]
fn classified_operational_failure_is_observed_despite_later_cancellation() {
    let service = Rc::new(SelectedRoute::default());
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = runtime(service.clone()).with_cancellation_token(cancellation.token());
    let network = network();
    let Value::Object(object) = network.value() else {
        panic!("network object expected")
    };
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let outcome = block_on(execute_network(NetworkExecutionRequest {
        runtime: Some(&runtime),
        network: object,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .unwrap();
    let observation = outcome.placement.expect("selected route has observation");
    cancellation.cancel(runmat_execution::CancellationReason::User);
    observation.operational_failure();

    let feedback = service.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(!feedback[0].succeeded);
    assert_eq!(
        feedback[0].kind,
        runmat_execution::ExecutionCandidateKind::SharedRuntime
    );
}

#[test]
fn selected_provider_failure_is_observed_when_cancellation_races_the_stage() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static RacingReluProvider = Box::leak(Box::new(RacingReluProvider {
        owner: TestAllocationOwner::new(),
        cancellation: runmat_execution::cancellation::CancellationSource::new(),
        stage_calls: AtomicUsize::new(0),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let service = Rc::new(SelectedRoute {
        prefer_provider: true,
        ..SelectedRoute::default()
    });
    let runtime = runtime(service.clone()).with_cancellation_token(provider.cancellation.token());
    let network = relu_network();
    let Value::Object(object) = network.value() else {
        panic!("network object expected")
    };
    let input = Value::Tensor(Tensor::new(vec![1.0, -2.0], vec![1, 2]).unwrap());
    let error = block_on(execute_network(NetworkExecutionRequest {
        runtime: Some(&runtime),
        network: object,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .err()
    .expect("selected provider ReLU must fail");
    assert_ne!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(error
        .message
        .contains("provider ReLU failed after cancellation raced the result"));
    assert!(provider.cancellation.token().is_cancelled());
    assert_eq!(provider.stage_calls.load(Ordering::SeqCst), 1);
    let feedback = service.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(!feedback[0].succeeded);
    assert_eq!(
        feedback[0].kind,
        runmat_execution::ExecutionCandidateKind::ProviderGraph
    );
}
