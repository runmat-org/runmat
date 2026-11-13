use super::*;
use crate::fusion::{
    FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, ImageNormalizePattern,
};
use crate::fusion_exec::*;
use crate::graph::{InstrSpan, ShapeInfo};
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation},
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use std::{collections::HashMap, rc::Rc};

struct RejectingProvider;

struct ShaderFeatureProvider {
    available: bool,
}

impl runmat_accelerate_api::AccelProvider for ShaderFeatureProvider {
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("shader feature selection must not upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("shader feature selection must not download")
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "shader-feature-selection-fixture".into()
    }

    fn supports_shader_requirements(
        &self,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
    ) -> bool {
        requirements == runmat_accelerate_api::ProviderShaderRequirements::Basic || self.available
    }

    fn query_feasibility(
        &self,
        _: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        runmat_accelerate_api::ProviderFeasibility::supported(
            runmat_accelerate_api::ProviderResourceEstimate {
                output_bytes: Some(8),
                transient_bytes: Some(0),
                dispatches: Some(1),
            },
        )
    }
}

struct NoTouchImageProvider;

impl runmat_accelerate_api::AccelProvider for NoTouchImageProvider {
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("wrong-rank image must not upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("wrong-rank image must not download")
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F32
    }

    fn device_info(&self) -> String {
        "no-touch-image-fixture".into()
    }

    fn query_feasibility(
        &self,
        _: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        panic!("wrong-rank image must not enter provider placement")
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_wrong_rank_resumes_runtime_before_placement_or_upload() {
    runmat_gc::gc_test_context(|| {
        static PROVIDER: NoTouchImageProvider = NoTouchImageProvider;
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
        let pattern = FusionPattern::ImageNormalize(ImageNormalizePattern {
            input: 0,
            epsilon: ImageScalar::Constant(0.0),
            gain: None,
            bias: None,
            gamma: None,
            clamp_zero: false,
            clamp_zero_value: None,
            bound_mean_axes: Vec::new(),
            requires_single_input: true,
            reduction_routes: None,
        });
        let mut plan = make_plan(Vec::new());
        plan.group.kind = FusionKind::ImageNormalize;
        plan.group.pattern = Some(pattern.clone());
        plan.group.shape = ShapeInfo::Unknown;
        plan.inputs = vec![0];
        plan.stack_pattern = vec![0];
        plan.kernel.kind = FusionKind::ImageNormalize;
        plan.pattern = Some(pattern);
        let runtime = context(Vec::new());

        for shape in [vec![2], vec![2, 2], vec![1, 2, 2, 1]] {
            let elements: usize = shape.iter().product();
            let request = FusionExecutionRequest {
                plan: &plan,
                inputs: vec![Value::Tensor(
                    runmat_value::Tensor::from_f32(vec![1.0_f32; elements], shape).unwrap(),
                )],
                placement: Some(crate::placement::PlacementCorrelationId(91)),
                runtime: Some(runtime.clone()),
            };
            assert!(AdmittedBindings::new(&request).is_some());
            assert!(matches!(
                futures::executor::block_on(execute_image_normalize_placed(request)).unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_bound_fold_missing_request_value_cannot_use_captured_constant() {
    runmat_gc::gc_test_context(|| {
        static PROVIDER: NoTouchImageProvider = NoTouchImageProvider;
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));
        let pattern = FusionPattern::ImageNormalize(ImageNormalizePattern {
            input: 0,
            epsilon: ImageScalar::Constant(0.0),
            gain: None,
            bias: None,
            gamma: None,
            clamp_zero: true,
            clamp_zero_value: Some(1),
            bound_mean_axes: Vec::new(),
            requires_single_input: true,
            reduction_routes: None,
        });
        let mut plan = make_plan(Vec::new());
        plan.group.kind = FusionKind::ImageNormalize;
        plan.group.pattern = Some(pattern.clone());
        plan.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(2), Some(2)]);
        plan.inputs = vec![0];
        plan.stack_pattern = vec![0];
        plan.kernel.kind = FusionKind::ImageNormalize;
        plan.pattern = Some(pattern);
        plan.const_values.insert(
            1,
            Value::Tensor(runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
        );
        let request = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::Tensor(
                runmat_value::Tensor::from_f32(vec![1.0; 4], vec![1, 2, 2]).unwrap(),
            )],
            placement: None,
            runtime: Some(context(Vec::new())),
        };
        assert!(AdmittedBindings::new(&request).is_some());
        assert!(matches!(
            futures::executor::block_on(execute_image_normalize_vm(request, None)).unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
    });
}

impl runmat_accelerate_api::AccelProvider for RejectingProvider {
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("rejected fusion must not upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("rejected fusion must not download")
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        runmat_accelerate_api::ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "rejecting-fusion-fixture".into()
    }

    fn query_feasibility(
        &self,
        _: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        runmat_accelerate_api::ProviderFeasibility::rejected(
            runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
            "fusion_fixture_rejected",
        )
    }
}

fn rejected_fusion(
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
) -> (FusionCpuObservation, Rc<crate::placement::PlacementSession>) {
    let session = Rc::new(crate::placement::PlacementSession::default());
    let bindings = runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin");
    let mut runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_builtin(Rc::new(Selected(bindings)))
            .with_placement(session.clone()),
    );
    if let Some(token) = cancellation {
        runtime = runtime.with_cancellation_token(token);
    }
    let plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
    let mut invocation = request(&plan, &runtime);
    invocation.placement = Some(crate::placement::PlacementCorrelationId(71));
    let admitted = AdmittedBindings::new(&invocation).expect("canonical selected binding");
    let outcome = select_fusion_provider(
        &invocation,
        &admitted,
        &RejectingProvider,
        &[vec![1, 1]],
        None,
    )
    .expect("correlated rejected provider resumes shared bytecode");
    let FusionExecutionOutcome::SharedRuntime(ticket) = outcome else {
        panic!("rejected provider must select the shared runtime")
    };
    (ticket, session)
}

#[test]
fn provider_rejection_cpu_region_reports_only_completed_work() {
    runmat_gc::gc_test_context(|| {
        let (ticket, session) = rejected_fusion(None);
        assert!(session.profile_snapshot().feedback.is_empty());
        ticket.complete();
        let feedback = session.profile_snapshot().feedback;
        assert_eq!(feedback.len(), 1);
        assert_eq!(
            feedback[0].kind,
            runmat_execution::ExecutionCandidateKind::SharedRuntime
        );
        assert_eq!(feedback[0].samples, 1);
        assert_eq!(feedback[0].failures, 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_shader_requirement_rejection_retains_shared_runtime_feedback() {
    runmat_gc::gc_test_context(|| {
        let session = Rc::new(crate::placement::PlacementSession::default());
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_builtin(Rc::new(Selected(
                    runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin"),
                )))
                .with_placement(session.clone()),
        );
        let plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
        let mut invocation = request(&plan, &runtime);
        invocation.placement = Some(crate::placement::PlacementCorrelationId(72));
        let admitted = AdmittedBindings::new(&invocation).expect("selected binding");
        let provider = ShaderFeatureProvider { available: false };
        let outcome = select_fusion_provider(
            &invocation,
            &admitted,
            &provider,
            &[vec![1, 1]],
            Some(runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords),
        )
        .expect("missing shader feature must select shared runtime");
        let FusionExecutionOutcome::SharedRuntime(ticket) = outcome else {
            panic!("missing shader feature must not select provider")
        };
        assert!(session.profile_snapshot().feedback.is_empty());
        ticket.complete();
        let feedback = session.profile_snapshot().feedback;
        assert_eq!(feedback.len(), 1);
        assert_eq!(
            feedback[0].kind,
            runmat_execution::ExecutionCandidateKind::SharedRuntime
        );
        assert_eq!(feedback[0].samples, 1);

        // A capable provider remains eligible for the exact same typed source.
        let provider = ShaderFeatureProvider { available: true };
        let direct = request(&plan, &runtime);
        let admitted = AdmittedBindings::new(&direct).expect("selected binding");
        assert!(matches!(
            select_fusion_provider(
                &direct,
                &admitted,
                &provider,
                &[vec![1, 1]],
                Some(runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords),
            )
            .expect("capable provider retains direct route"),
            FusionExecutionOutcome::Executed(_)
        ));
    });
}

#[test]
fn provider_rejection_cpu_region_failure_abandonment_and_cancel_are_not_successes() {
    runmat_gc::gc_test_context(|| {
        let (ticket, session) = rejected_fusion(None);
        let result: Result<(), &'static str> = {
            let _region_observation = ticket;
            Err("shared bytecode failed")
        };
        assert!(result.is_err());
        assert_eq!(session.profile_snapshot().feedback[0].samples, 0);
        assert_eq!(session.profile_snapshot().feedback[0].failures, 1);

        let (ticket, session) = rejected_fusion(None);
        let abandoned = async move {
            std::future::pending::<()>().await;
            ticket.complete();
        };
        drop(abandoned);
        assert_eq!(session.profile_snapshot().feedback[0].samples, 0);
        assert_eq!(session.profile_snapshot().feedback[0].failures, 1);

        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let (ticket, session) = rejected_fusion(Some(cancellation.token()));
        cancellation.cancel(runmat_execution::CancellationReason::User);
        ticket.complete();
        assert_eq!(session.profile_snapshot().feedback[0].samples, 0);
        assert_eq!(session.profile_snapshot().feedback[0].failures, 1);
    });
}

#[test]
fn direct_provider_fusion_rejection_keeps_caller_intent() {
    runmat_gc::gc_test_context(|| {
        let session = Rc::new(crate::placement::PlacementSession::default());
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_builtin(Rc::new(Selected(
                    runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin"),
                )))
                .with_placement(session.clone()),
        );
        let plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
        let invocation = request(&plan, &runtime);
        let admitted = AdmittedBindings::new(&invocation).expect("canonical selected binding");
        assert!(select_fusion_provider(
            &invocation,
            &admitted,
            &RejectingProvider,
            &[vec![1, 1]],
            None
        )
        .is_err());
        assert!(session.profile_snapshot().feedback.is_empty());
    });
}

#[test]
fn private_named_plus_binding_cannot_claim_canonical_mixed_add() {
    let plan = make_plan(vec![named("plus", FusionCallOrigin::named("plus"))]);
    let canonical = context(runmat_runtime::builtin::runtime_builtin_bindings_by_name(
        "plus",
    ));
    assert!(AdmittedBindings::new(&request(&plan, &canonical)).is_some());

    let mut overridden = runmat_runtime::builtin::runtime_builtin_bindings_by_name("plus");
    assert_eq!(overridden.len(), 1);
    overridden[0].implementation = replacement;
    let private = context(overridden);
    assert!(AdmittedBindings::new(&request(&plan, &private)).is_none());
}

struct Selected(Vec<RuntimeBuiltinBinding>);
impl RuntimeBuiltinService for Selected {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.0
            .iter()
            .filter(|binding| binding.identity.builtin.name == name)
            .copied()
            .collect()
    }
}
fn replacement(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        let sequence = runmat_value::ValueSequence::single(Value::Num(99.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}
fn context(bindings: Vec<RuntimeBuiltinBinding>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Selected(bindings))))
}
fn named(name: &str, origin: FusionCallOrigin) -> FusionOp {
    FusionOp::Builtin {
        name: name.into(),
        origin,
        inputs: vec![0, 1],
        output: Some(2),
    }
}
fn make_plan(operations: Vec<FusionOp>) -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Scalar,
            span: InstrSpan { start: 0, end: 4 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(2),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}

#[test]
fn only_unbound_constants_are_independent_preflight_leaves() {
    let mut plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
    plan.const_values
        .insert(0, Value::Int(runmat_value::IntValue::I64(-9)));
    plan.constants.insert(1, Value::Num(-8.0));
    assert_eq!(unbound_constant_values(&plan).count(), 0);
    let actual = [Value::Num(2.0), Value::Num(3.0)];
    assert!(reject_native_integer_fusion_values(
        &actual,
        &unbound_constant_values(&plan).collect::<Vec<_>>()
    )
    .is_ok());
    plan.const_values.insert(3, Value::Num(7.0));
    plan.constants
        .insert(1000, Value::Int(runmat_value::IntValue::I64(6)));
    assert_eq!(unbound_constant_values(&plan).count(), 2);
    assert!(reject_native_integer_fusion_values(
        &actual,
        &unbound_constant_values(&plan).collect::<Vec<_>>()
    )
    .is_err());
}

#[test]
fn selected_binding_closed_value_precheck_uses_the_invocation_leaf() {
    let mut plan = make_plan(Vec::new());
    plan.const_values.insert(
        0,
        Value::Object(runmat_value::ObjectInstance::new("StaleSourceObject")),
    );
    {
        let request = FusionExecutionRequest {
            plan: &plan,
            inputs: vec![Value::Num(2.0), Value::Num(3.0)],
            placement: None,
            runtime: None,
        };
        // This tests only the selected-binding precheck. The opaque structural
        // constant still fails the independent exact source-digest admission.
        assert!(AdmittedBindings::new(&request).is_some());
    }
    plan.const_values.insert(
        3,
        Value::Object(runmat_value::ObjectInstance::new("UnboundObject")),
    );
    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::Num(2.0), Value::Num(3.0)],
        placement: None,
        runtime: None,
    };
    assert!(AdmittedBindings::new(&request).is_none());
}

#[test]
fn image_scalar_control_uses_invocation_input_before_shadow_constant() {
    let mut plan = make_plan(Vec::new());
    plan.inputs = vec![0];
    plan.const_values.insert(0, Value::Num(-9.0));
    let request = FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::Num(3.0)],
        placement: None,
        runtime: None,
    };
    assert_eq!(
        resolve_image_scalar_value(
            &ImageScalar::Value(0),
            &plan,
            &request,
            FusionPlacementObserver::new(None),
        )
        .unwrap(),
        3.0
    );
}

fn request<'a>(plan: &'a FusionGroupPlan, runtime: &RuntimeContext) -> FusionExecutionRequest<'a> {
    FusionExecutionRequest {
        plan,
        inputs: vec![Value::Num(2.), Value::Num(3.)],
        placement: None,
        runtime: Some(runtime.clone()),
    }
}

#[test]
fn missing_provider_is_pre_effect_shared_runtime_only_for_host_requests() {
    runmat_gc::gc_test_context(|| {
        let runtime = context(runmat_runtime::builtin::runtime_builtin_bindings_by_name(
            "sin",
        ));
        let mut plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
        let mut invocation = request(&plan, &runtime);
        assert!(matches!(
            admit_provider_availability(&invocation, None).unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
        assert!(matches!(
            admit_provider_availability(&invocation, Some(&RejectingProvider)).unwrap(),
            FusionExecutionOutcome::Executed(_)
        ));

        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let explicit = allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let resident = Value::GpuTensor(explicit);
        invocation.inputs[0] =
            Value::Cell(runmat_value::CellArray::new(vec![resident.clone()], 1, 1).unwrap());
        assert!(admit_provider_availability(&invocation, None).is_err());
        assert_eq!(allocation.probe.retirements(), 0);
        drop(invocation);

        plan.const_values.insert(9, resident.clone());
        let invocation = request(&plan, &runtime);
        assert!(admit_provider_availability(&invocation, None).is_err());
        assert_eq!(allocation.probe.retirements(), 0);

        let mut indexed_plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
        indexed_plan.constants.insert(
            0,
            Value::Cell(runmat_value::CellArray::new(vec![resident.clone()], 1, 1).unwrap()),
        );
        let indexed = request(&indexed_plan, &runtime);
        assert!(matches!(
            admit_provider_availability(&indexed, None).unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
        drop(indexed);
        indexed_plan.constants.insert(
            9,
            Value::Cell(runmat_value::CellArray::new(vec![resident], 1, 1).unwrap()),
        );
        let unbound = request(&indexed_plan, &runtime);
        assert!(admit_provider_availability(&unbound, None).is_err());
        assert_eq!(allocation.probe.retirements(), 0);
    });
}

#[test]
fn selected_named_bindings_reject_replacements_absence_and_ambiguity() {
    runmat_gc::gc_test_context(|| {
        for (name, args) in [
            ("sin", vec![Value::Num(2.)]),
            ("mtimes", vec![Value::Num(2.), Value::Num(3.)]),
            ("power", vec![Value::Num(2.), Value::Num(3.)]),
        ] {
            let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
            assert_eq!(original.len(), 1, "{name}");
            let plan = make_plan(vec![named(name, FusionCallOrigin::named(name))]);
            assert!(AdmittedBindings::new(&request(&plan, &context(original.clone()))).is_some());
            let mut changed = original.clone();
            changed[0].implementation = replacement;
            let selected = context(changed);
            assert!(AdmittedBindings::new(&request(&plan, &selected)).is_none());
            // The retained route really reaches the selected implementation.
            let result = futures::executor::block_on(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &selected, name, &args, 1,
                ),
            )
            .unwrap();
            assert_eq!(result, Value::Num(99.));
            assert!(AdmittedBindings::new(&request(&plan, &context(vec![]))).is_none());
            assert!(AdmittedBindings::new(&request(
                &plan,
                &context(vec![original[0], original[0]])
            ))
            .is_none());
        }
    });
}

#[test]
fn primitive_matrix_syntax_is_not_a_named_service_call_and_identity_retains_origin() {
    runmat_gc::gc_test_context(|| {
        let name = "mtimes";
        let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        let original_context = context(original.clone());
        let mut overridden = original;
        overridden[0].implementation = replacement;
        let overridden = context(overridden);
        let primitive = make_plan(vec![named(
            name,
            FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
        )]);
        let named = make_plan(vec![named(name, FusionCallOrigin::named(name))]);
        let primitive_proof = AdmittedBindings::new(&request(&primitive, &overridden)).unwrap();
        assert!(AdmittedBindings::new(&request(&named, &overridden)).is_none());
        let named_proof = AdmittedBindings::new(&request(&named, &original_context)).unwrap();
        assert_ne!(
            primitive_proof.identity(None).unwrap(),
            named_proof.identity(None).unwrap()
        );
        let mut object = request(&primitive, &overridden);
        object.inputs[0] = Value::Object(runmat_value::ObjectInstance::new("OverloadedMatrix"));
        assert!(AdmittedBindings::new(&object).is_none());
        let forged = make_plan(vec![FusionOp::Builtin {
            name: "sin".into(),
            origin: FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
            inputs: vec![0, 1],
            output: Some(2),
        }]);
        assert!(AdmittedBindings::new(&request(&forged, &original_context)).is_none());
    });
}

#[test]
fn every_placed_family_returns_to_runtime_before_provider_or_domain_work() {
    runmat_gc::gc_test_context(|| {
        let mut selected = runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin");
        selected[0].implementation = replacement;
        let context = context(selected);
        let plan = make_plan(vec![named("sin", FusionCallOrigin::named("sin"))]);
        // An arbitrary specialized shape/pattern cannot authorize bypassing a
        // selected override. No provider is installed or needed for this gate.
        futures::executor::block_on(async {
            assert!(matches!(
                execute_elementwise_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_matmul_epilogue_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_power_step_normalize_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_explained_variance_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
            assert!(matches!(
                execute_image_normalize_placed(request(&plan, &context))
                    .await
                    .unwrap(),
                FusionExecutionOutcome::SharedRuntime(_)
            ));
        });
        assert!(matches!(
            execute_reduction_with_shape_placed(request(&plan, &context), 1, 1, &[1, 1], 64)
                .unwrap(),
            FusionExecutionOutcome::SharedRuntime(_)
        ));
        let legacy = make_plan(vec![named("sum", FusionCallOrigin::NamedUnresolved)]);
        assert!(AdmittedBindings::new(&request(&legacy, &context)).is_none());
        let composite = make_plan(vec![
            named(
                "mtimes",
                FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply),
            ),
            FusionOp::Builtin {
                name: "sin".into(),
                origin: FusionCallOrigin::named("sin"),
                inputs: vec![2],
                output: Some(3),
            },
        ]);
        assert!(AdmittedBindings::new(&request(&composite, &context)).is_none());
    });
}
