use super::*;
use runmat_runtime::context::preparation::{
    AdmittedBuiltinPreparation, AdmittedSumReduction, BuiltinPreparationAdmission,
};
use runmat_types::{NumericClass, NumericDomain};

fn logical_request() -> AdmittedBuiltinPreparation {
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name("logical").unwrap();
    AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: runmat_builtins::BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: Some(1),
        },
        vec![Value::Tensor(Tensor::new(vec![1.; 4], vec![1, 4]).unwrap())],
    )
    .unwrap()
}

#[test]
fn unquoted_dispatcher_call_observes_only_after_exact_full_call_publication() {
    let session = Rc::new(crate::placement::PlacementSession::default());
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_placement(session.clone())
            .with_acceleration(Rc::new(Adapter(&UNQUOTED_PROVIDER, 0))),
    );
    let input = Value::Tensor(Tensor::new(vec![0., 2., -3., 0.], vec![1, 4]).unwrap());
    let args = [input];
    let pending =
        runmat_runtime::dispatcher::call_builtin_async_in_context(&context, "logical", &args);
    assert!(session.ordinary_feedback_for_test().is_empty());
    let output = futures::executor::block_on(context.scope(pending)).unwrap();
    let Value::LogicalArray(logical) = output.value() else {
        panic!("ordinary host logical output")
    };
    assert_eq!(logical.shape, [1, 4]);
    assert_eq!(logical.data.as_slice(), &[0, 1, 1, 0]);
    let snapshot = session.ordinary_feedback_for_test();
    assert_eq!(snapshot.len(), 1);
    assert!(session.profile_snapshot().feedback.is_empty());
    let sample = &snapshot[0];
    assert_eq!(
        sample.kind,
        runmat_execution::ExecutionCandidateKind::SharedRuntime
    );
    assert_eq!(sample.samples, 1);
    assert_eq!(sample.failures, 0);
    assert!(matches!(
        sample.signature.revision.authority,
        runmat_execution::PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
    ));
}

#[test]
fn scalar_provider_preparation_fallback_keeps_original_host_body_without_feedback() {
    struct ScalarFallbackAdapter {
        provider: &'static dyn AccelProvider,
        selected_fallbacks: Rc<Cell<usize>>,
    }
    impl RuntimeAccelerationService for ScalarFallbackAdapter {
        fn prepare_admitted(
            &self,
            context: RuntimeContext,
            request: AdmittedBuiltinPreparation,
        ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
            let provider = self.provider;
            let selected_fallbacks = self.selected_fallbacks.clone();
            Box::pin(async move {
                let auto = NativeAutoOffload::new(
                    provider,
                    ThresholdConfig {
                        // The tiny scalar's actual host work is made more
                        // expensive than this fixture's provider quote. This
                        // selects the legal provider candidate through the
                        // real whole-work owner before preparation falls back.
                        cpu_elem_per_elem: 1.0,
                        ..ThresholdConfig::default()
                    },
                );
                let prepared = super::super::admitted::prepare_async(&auto, &context, request)
                    .await
                    .map_err(|error| *error)?;
                assert!(
                    prepared
                        .observation
                        .as_ref()
                        .is_some_and(BuiltinCallObservation::shared_fallback_selected),
                    "provider selection must become the typed shared fallback route"
                );
                selected_fallbacks.set(selected_fallbacks.get() + 1);
                let (operands, guard) = prepared.into_execution();
                PreparedBuiltinOperands::new(operands, Some(Box::new(guard)))
            })
        }
    }

    let provider = super::admitted_owners::owner();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for (name, args, expected) in [
        (
            "int32",
            vec![Value::Num(19.0)],
            Value::Int(runmat_value::IntValue::I32(19)),
        ),
        (
            "uint8",
            vec![Value::Num(19.0)],
            Value::Int(runmat_value::IntValue::U8(19)),
        ),
        (
            "plus",
            vec![Value::Num(19.0), Value::Num(2.0)],
            Value::Num(21.0),
        ),
    ] {
        let session = Rc::new(crate::placement::PlacementSession::default());
        let selected_fallbacks = Rc::new(Cell::new(0));
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_placement(session.clone())
                .with_acceleration(Rc::new(ScalarFallbackAdapter {
                    provider,
                    selected_fallbacks: selected_fallbacks.clone(),
                })),
        );
        let output = futures::executor::block_on(context.scope(
            runmat_runtime::dispatcher::call_builtin_async_in_context(&context, name, &args),
        ))
        .unwrap();
        assert_eq!(output.value(), &expected, "{name}");
        assert_eq!(
            selected_fallbacks.get(),
            1,
            "{name} selected provider fallback"
        );
        assert!(
            session.ordinary_feedback_for_test().is_empty()
                && session.profile_snapshot().feedback.is_empty(),
            "{name} must not publish a successful provider or private-host sample"
        );
    }
}

fn conversion_like_request(name: &str) -> AdmittedBuiltinPreparation {
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
    AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: runmat_builtins::BuiltinPreparationFormId(2),
            signature: 1,
            input_count: 3,
            requested_outputs: Some(1),
        },
        vec![Value::Tensor(
            Tensor::new(vec![1.25, -0.0], vec![1, 2]).unwrap(),
        )],
    )
    .unwrap()
}

#[test]
fn runtime_owned_conversion_like_form_never_enters_native_placement_or_feedback() {
    for name in ["single", "double"] {
        for provider in [&PROVIDER as &dyn AccelProvider, &REJECTING_PROVIDER] {
            let probe = Rc::new(Probe::default());
            probe.provider.set(true);
            let context = runtime(&probe);
            let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
            let prepared = super::super::admitted::prepare_with(
                &auto,
                &context,
                conversion_like_request(name),
            )
            .unwrap();
            assert_eq!(prepared.args().len(), 1);
            assert!(matches!(&prepared.args()[0], Value::Tensor(value) if value.shape() == [1, 2]));
            assert!(probe.requests.borrow().is_empty());
            let result = futures::executor::block_on(prepared.execute(|mut operands| async move {
                Ok(rooted(
                    operands.pop().expect("unchanged conversion data operand"),
                ))
            }))
            .unwrap();
            assert!(matches!(result.value(), Value::Tensor(_)));
            assert!(probe.requests.borrow().is_empty());
            assert!(probe.feedback.borrow().is_empty());
        }
    }
}

fn sum_request(axes: Vec<usize>) -> AdmittedBuiltinPreparation {
    sum_request_for(runmat_types::SumInvocation::Sum, axes)
}

fn sum_request_for(
    invocation: runmat_types::SumInvocation,
    axes: Vec<usize>,
) -> AdmittedBuiltinPreparation {
    let entry = runmat_builtins::sum_invocation_entry(invocation);
    let runmat_builtins::BuiltinOperandPreparation::Forms([form]) = entry.placement.preparation
    else {
        panic!("one canonical sum preparation form")
    };
    let [signature] = form.signatures else {
        panic!("one canonical variadic signature")
    };
    AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: form.binding_variant,
            },
            form: form.id,
            signature: *signature,
            input_count: match invocation {
                runmat_types::SumInvocation::Sum => 4,
                runmat_types::SumInvocation::NanSum => 2,
            },
            requested_outputs: Some(1),
        },
        vec![Value::Tensor(Tensor::new(vec![1.; 4], vec![2, 2]).unwrap())],
    )
    .unwrap()
    .with_sum_reduction(AdmittedSumReduction {
        invocation,
        axes,
        missing: runmat_builtins::ReductionNaN::Omit,
        output: runmat_builtins::SumOutputClass::Double,
        input_class: NumericClass::Double,
        input_domain: NumericDomain::Real,
        accumulation_class: NumericClass::Double,
        publication_class: NumericClass::Double,
    })
    .unwrap()
}

fn native_integer_sum_request() -> AdmittedBuiltinPreparation {
    let invocation = runmat_types::SumInvocation::Sum;
    let entry = runmat_builtins::sum_invocation_entry(invocation);
    let runmat_builtins::BuiltinOperandPreparation::Forms([form]) = entry.placement.preparation
    else {
        panic!("one canonical sum preparation form")
    };
    let [signature] = form.signatures else {
        panic!("one canonical variadic signature")
    };
    AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: form.binding_variant,
            },
            form: form.id,
            signature: *signature,
            input_count: 2,
            requested_outputs: Some(1),
        },
        vec![Value::Int(runmat_value::IntValue::I8(100))],
    )
    .unwrap()
    .with_sum_reduction(AdmittedSumReduction {
        invocation,
        axes: vec![0],
        missing: runmat_builtins::ReductionNaN::Include,
        output: runmat_builtins::SumOutputClass::Native,
        input_class: NumericClass::Int8,
        input_domain: NumericDomain::Real,
        accumulation_class: NumericClass::Int8,
        publication_class: NumericClass::Int8,
    })
    .unwrap()
}

#[test]
fn native_integer_sum_rejected_provider_keeps_cpu_observation() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(&REJECTING_PROVIDER, ThresholdConfig::default());
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, native_integer_sum_request())
            .unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "sum");
    let result = futures::executor::block_on(
        prepared.execute(|_| async { Ok(rooted(Value::Int(runmat_value::IntValue::I8(100)))) }),
    );
    assert_eq!(
        result.unwrap().value(),
        &Value::Int(runmat_value::IntValue::I8(100))
    );
    let feedback = probe.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(feedback[0].succeeded);
    assert_eq!(feedback[0].candidate, "cpu.shared.sum");
}

#[test]
fn nansum_admitted_identity_and_abandonment_use_the_same_session_owner() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let invocation = runmat_types::SumInvocation::NanSum;
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, sum_request_for(invocation, vec![0]))
            .unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "nansum");
    assert!(probe.feedback.borrow().is_empty());
    drop(prepared.execute(|_| async { Ok(rooted(Value::Num(4.0))) }));
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn admitted_sum_uses_existing_session_only_for_real_reduction_work() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let identity =
        super::super::admitted::prepare_with(&auto, &context, sum_request(vec![])).unwrap();
    assert!(matches!(identity.args()[0], Value::Tensor(_)));
    assert!(probe.requests.borrow().is_empty());
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, sum_request(vec![0])).unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "sum");
    assert!(probe.feedback.borrow().is_empty());
    futures::executor::block_on(prepared.execute(|_| async { Ok(rooted(Value::Num(4.0))) }))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
}

#[test]
fn admitted_cpu_route_uses_existing_session_and_finishes_after_body() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
    assert_eq!(probe.requests.borrow().len(), 1);
    assert_eq!(probe.requests.borrow()[0].signature.operation, "logical");
    assert!(probe.feedback.borrow().is_empty());
    futures::executor::block_on(prepared.execute(|_| async { Ok(rooted(Value::Bool(true))) }))
        .unwrap();
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}

#[test]
fn rejected_provider_uses_cpu_only_session_and_observes_actual_body_once() {
    for succeeds in [true, false] {
        let probe = Rc::new(Probe::default());
        let context = runtime(&probe);
        let auto = NativeAutoOffload::new(&REJECTING_PROVIDER, ThresholdConfig::default());
        let prepared =
            super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
        let requests = probe.requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].signature.operation, "logical");
        let candidates = &requests[0].graph.nodes[0].candidates;
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].descriptor.kind,
            ExecutionCandidateKind::SharedRuntime
        );
        drop(requests);
        assert!(probe.feedback.borrow().is_empty());
        let result = futures::executor::block_on(prepared.execute(|_| async move {
            if succeeds {
                Ok(rooted(Value::Bool(true)))
            } else {
                Err(runmat_runtime::build_runtime_error("test host failure").build())
            }
        }));
        assert_eq!(result.is_ok(), succeeds);
        let feedback = probe.feedback.borrow();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].succeeded, succeeds);
    }
}

#[test]
fn rejected_provider_unpolled_body_aborts_once_without_success_observation() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let auto = NativeAutoOffload::new(&REJECTING_PROVIDER, ThresholdConfig::default());
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
    drop(prepared.execute(|_| async { Ok(rooted(Value::Bool(true))) }));
    let feedback = probe.feedback.borrow();
    assert!(feedback.is_empty());
}

#[test]
fn rejected_provider_cancelled_body_never_reports_cpu_success() {
    let probe = Rc::new(Probe::default());
    let source = runmat_execution::cancellation::CancellationSource::new();
    let context = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        source.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_placement(probe.clone()));
    let auto = NativeAutoOffload::new(&REJECTING_PROVIDER, ThresholdConfig::default());
    let prepared =
        super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
    source.cancel(runmat_execution::CancellationReason::User);
    let _ =
        futures::executor::block_on(prepared.execute(|_| async { Ok(rooted(Value::Bool(true))) }));
    let feedback = probe.feedback.borrow();
    assert!(feedback.is_empty());
}

#[test]
fn admitted_unpolled_future_and_panicking_body_abort_once() {
    for panic_body in [false, true] {
        let probe = Rc::new(Probe::default());
        let context = runtime(&probe);
        let auto = NativeAutoOffload::new(&PROVIDER, ThresholdConfig::default());
        let prepared =
            super::super::admitted::prepare_with(&auto, &context, logical_request()).unwrap();
        if panic_body {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                futures::executor::block_on(prepared.execute(|_| async {
                    panic!("admitted body failure");
                }))
            }));
            assert!(outcome.is_err());
        } else {
            drop(prepared.execute(|_| async { Ok(rooted(Value::Bool(true))) }));
        }
        assert!(probe.feedback.borrow().is_empty());
    }
}

#[test]
fn installation_preserves_injected_acceleration_and_placement_authority() {
    let probe = Rc::new(Probe::default());
    let context = install_preparation_service(runtime(&probe));
    let original = context.service_ports().acceleration().unwrap().clone();
    let placement = context.service_ports().placement().unwrap().clone();
    let context = install_preparation_service(context);
    assert!(Rc::ptr_eq(
        &original,
        context.service_ports().acceleration().unwrap()
    ));
    assert!(Rc::ptr_eq(
        &placement,
        context.service_ports().placement().unwrap()
    ));
}

#[test]
fn canonical_calls_are_not_speculatively_prepared_by_legacy_route() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let request = logical_request();
    let prepared = futures::executor::block_on(prepare_builtin_call(
        "logical",
        request.operands(),
        &context,
    ))
    .unwrap();
    assert_eq!(prepared.args(), request.operands());
    assert!(probe.requests.borrow().is_empty());
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn filter_and_filtfilt_canonical_calls_preserve_resident_signal_without_legacy_specs() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, HostNumericDataOwned,
        HostNumericTensorOwned,
    };

    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0; 40]),
            shape: vec![1, 40],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for name in ["filter", "filtfilt"] {
        let probe = Rc::new(Probe::default());
        let context = runtime(&probe);
        let args = [Value::GpuTensor(allocation.handle.clone())];
        let prepared = futures::executor::block_on(prepare_builtin_call(name, &args, &context))
            .expect("canonical runtime owns IIR operand preparation");
        let [Value::GpuTensor(retained)] = prepared.args() else {
            panic!("canonical call must preserve resident signal")
        };
        assert_eq!(retained.buffer_id(), allocation.handle.buffer_id());
        assert!(retained.allocation_lease().validate_live().is_ok());
        assert!(probe.requests.borrow().is_empty());
        assert!(probe.feedback.borrow().is_empty());
    }
    assert!(allocation.handle.allocation_lease().validate_live().is_ok());
}

#[test]
fn canonical_color_sinks_retain_resident_operands_for_domain_admission() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(
            vec![1, 1, 3],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let args = [Value::GpuTensor(allocation.handle)];
    for name in [
        "gray2rgb", "hsv2rgb", "lab2rgb", "rgb2gray", "rgb2hsv", "rgb2lab",
    ] {
        let prepared =
            futures::executor::block_on(prepare_builtin_call(name, &args, &context)).unwrap();
        assert_eq!(
            prepared.args(),
            &args,
            "{name} must retain resident admission input"
        );
        assert!(
            probe.requests.borrow().is_empty(),
            "{name} must not run legacy auto planning"
        );
        assert!(
            probe.feedback.borrow().is_empty(),
            "{name} must not report legacy feedback"
        );
    }
    let indexed_args = [
        args[0].clone(),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap()),
    ];
    let prepared =
        futures::executor::block_on(prepare_builtin_call("ind2rgb", &indexed_args, &context))
            .unwrap();
    assert_eq!(prepared.args(), &indexed_args);
    assert!(probe.requests.borrow().is_empty());
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn nansum_waits_for_its_typed_admission_instead_of_legacy_prepromotion() {
    let probe = Rc::new(Probe::default());
    let context = runtime(&probe);
    let operands = [Value::Tensor(Tensor::new(vec![1.; 4], vec![2, 2]).unwrap())];
    let prepared = futures::executor::block_on(prepare_builtin_call(
        runmat_builtins::sum_invocation_entry(runmat_types::SumInvocation::NanSum)
            .identity
            .name,
        &operands,
        &context,
    ))
    .unwrap();
    assert_eq!(prepared.args(), &operands);
    assert!(probe.requests.borrow().is_empty());
    assert!(probe.feedback.borrow().is_empty());
}

#[test]
fn admitted_upload_failure_keeps_acceleration_error_identity() {
    let probe = Rc::new(Probe::default());
    probe.provider.set(true);
    let provider = admitted_owners::owner_with_upload_failure(true);
    let auto = NativeAutoOffload::new(provider, ThresholdConfig::default());
    let error =
        match super::super::admitted::prepare_with(&auto, &runtime(&probe), logical_request()) {
            Err(error) => error,
            Ok(_) => panic!("provider rejects upload"),
        };
    assert_eq!(
        error.identifier(),
        Some("RunMat:AccelerationOperationFailed")
    );
    assert!(error.message().contains("prepare builtin args"));
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
}
