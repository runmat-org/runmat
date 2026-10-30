//! End-to-end caller routes over the strongly typed fixture. These witnesses
//! qualify orchestration and ownership only; real numerical GPU execution is
//! owned by the WGPU product tests.
use super::*;
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_value::NumericStorage;
#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll};

fn request<'a>(
    plan: &'a FusionGroupPlan,
    input: Value,
    placement: Option<crate::placement::PlacementCorrelationId>,
    runtime: Option<RuntimeContext>,
) -> FusionExecutionRequest<'a> {
    FusionExecutionRequest {
        plan,
        inputs: vec![input],
        placement,
        runtime,
    }
}

fn execute(
    request: FusionExecutionRequest<'_>,
) -> ProviderExecutionResult<FusionExecutionOutcome<Value>> {
    futures::executor::block_on(execute_reduction_with_shape_placed_async(
        request,
        2,
        1,
        &[1, 1],
        8,
    ))
}

fn assert_output(
    provider: &TypedCallerProvider,
    outcome: FusionExecutionOutcome<Value>,
    expected: f64,
    element: NumericElementType,
) -> GpuTensorHandle {
    let FusionExecutionOutcome::Executed(Value::GpuTensor(output)) = outcome else {
        panic!("typed provider route must execute")
    };
    assert_eq!(output.shape(), [1, 1]);
    assert_eq!(output.physical().element_type(), element);
    let downloaded = futures::executor::block_on(provider.download_numeric(&output)).unwrap();
    assert_eq!(downloaded.shape, vec![1, 1]);
    let expected_data = match element {
        NumericElementType::F32 => HostNumericDataOwned::F32(vec![expected as f32]),
        NumericElementType::F64 => HostNumericDataOwned::F64(vec![expected]),
        _ => panic!("typed caller output must be floating real"),
    };
    assert_eq!(downloaded.data, expected_data);
    output
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn host_single_upload_preserves_source_class_with_double_result_control() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let plan = sum_plan_double();
        let input =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, 2.5]), vec![2, 1]).unwrap();
        let output = assert_output(
            provider,
            execute(request(&plan, Value::Tensor(input), None, None)).unwrap(),
            3.75,
            NumericElementType::F64,
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(
            *provider.uploaded_elements.lock().unwrap(),
            vec![NumericElementType::F32]
        );
        assert_eq!(provider.cast_proposals.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 1);
        assert_eq!(
            *provider.submitted_strategy.lock().unwrap(),
            Some(ReductionStrategy::SinglePass)
        );
        drop(output);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].retirements(), 1);
        assert_eq!(probes[1].retirements(), 1);
        drop(probes);

        // The same selected public mathematics requires Double accumulation
        // even when its input/output class is Single. A provider without that
        // arithmetic must not retry the request through a raw F32 legacy hook.
        for name in ["sum", "mean"] {
            let provider =
                TypedCallerProvider::registered_with_precision(1 << 20, ProviderPrecision::F32);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let mut plan = sum_plan();
            let FusionOp::Builtin {
                name: terminal,
                origin,
                ..
            } = &mut plan.operations[0]
            else {
                unreachable!()
            };
            *terminal = name.into();
            *origin = crate::graph::FusionCallOrigin::named(name);
            let input = provider
                .resident_input(vec![1.25, 2.5], NumericElementType::F32)
                .unwrap()
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let source = Value::GpuTensor(input.clone());
            let invocation = request(&plan, source.clone(), None, None);
            let bindings = AdmittedBindings::new(&invocation).unwrap();
            let declaration = bindings.reduction(0).unwrap();
            let controls = [Value::Num(1.0)];
            let prepared = futures::executor::block_on((declaration.prepare)(
                runmat_runtime::builtins::common::spec::FusionReductionControls {
                    values: &controls,
                    requested_outputs: 1,
                },
            ))
            .unwrap()
            .unwrap();
            let contract = prepared
                .resolve(&runmat_runtime::value_fact::value_fact(&source))
                .unwrap()
                .unwrap();
            assert_eq!(
                contract.accumulator_class,
                runmat_types::NumericClass::Double
            );
            assert_eq!(contract.axes, [0]);
            assert_eq!(contract.output_shape, [1, 1]);
            for synchronous in [false, true] {
                let request = request(&plan, source.clone(), None, None);
                let result = if synchronous {
                    execute_reduction_with_shape_placed(request, 2, 1, &[1, 1], 8)
                } else {
                    futures::executor::block_on(execute_reduction_with_shape_placed_async(
                        request,
                        2,
                        1,
                        &[1, 1],
                        8,
                    ))
                }
                .unwrap();
                assert!(matches!(result, FusionExecutionOutcome::SharedRuntime(_)));
            }
            assert_eq!(provider.configs.load(Ordering::SeqCst), 0);
            assert_eq!(provider.feasibility.load(Ordering::SeqCst), 0);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
            assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
            assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 0);
            drop(invocation);
            drop(source);
            drop(input);
            assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 1);
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires RUNMAT_ALLOW_PRECISION_DOWNCAST=1 in a dedicated test process"]
fn opted_host_double_casts_once_to_selected_single_upload() {
    assert_eq!(
        crate::precision::PrecisionPolicy::current(),
        crate::precision::PrecisionPolicy::AllowDoubleToSingle
    );
    runmat_gc::gc_test_context(|| {
        let provider =
            TypedCallerProvider::registered_with_precision(1 << 20, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let plan = min_plan();
        // The selected F32 minimum is 2^24, whereas an uncast F64 minimum is 2^24+1.
        let input = Tensor::new(vec![16_777_217.0, 16_777_218.0], vec![2, 1]).unwrap();
        let output = assert_output(
            provider,
            execute(request(&plan, Value::Tensor(input), None, None)).unwrap(),
            16_777_216.0,
            NumericElementType::F32,
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(
            *provider.uploaded_elements.lock().unwrap(),
            vec![NumericElementType::F32]
        );
        assert_eq!(provider.cast_proposals.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 1);
        assert_eq!(
            *provider.submitted_input_values.lock().unwrap(),
            Some(vec![16_777_216.0, 16_777_218.0]),
            "selected submission must consume F32-rounded host words"
        );
        drop(output);
        assert_eq!(
            provider
                .probes
                .lock()
                .unwrap()
                .iter()
                .map(RetirementProbe::retirements)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_double_borrows_without_upload_and_executes_two_pass() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        *provider.strategies.lock().unwrap() = vec![ReductionStrategy::TwoPass];
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .resident_input(vec![4.0, 5.0], NumericElementType::F64)
            .unwrap();
        let plan = sum_plan();
        let output = assert_output(
            provider,
            execute(request(&plan, Value::GpuTensor(source.clone()), None, None)).unwrap(),
            9.0,
            NumericElementType::F64,
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
        assert_eq!(
            *provider.submitted_strategy.lock().unwrap(),
            Some(ReductionStrategy::TwoPass)
        );
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].retirements(), 0, "borrowed source stays owned");
        assert_eq!(probes[1].retirements(), 0, "published output stays owned");
        drop(probes);
        drop(output);
        assert_eq!(provider.probes.lock().unwrap()[1].retirements(), 1);
        drop(source);
        assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 1);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn resident_single_borrows_without_implicit_upcast() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .resident_input(vec![2.0, 3.0], NumericElementType::F32)
            .unwrap();
        let plan = sum_plan();
        let output = assert_output(
            provider,
            execute(request(&plan, Value::GpuTensor(source.clone()), None, None)).unwrap(),
            5.0,
            NumericElementType::F32,
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_proposals.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 1);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].retirements(), 0);
        drop(probes);
        drop(output);
        drop(source);
        assert_eq!(
            provider
                .probes
                .lock()
                .unwrap()
                .iter()
                .map(RetirementProbe::retirements)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    });
}

// The precision policy is process-global Lazy state. Run this ignored test in
// a dedicated process with RUNMAT_ALLOW_PRECISION_DOWNCAST=1, never set it here.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires RUNMAT_ALLOW_PRECISION_DOWNCAST=1 in a dedicated test process"]
fn opted_resident_double_uses_owned_single_cast_recipe_before_submission() {
    assert_eq!(
        crate::precision::PrecisionPolicy::current(),
        crate::precision::PrecisionPolicy::AllowDoubleToSingle
    );
    runmat_gc::gc_test_context(|| {
        let provider =
            TypedCallerProvider::registered_with_precision(1 << 20, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .resident_input(vec![16_777_217.0, 16_777_218.0], NumericElementType::F64)
            .unwrap();
        let plan = min_plan();
        let output = assert_output(
            provider,
            execute(request(&plan, Value::GpuTensor(source.clone()), None, None)).unwrap(),
            16_777_216.0,
            NumericElementType::F32,
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_proposals.load(Ordering::SeqCst), 3);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 1);
        assert_eq!(
            *provider.submitted_input_values.lock().unwrap(),
            Some(vec![16_777_216.0, 16_777_218.0]),
            "selected submission must consume the ready F32 cast words"
        );
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 3);
        assert_eq!(probes[0].retirements(), 0, "source remains borrowed");
        assert_eq!(
            probes[1].retirements(),
            1,
            "prepared cast retires after submit"
        );
        assert_eq!(probes[2].retirements(), 0, "result remains published");
        drop(probes);
        drop(output);
        drop(source);
        assert_eq!(
            provider
                .probes
                .lock()
                .unwrap()
                .iter()
                .map(RetirementProbe::retirements)
                .collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires RUNMAT_ALLOW_PRECISION_DOWNCAST=1 in a dedicated test process"]
fn opted_unpolled_and_pending_cast_futures_preserve_owner_lifetimes_without_submission() {
    assert_eq!(
        crate::precision::PrecisionPolicy::current(),
        crate::precision::PrecisionPolicy::AllowDoubleToSingle
    );
    runmat_gc::gc_test_context(|| {
        let provider =
            TypedCallerProvider::registered_with_precision(1 << 20, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider
            .resident_input(vec![2.0, 3.0], NumericElementType::F64)
            .unwrap();
        let plan = min_plan();

        let never_polled = execute_reduction_with_shape_placed_async(
            request(&plan, Value::GpuTensor(source.clone()), None, None),
            2,
            1,
            &[1, 1],
            8,
        );
        drop(never_polled);
        assert_eq!(provider.configs.load(Ordering::SeqCst), 0);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        assert_eq!(provider.probes.lock().unwrap().len(), 1);
        assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 0);

        *provider.cast_execution_mode.lock().unwrap() = CastExecutionMode::PendingAfterAllocation;
        provider.estimated_execution_ns.store(1, Ordering::SeqCst);
        let session = Rc::new(crate::placement::PlacementSession::default());
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()))
        .with_cancellation_token(cancellation.token());
        let mut pending = Box::pin(execute_reduction_with_shape_placed_async(
            request(
                &plan,
                Value::GpuTensor(source.clone()),
                Some(crate::placement::PlacementCorrelationId(914)),
                Some(runtime),
            ),
            2,
            1,
            &[1, 1],
            8,
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(provider.cast_proposals.load(Ordering::SeqCst), 3);
        assert_eq!(provider.cast_executions.load(Ordering::SeqCst), 1);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        {
            let probes = provider.probes.lock().unwrap();
            assert_eq!(probes.len(), 2);
            assert_eq!(probes[0].retirements(), 0);
            assert_eq!(probes[1].retirements(), 0);
        }

        cancellation.cancel(runmat_execution::CancellationReason::User);
        drop(pending);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes[0].retirements(), 0, "borrowed source remains live");
        assert_eq!(
            probes[1].retirements(),
            1,
            "pending prepared output retires exactly once on abandonment"
        );
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        assert_eq!(*provider.submitted_input_values.lock().unwrap(), None);
        assert!(session
            .profile_snapshot()
            .feedback
            .iter()
            .all(|series| series.samples == 0));
        drop(probes);
        drop(source);
        assert_eq!(provider.probes.lock().unwrap()[0].retirements(), 1);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn canonical_session_can_select_cpu_without_running_preparation() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        provider
            .estimated_execution_ns
            .store(100_000, Ordering::SeqCst);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let session = Rc::new(crate::placement::PlacementSession::default());
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()));
        let plan = sum_plan();
        let outcome = execute(request(
            &plan,
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Some(crate::placement::PlacementCorrelationId(912)),
            Some(runtime),
        ))
        .unwrap();
        let FusionExecutionOutcome::SharedRuntime(observation) = outcome else {
            panic!("expensive provider candidate must lose to canonical CPU candidate")
        };
        assert_eq!(provider.configs.load(Ordering::SeqCst), 1);
        assert_eq!(provider.candidates.load(Ordering::SeqCst), 1);
        assert_eq!(provider.feasibility.load(Ordering::SeqCst), 1);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        observation.complete();
        assert_eq!(session.profile_snapshot().feedback.len(), 1);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelled_canonical_session_refuses_before_preparation_effects() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let session = Rc::new(crate::placement::PlacementSession::default());
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(session))
        .with_cancellation_token(cancellation.token());
        let plan = sum_plan();
        let error = execute(request(
            &plan,
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Some(crate::placement::PlacementCorrelationId(913)),
            Some(runtime),
        ))
        .err()
        .expect("cancelled placement must not begin selected preparation");
        assert!(!error.is_after_effect());
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while !matches!(
            cause.downcast_ref::<runmat_runtime::context::host_allocation::HostAllocationError>(),
            Some(runmat_runtime::context::host_allocation::HostAllocationError::Cancelled)
        ) {
            cause = cause
                .source()
                .expect("original cancellation cause retained");
        }
        // Canonical allocation-policy admission observes cancellation before
        // even querying provider configuration or preparation candidates.
        assert_eq!(provider.configs.load(Ordering::SeqCst), 0);
        assert_eq!(provider.candidates.load(Ordering::SeqCst), 0);
        assert_eq!(provider.feasibility.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        assert!(provider.probes.lock().unwrap().is_empty());
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_or_unsupported_submission_is_terminal_after_preparation() {
    for mode in [SubmissionMode::Unsupported, SubmissionMode::MalformedOutput] {
        for projected in [false, true] {
            runmat_gc::gc_test_context(|| {
                let provider = TypedCallerProvider::registered(1 << 20);
                *provider.submission_mode.lock().unwrap() = mode;
                let _selected = ThreadProviderGuard::set(Some(provider));
                let plan = if projected {
                    super::projection::power_plan()
                } else {
                    sum_plan()
                };
                let invocation = if projected {
                    FusionExecutionRequest {
                        plan: &plan,
                        inputs: vec![
                            Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                            Value::Num(2.0),
                        ],
                        placement: None,
                        runtime: None,
                    }
                } else {
                    host_request(&plan, vec![1.0, 2.0])
                };
                let error = execute(invocation)
                    .err()
                    .expect("selected malformed/unsupported submission must fail");
                assert!(error.is_after_effect());
                assert!(!error.is_unsupported(), "AfterEffect must prohibit replay");
                if mode == SubmissionMode::Unsupported {
                    let mut cause: &(dyn std::error::Error + 'static) = &error;
                    loop {
                        if cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
                            break;
                        }
                        cause = cause.source().expect("typed unsupported cause retained");
                    }
                }
                assert_eq!(provider.submissions.load(Ordering::SeqCst), 1);
                let probes = provider.probes.lock().unwrap();
                assert_eq!(probes[0].retirements(), 1);
                if mode == SubmissionMode::MalformedOutput {
                    assert_eq!(probes.len(), 2);
                    assert_eq!(probes[1].retirements(), 1, "malformed output retires");
                } else {
                    assert_eq!(probes.len(), 1);
                }
            });
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn provider_switch_during_upload_is_detected_before_submission() {
    runmat_gc::gc_test_context(|| {
        let provider = TypedCallerProvider::registered(1 << 20);
        let replacement = TypedCallerProvider::registered(1 << 20);
        *provider.switch_after_upload.lock().unwrap() = Some(replacement);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let plan = sum_plan();
        let error = execute(host_request(&plan, vec![1.0, 2.0]))
            .err()
            .expect("provider change across preparation await must be rejected");
        assert!(error.is_after_effect());
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.submissions.load(Ordering::SeqCst), 0);
        assert_eq!(replacement.submissions.load(Ordering::SeqCst), 0);
        let probes = provider.probes.lock().unwrap();
        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].retirements(), 1);
    });
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn typed_vm_provider_without_observation_receipt_cannot_mint_full_call_completion() {
    for projected in [false, true] {
        runmat_gc::gc_test_context(|| {
            let provider = TypedCallerProvider::registered(1 << 20);
            provider.estimated_execution_ns.store(1, Ordering::SeqCst);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let session = Rc::new(crate::placement::PlacementSession::default());
            let runtime = RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()));
            let plan = if projected {
                super::projection::power_plan()
            } else {
                sum_plan()
            };
            let semantic = runmat_runtime::context::VmFusionSemanticIdentity::from_validated_digest(
                crate::fusion::exact_source_plan_identity(&plan)
                    .expect("typed fixture has exact semantic identity"),
            );
            let installed = runmat_runtime::context::VmFusionInstalledRegion::install(
                &runtime,
                semantic,
                plan.group.id,
                plan.group.span.start,
                plan.group.span.end,
            )
            .expect("same-context VM region");
            let right = installed.right(&runtime).expect("same-context VM right");
            assert!(right.placement_revision().is_some());
            let input = provider
                .resident_input(
                    vec![1.0, 2.0],
                    if projected {
                        NumericElementType::F32
                    } else {
                        NumericElementType::F64
                    },
                )
                .unwrap();
            let mut invocation = request(
                &plan,
                Value::GpuTensor(input),
                Some(crate::placement::PlacementCorrelationId(915)),
                Some(runtime),
            );
            if projected {
                invocation.inputs.push(Value::Num(2.0));
            }
            let outcome = futures::executor::block_on(
                execute_reduction_with_shape_placed_async_vm(invocation, 2, 1, &[1, 1], 8, right),
            )
            .unwrap();
            let FusionExecutionOutcome::Executed(selected) = outcome else {
                panic!("selected typed provider route must execute")
            };
            assert!(matches!(selected.value, Value::GpuTensor(_)));
            assert_eq!(provider.submissions.load(Ordering::SeqCst), 1);
            assert!(
                selected.completion.is_none(),
                "default provider reports Unsupported observation, never an empty receipt"
            );
            assert!(session.profile_snapshot().feedback.is_empty());
        });
    }
}
