use super::*;
use runmat_runtime::context::{RuntimeAccelerationService, RuntimeContext};
use runmat_runtime::context::{RuntimePlacementService, RuntimeServicePorts};
use runmat_runtime::numeric_region::service::{
    NativeNumericRegionOutcome, NativeNumericRegionRequest,
};
use runmat_runtime::numeric_region::{
    NumericBinaryOperation, NumericRegionNode, NumericRegionProgram,
};
use std::cell::RefCell;

#[derive(Default)]
struct GraphProbe {
    session: crate::placement::PlacementSession,
    kinds: RefCell<Vec<ExecutionCandidateKind>>,
}
impl RuntimePlacementService for GraphProbe {
    fn select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<runmat_runtime::context::RuntimeWholeWorkSelection, runmat_runtime::RuntimeError>
    {
        *self.kinds.borrow_mut() = request
            .candidates
            .iter()
            .map(|candidate| candidate.kind)
            .collect();
        RuntimePlacementService::select_whole_work(&self.session, request)
    }
    fn plan(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, runmat_runtime::RuntimeError> {
        RuntimePlacementService::plan(&self.session, request)
    }
    fn plan_with_observation_permit(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_runtime::context::RuntimeGraphPlacementSelection, runmat_runtime::RuntimeError>
    {
        *self.kinds.borrow_mut() = request.graph.nodes[0]
            .candidates
            .iter()
            .map(|candidate| candidate.descriptor.kind)
            .collect();
        RuntimePlacementService::plan_with_observation_permit(&self.session, request)
    }
    fn observe(
        &self,
        feedback: runmat_execution::PlacementFeedback,
    ) -> Result<(), runmat_runtime::RuntimeError> {
        RuntimePlacementService::observe(&self.session, feedback)
    }
    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation) {
        RuntimePlacementService::invalidate(&self.session, invalidation)
    }
}

fn runtime(probe: &Rc<GraphProbe>) -> RuntimeContext {
    // Preserve the existing producer's precise structural decline diagnostic
    // on an unexpected Native outcome; route assertions below remain strict.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("off"))
        .is_test(true)
        .try_init();
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_placement(probe.clone()))
}

fn assert_provider_graph(probe: &GraphProbe) {
    let kinds = probe.kinds.borrow();
    // This fixture has no physical WGPU queue-tail capability; the original
    // provider-delegating source route cannot mint completion observations.
    assert!(!kinds.contains(&ExecutionCandidateKind::SpecializedNativeCpu));
    // These physical lifecycle cases use broadcast shapes unsupported by
    // the equal-shape vectorized evaluator, not an invented CPU price.
    assert!(!kinds.contains(&ExecutionCandidateKind::VectorizedNativeCpu));
    assert!(kinds.contains(&ExecutionCandidateKind::ProviderFusion));
    assert!(!kinds.contains(&ExecutionCandidateKind::SharedRuntime));
    assert!(!kinds.contains(&ExecutionCandidateKind::OriginalNativeRegion));
}

fn request(runtime: &RuntimeContext, source_host: bool) -> NativeNumericRegionRequest {
    let left = Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
    let right = Value::Tensor(runmat_value::Tensor::new(vec![3.0, 4.0], vec![2]).unwrap());
    request_values(runtime, &left, &right, source_host)
}

fn provider_broadcast_request(runtime: &RuntimeContext) -> NativeNumericRegionRequest {
    let left = Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let right = Value::Tensor(runmat_value::Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap());
    assert!(runmat_runtime::numeric_region::workload(&[&left, &right]).is_none());
    request_values(runtime, &left, &right, false)
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_adapter_reaches_existing_provider_and_does_not_replay_late_upload_failure() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::new(2);
        provider.native_quote_ns.store(1, Ordering::SeqCst);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let probe = Rc::new(GraphProbe::default());
        let runtime = runtime(&probe);
        let request = provider_broadcast_request(&runtime);
        let error = match futures::executor::block_on(
            crate::native_auto::AdmittedAcceleration
                .execute_numeric_region(runtime, request)
                .unwrap(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("late upload failure must not become a native CPU decline"),
        };
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        assert!(std::error::Error::source(&error).is_some());
        assert_provider_graph(&probe);
        for probe in provider.probes.lock().unwrap().iter() {
            assert_eq!(probe.retirements(), 1);
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_adapter_pending_drop_retires_staged_inputs_and_unpolled_drop_does_no_work() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::new(usize::MAX);
        provider.native_quote_ns.store(1, Ordering::SeqCst);
        provider.suspend_dispatch.store(true, Ordering::SeqCst);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let probe = Rc::new(GraphProbe::default());
        let runtime = runtime(&probe);
        let unpolled = crate::native_auto::AdmittedAcceleration
            .execute_numeric_region(runtime.clone(), provider_broadcast_request(&runtime))
            .unwrap();
        drop(unpolled);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        let mut future = crate::native_auto::AdmittedAcceleration
            .execute_numeric_region(runtime.clone(), provider_broadcast_request(&runtime))
            .unwrap();
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        assert_provider_graph(&probe);
        drop(future);
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        for probe in provider.probes.lock().unwrap().iter() {
            assert_eq!(probe.retirements(), 1);
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_cpu_wins_same_region_comparison_without_provider_work() {
    runmat_gc::gc_test_context(|| {
        for (quote_outputs, source_host) in [(true, true), (false, true), (true, false)] {
            let provider = FailingProvider::new(usize::MAX);
            provider
                .quote_outputs
                .store(quote_outputs, Ordering::SeqCst);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let probe = Rc::new(GraphProbe::default());
            let runtime = runtime(&probe);
            let outcome = futures::executor::block_on(
                crate::native_auto::AdmittedAcceleration
                    .execute_numeric_region(runtime.clone(), request(&runtime, source_host))
                    .unwrap(),
            )
            .unwrap();
            let NativeNumericRegionOutcome::NativeCpu { kind, host, .. } = outcome else {
                panic!("legal Native CPU selection without provider work");
            };
            if !source_host {
                // This real public capture lacks source-host operation sites.
                // Its independently quoted vectorized execution is therefore
                // the only complete host alternative and must actually win.
                assert_eq!(
                    kind,
                    runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Vectorized
                );
                assert!(host.is_none());
            } else if kind
                == runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic
            {
                assert!(
                    host.is_some(),
                    "explored source-host route retains its real owner"
                );
            }
            let kinds = probe.kinds.borrow();
            assert!(kinds.contains(&ExecutionCandidateKind::VectorizedNativeCpu));
            assert_eq!(
                kinds.contains(&ExecutionCandidateKind::SpecializedNativeCpu),
                source_host
            );
            assert_eq!(
                kinds.contains(&ExecutionCandidateKind::ProviderFusion),
                quote_outputs
            );
            assert!(!kinds.contains(&ExecutionCandidateKind::OriginalNativeRegion));
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        }
    });
}

fn request_values(
    runtime: &RuntimeContext,
    left: &Value,
    right: &Value,
    source_host: bool,
) -> NativeNumericRegionRequest {
    let program = NumericRegionProgram {
        nodes: vec![
            NumericRegionNode::Input(0),
            NumericRegionNode::Input(1),
            NumericRegionNode::Binary {
                operation: NumericBinaryOperation::Add,
                left: 0,
                right: 1,
            },
        ],
        outputs: vec![2],
    };
    let request = NativeNumericRegionRequest::capture_verified(
        runtime,
        runmat_types::RegionId {
            function: runmat_types::ProgramFunctionId(0),
            ordinal: 0,
        },
        runmat_runtime::native::NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: runmat_runtime::native::NativeSitePhase::RVALUE,
            ordinal: 0,
            reserved: 0,
        },
        &program,
        &[left, right],
    )
    .unwrap()
    .unwrap();
    if source_host {
        request.with_native_operation_sites([site()]).unwrap()
    } else {
        request
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_single_broadcast_reaches_existing_typed_provider_without_f64_cpu_conversion() {
    runmat_gc::gc_test_context(|| {
        // Exercise the explicit Single ABI on an F64-default provider, rather
        // than the already-valid default F32 compatibility dispatch.
        let provider = FailingProvider::with_abi(usize::MAX, ProviderPrecision::F64, true);
        provider.native_quote_ns.store(1, Ordering::SeqCst);
        provider.successful_fusion.store(true, Ordering::SeqCst);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let probe = Rc::new(GraphProbe::default());
        let runtime = runtime(&probe);
        let left =
            Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let right =
            Value::Tensor(runmat_value::Tensor::from_f32(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap());
        let outcome = futures::executor::block_on(
            crate::native_auto::AdmittedAcceleration
                .execute_numeric_region(
                    runtime.clone(),
                    request_values(&runtime, &left, &right, false),
                )
                .unwrap(),
        )
        .unwrap();
        let NativeNumericRegionOutcome::Completed { outputs, .. } = outcome else {
            panic!("real Single broadcast must reach its typed producer")
        };
        let Value::GpuTensor(output) = &outputs.values()[0] else {
            panic!("provider output")
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(
            output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        assert!(!probe
            .kinds
            .borrow()
            .contains(&ExecutionCandidateKind::VectorizedNativeCpu));
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        assert_eq!(provider.typed_dispatches.load(Ordering::SeqCst), 1);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_explicit_resident_floating_inputs_keep_owner_class_intent_and_do_not_upload_or_gather() {
    runmat_gc::gc_test_context(|| {
        for (precision, element) in [
            (
                ProviderPrecision::F32,
                runmat_accelerate_api::NumericElementType::F32,
            ),
            (
                ProviderPrecision::F64,
                runmat_accelerate_api::NumericElementType::F64,
            ),
        ] {
            let provider = FailingProvider::with_abi(usize::MAX, precision, true);
            provider.native_quote_ns.store(1, Ordering::SeqCst);
            provider.successful_fusion.store(true, Ordering::SeqCst);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let allocation = provider
                .allocations
                .allocate_zeroed(
                    vec![2, 1],
                    element,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            let handle = allocation
                .handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .unwrap();
            let left = Value::GpuTensor(handle.clone());
            let right = Value::GpuTensor(handle.clone());
            let probe = Rc::new(GraphProbe::default());
            let runtime = runtime(&probe);
            let outcome = futures::executor::block_on(
                crate::native_auto::AdmittedAcceleration
                    .execute_numeric_region(
                        runtime.clone(),
                        request_values(&runtime, &left, &right, true),
                    )
                    .unwrap(),
            )
            .unwrap();
            let NativeNumericRegionOutcome::Completed { outputs, .. } = outcome else {
                panic!("real resident input must reach provider")
            };
            let Value::GpuTensor(output) = &outputs.values()[0] else {
                panic!("provider output")
            };
            assert_eq!(output.device_id(), handle.device_id());
            assert_eq!(output.physical().element_type(), element);
            assert!(runmat_accelerate_api::handle_is_explicit(output));
            assert!(!probe.kinds.borrow().iter().any(|kind| matches!(
                kind,
                ExecutionCandidateKind::SpecializedNativeCpu
                    | ExecutionCandidateKind::VectorizedNativeCpu
            )));
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_source_host_single_broadcast_executes_canonical_recipe_and_confirms_exactly_once() {
    runmat_gc::gc_test_context(|| {
        for shader_unavailable in [false, true] {
            let provider = FailingProvider::with_abi(usize::MAX, ProviderPrecision::F32, true);
            provider.native_quote_ns.store(100_000, Ordering::SeqCst);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let probe = Rc::new(GraphProbe::default());
            let runtime = runtime(&probe);
            let left =
                Value::Tensor(runmat_value::Tensor::from_f32(vec![1., 2.], vec![2, 1]).unwrap());
            // Mixed Double scalar arithmetic computes in Double before Single
            // publication: an F32 shader is genuinely unavailable, but the same
            // canonical Native source-host operation remains executable.
            let right_input = if shader_unavailable {
                Value::Num(0.25)
            } else {
                Value::Tensor(runmat_value::Tensor::from_f32(vec![3., 4., 5.], vec![1, 3]).unwrap())
            };
            let outcome = futures::executor::block_on(
                crate::native_auto::AdmittedAcceleration
                    .execute_numeric_region(
                        runtime.clone(),
                        request_values(&runtime, &left, &right_input, true),
                    )
                    .unwrap(),
            )
            .unwrap();
            let NativeNumericRegionOutcome::NativeCpu {
                kind: runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic,
                right,
                host: Some(host),
                observation,
            } = outcome
            else {
                panic!("real source-host candidate")
            };
            assert!(probe
                .kinds
                .borrow()
                .contains(&ExecutionCandidateKind::SpecializedNativeCpu));
            assert!(!probe
                .kinds
                .borrow()
                .contains(&ExecutionCandidateKind::VectorizedNativeCpu));
            let route = host.route().clone();
            route
                .begin(&runtime, site(), runmat_types::OperatorKind::Add)
                .unwrap();
            let scoped = runtime.clone().with_native_region_host_route(route.clone());
            let output =
                futures::executor::block_on(runmat_runtime::call_builtin_async_in_context(
                    &scoped,
                    "plus",
                    &[
                        host.input(0).unwrap().clone(),
                        host.input(1).unwrap().clone(),
                    ],
                ))
                .unwrap();
            let Value::Tensor(output) = output.value() else {
                panic!("single numerical host output")
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
            if shader_unavailable {
                assert_eq!(output.shape(), &[2, 1]);
                assert_eq!(output.as_f32_slice().unwrap(), &[1.25, 2.25]);
            } else {
                assert_eq!(output.shape(), &[2, 3]);
                assert_eq!(output.as_f32_slice().unwrap(), &[4., 5., 5., 6., 6., 7.]);
            }
            route.finish(site()).unwrap();
            assert!(route.complete(&runtime));
            if let Some(observation) = observation {
                observation.finish(&runtime, right, true);
            }
            assert!(route.finish(site()).is_err());
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.typed_dispatches.load(Ordering::SeqCst), 0);
            assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_automatic_resident_host_route_uses_one_private_exact_readback_without_rebinding() {
    runmat_gc::gc_test_context(|| {
        for (precision, element) in [
            (
                ProviderPrecision::F32,
                runmat_accelerate_api::NumericElementType::F32,
            ),
            (
                ProviderPrecision::F64,
                runmat_accelerate_api::NumericElementType::F64,
            ),
        ] {
            let provider = FailingProvider::with_abi(usize::MAX, precision, true);
            provider.allow_readback.store(true, Ordering::SeqCst);
            provider.native_quote_ns.store(100_000, Ordering::SeqCst);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let allocation = provider
                .allocations
                .allocate_zeroed(
                    vec![2, 1],
                    element,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            let handle = allocation
                .handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .unwrap();
            let left = Value::GpuTensor(handle.clone());
            let right_input = Value::GpuTensor(handle.clone());
            let probe = Rc::new(GraphProbe::default());
            let cancellation = runmat_execution::cancellation::CancellationSource::new();
            let runtime = runtime(&probe).with_cancellation_token(cancellation.token());
            let outcome = futures::executor::block_on(
                crate::native_auto::AdmittedAcceleration
                    .execute_numeric_region(
                        runtime.clone(),
                        request_values(&runtime, &left, &right_input, true),
                    )
                    .unwrap(),
            )
            .unwrap();
            let NativeNumericRegionOutcome::NativeCpu {
                kind: runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic,
                right,
                host: Some(host),
                observation,
            } = outcome
            else {
                panic!("admitted automatic resident source-host")
            };
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                1,
                "same physical source is retained once"
            );
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert!(
                matches!(&left, Value::GpuTensor(original) if original.view_identity() == handle.view_identity())
            );
            assert_eq!(
                runmat_accelerate_api::handle_provenance(&handle),
                Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            );
            handle.allocation_lease().validate_live().unwrap();
            for index in 0..2 {
                let Value::Tensor(private) = host.input(index).unwrap() else {
                    panic!("private resident input must be a host numerical mirror")
                };
                assert_eq!(private.shape(), &[2, 1]);
                match element {
                    runmat_accelerate_api::NumericElementType::F32 => {
                        assert_eq!(private.as_f32_slice().unwrap(), &[0., 0.]);
                    }
                    runmat_accelerate_api::NumericElementType::F64 => {
                        assert_eq!(private.as_f64_slice().unwrap(), &[0., 0.]);
                    }
                    _ => unreachable!(),
                }
            }
            let route = host.route().clone();
            if element == runmat_accelerate_api::NumericElementType::F32 {
                let mut wrong = site();
                wrong.ordinal += 1;
                assert!(route
                    .clone()
                    .begin(&runtime, wrong, runmat_types::OperatorKind::Add)
                    .is_err());
            } else {
                assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
                assert!(runtime.is_cancelled());
            }
            // Neither wrong occurrence nor cancellation grants permission
            // to replay the original GPU source after private preparation.
            assert!(route
                .begin(&runtime, site(), runmat_types::OperatorKind::Add)
                .is_err());
            drop((host, right, observation));
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
            assert_eq!(provider.typed_dispatches.load(Ordering::SeqCst), 0);
        }
    });
}
