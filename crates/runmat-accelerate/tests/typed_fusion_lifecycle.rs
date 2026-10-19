//! Actual async fusion boundary; the recorder is not a numerical GPU oracle.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
use fixture::{cast, plan};
use futures::FutureExt;
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::fusion_exec::{execute_elementwise, FusionExecutionRequest};
use runmat_accelerate::graph::{VarBinding, VarKind};
use runmat_accelerate_api::allocation::WeakGpuTensorHandle;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_value::Value;
use std::future::Future;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::task::{Context, Poll};

struct Recorder {
    owner: TestAllocationOwner,
    casts: AtomicUsize,
    mode: AtomicUsize,
    allocations: Mutex<Vec<(RetirementProbe, WeakGpuTensorHandle)>>,
}
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn new_owner() -> &'static Recorder {
    Box::leak(Box::new(Recorder {
        owner: TestAllocationOwner::new(),
        casts: AtomicUsize::new(0),
        mode: AtomicUsize::new(0),
        allocations: Mutex::new(Vec::new()),
    }))
}
impl Recorder {
    fn input(&self, element: NumericElementType) -> TestAllocation {
        self.owner
            .allocate_zeroed(vec![1, 2], element, GpuTensorStorage::Real)
            .unwrap()
    }
    fn fresh(&self, element: NumericElementType) -> GpuTensorHandle {
        self.track(self.input(element))
    }
    fn track(&self, allocation: TestAllocation) -> GpuTensorHandle {
        self.allocations
            .lock()
            .unwrap()
            .push((allocation.probe, allocation.handle.downgrade()));
        allocation.handle
    }
    fn reset(&self, mode: usize) {
        self.mode.store(mode, Ordering::SeqCst);
        self.casts.store(0, Ordering::SeqCst);
        let mut allocations = self.allocations.lock().unwrap();
        assert!(allocations
            .iter()
            .all(|(probe, _)| probe.retirements() == 1));
        allocations.clear();
    }
    fn assert_retirements(&self, expected: &[usize]) {
        let allocations = self.allocations.lock().unwrap();
        assert_eq!(allocations.len(), expected.len());
        for ((probe, weak), &count) in allocations.iter().zip(expected) {
            assert_eq!(probe.retirements(), count);
            if count == 1 {
                assert!(weak.acquire().is_err());
            }
        }
    }
}
fn transposed(handle: GpuTensorHandle) -> GpuTensorHandle {
    handle
        .with_view(
            handle
                .view()
                .clone()
                .with_layout(GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 1,
                })),
        )
        .unwrap()
}

fn facts(handle: &GpuTensorHandle) -> (GpuPhysicalDescriptor, GpuTensorView) {
    (*handle.physical(), handle.view().clone())
}
impl AccelProvider for Recorder {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn supports_shader_requirements(
        &self,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
    ) -> bool {
        matches!(
            requirements,
            runmat_accelerate_api::ProviderShaderRequirements::Basic
                | runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords
        )
    }
    fn device_info(&self) -> String {
        "typed-fusion-lifecycle".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: None,
            dispatches: Some(1),
        })
    }
    fn reshape(&self, input: &GpuTensorHandle, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        match self.mode.load(Ordering::SeqCst) {
            8 => {
                return Err(unsupported_provider_operation(
                    "layout materialization unavailable",
                ))
            }
            9 => anyhow::bail!("layout materialization failed"),
            10 => return Ok(input.clone()),
            _ => {}
        }
        let output = self.fresh(input.physical().element_type());
        // Every admitted layout request in this fixture preserves its two
        // elements; a failed view construction is a fixture invariant failure.
        Ok(output
            .with_view(GpuTensorView::contiguous(shape.to_vec()))
            .expect("fixture materialization preserves the allocated extent"))
    }
    fn fused_elementwise_multi<'a>(
        &'a self,
        _: &'a str,
        _: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &'a [GpuTensorHandle],
        _: &'a [usize],
        _: usize,
        output_classes: &'a [ProviderElementType],
    ) -> ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move {
            validate_fused_output_classes(self.precision(), output_classes)?;
            match self.mode.load(Ordering::SeqCst) {
                3 => return Err(unsupported_provider_operation("multi unavailable").into()),
                4 => return Err(anyhow::anyhow!("multi execution failed").into()),
                7 => return Ok(vec![inputs[0].clone(), self.fresh(NumericElementType::F64)]),
                _ => {}
            }
            let outputs: Vec<_> = output_classes
                .iter()
                .enumerate()
                .map(|(index, class)| {
                    let output = if *class == ProviderElementType::Logical
                        || (index == 0 && self.mode.load(Ordering::SeqCst) == 5)
                    {
                        self.track(
                            self.owner
                                .allocate_logical_zeroed(vec![1, 2], self.precision())
                                .unwrap(),
                        )
                    } else {
                        self.fresh(NumericElementType::F64)
                    };
                    if index == 0 && self.mode.load(Ordering::SeqCst) == 6 {
                        transposed(output)
                    } else {
                        output
                    }
                })
                .collect();
            Ok(outputs)
        })
    }
    fn fused_elementwise<'a>(
        &'a self,
        _: &'a str,
        _: runmat_accelerate_api::ProviderShaderRequirements,
        _: &'a [GpuTensorHandle],
        _: &'a [usize],
        _: usize,
        output_class: ProviderElementType,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            validate_fused_output_classes(self.precision(), &[output_class])?;
            Ok(if output_class == ProviderElementType::Logical {
                self.track(
                    self.owner
                        .allocate_logical_zeroed(vec![1, 2], self.precision())
                        .map_err(runmat_accelerate_api::ProviderExecutionError::from)?,
                )
            } else {
                self.fresh(NumericElementType::F64)
            })
        })
    }
    fn unary_single<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let ordinal = self.casts.fetch_add(1, Ordering::SeqCst);
            if ordinal == 1 {
                match self.mode.load(Ordering::SeqCst) {
                    0 => futures::future::pending::<()>().await,
                    1 => return Err(anyhow::anyhow!("second output adapter failed").into()),
                    _ => {}
                }
            }
            Ok(self.fresh(NumericElementType::F32))
        })
    }
    fn unary_double<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.fresh(NumericElementType::F64)) })
    }
}

fn request<'a>(
    plan: &'a runmat_accelerate::fusion::FusionGroupPlan,
    input: GpuTensorHandle,
) -> FusionExecutionRequest<'a> {
    FusionExecutionRequest {
        plan,
        inputs: vec![Value::GpuTensor(input)],
        placement: None,
        runtime: None,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn second_adapter_suspend_or_error_releases_every_fresh_allocation_without_publishing() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    for mode in [0, 1] {
        owner.reset(mode);
        let TestAllocation {
            handle: input,
            probe,
        } = owner.input(NumericElementType::F64);
        let before = facts(&input);
        let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        });
        let mut future = Box::pin(execute_elementwise(request(&plan, input.clone())));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        match future.as_mut().poll(&mut context) {
            Poll::Pending => assert_eq!(mode, 0),
            Poll::Ready(result) => {
                assert_eq!(mode, 1);
                assert!(result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("second output adapter failed"));
            }
        }
        drop(future);
        assert_eq!(facts(&input), before);
        owner.assert_retirements(&[1, 1, 1]);
        assert_eq!(probe.retirements(), 0);
        drop(input);
        assert_eq!(probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn duplicate_materialized_bindings_adopt_one_completed_result() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    owner.reset(2);
    let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
    for index in [0, 1] {
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index,
            },
        });
    }
    let TestAllocation {
        handle: input,
        probe,
    } = owner.input(NumericElementType::F64);
    let result = futures::executor::block_on(execute_elementwise(request(&plan, input))).unwrap();
    let Value::GpuTensor(first) = &result.materialized_stores[0].1 else {
        panic!("resident output")
    };
    let Value::GpuTensor(second) = &result.materialized_stores[1].1 else {
        panic!("resident output")
    };
    assert!(first.shares_allocation_with(second));
    assert_eq!(first.view_identity(), second.view_identity());
    assert_eq!(first.physical().element_type(), NumericElementType::F32);
    owner.assert_retirements(&[1, 1, 0, 0]);
    let retained = first.clone();
    drop(result);
    owner.assert_retirements(&[1, 1, 1, 0]);
    drop(retained);
    owner.assert_retirements(&[1, 1, 1, 1]);
    assert_eq!(probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn only_typed_unsupported_multi_falls_back_to_single_dispatches() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    for mode in [3, 4] {
        owner.reset(mode);
        let TestAllocation {
            handle: input,
            probe,
        } = owner.input(NumericElementType::F64);
        let before = facts(&input);
        let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        });
        let result =
            futures::executor::block_on(execute_elementwise(request(&plan, input.clone())));
        assert_eq!(facts(&input), before);
        if mode == 4 {
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("multi execution failed"));
            owner.assert_retirements(&[]);
            assert_eq!(owner.casts.load(Ordering::SeqCst), 0);
        } else {
            let result = result.unwrap();
            let Value::GpuTensor(_) = &result.final_value else {
                panic!("resident")
            };
            let Value::GpuTensor(_) = &result.materialized_stores[0].1 else {
                panic!("resident")
            };
            drop(result);
            owner.assert_retirements(&[1, 1, 1, 1]);
        }
        assert_eq!(probe.retirements(), 0);
        drop(input);
        assert_eq!(probe.retirements(), 1);
    }
}

struct DefaultMulti;
impl AccelProvider for DefaultMulti {
    fn device_info(&self) -> String {
        "default-multi".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn provider_default_multi_declares_unsupported_not_execution_failure() {
    let error = DefaultMulti
        .fused_elementwise_multi(
            "",
            runmat_accelerate_api::ProviderShaderRequirements::Basic,
            &[],
            &[1, 2],
            2,
            &[ProviderElementType::F64; 2],
        )
        .now_or_never()
        .expect("default future is ready")
        .unwrap_err();
    assert!(error.is_unsupported());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mismatched_fresh_producer_facts_are_rejected_and_borrowed_aliases_stay_live() {
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    for mode in [5, 6, 7] {
        owner.reset(mode);
        let TestAllocation {
            handle: input,
            probe,
        } = owner.input(NumericElementType::F64);
        let before = facts(&input);
        let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        });
        assert!(
            futures::executor::block_on(execute_elementwise(request(&plan, input.clone())))
                .is_err()
        );
        assert_eq!(facts(&input), before);
        assert_eq!(owner.casts.load(Ordering::SeqCst), 0);
        owner.assert_retirements(if mode == 7 { &[1] } else { &[1, 1] });
        assert_eq!(probe.retirements(), 0);
        drop(input);
        assert_eq!(probe.retirements(), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn materialized_input_is_owned_through_later_suspension_and_layout_failures_preserve_source() {
    use runmat_accelerate::fusion_exec::{execute_elementwise_placed, FusionExecutionOutcome};
    let owner = new_owner();
    let _provider = ThreadProviderGuard::set(Some(owner));
    for mode in [0, 8, 9, 10] {
        owner.reset(mode);
        let TestAllocation {
            handle: input,
            probe,
        } = owner.input(NumericElementType::F32);
        let input = transposed(input);
        let before = facts(&input);
        let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        });
        let mut future = Box::pin(execute_elementwise_placed(request(&plan, input.clone())));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        match future.as_mut().poll(&mut context) {
            Poll::Pending => assert_eq!(mode, 0),
            Poll::Ready(Ok(FusionExecutionOutcome::SharedRuntime(_))) => assert_eq!(mode, 8),
            Poll::Ready(Err(error)) => {
                assert!(mode == 9 || mode == 10);
                if mode == 9 {
                    assert!(error.to_string().contains("layout materialization failed"));
                }
            }
            _ => panic!("unexpected completed provider result"),
        }
        drop(future);
        assert_eq!(facts(&input), before);
        owner.assert_retirements(if mode == 0 { &[1, 1, 1, 1, 1] } else { &[] });
        assert_eq!(probe.retirements(), 0);
        drop(input);
        assert_eq!(probe.retirements(), 1);
    }
}
