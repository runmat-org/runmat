use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec};
use crate::graph::InstrSpan;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostNumericTensorView, HostTensorView, ProviderFeasibility,
    ProviderFeasibilityQuery, ThreadProviderGuard,
};
use std::future::Future;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

struct FailingProvider {
    typed: bool,
    typed_dispatches: AtomicUsize,
    precision: ProviderPrecision,
    allocations: TestAllocationOwner,
    fail_upload_at: usize,
    uploads: AtomicUsize,
    dispatches: AtomicUsize,
    suspend_dispatch: AtomicBool,
    probes: Mutex<Vec<RetirementProbe>>,
}

impl FailingProvider {
    fn new(fail_upload_at: usize) -> &'static Self {
        Self::with_precision(fail_upload_at, ProviderPrecision::F64)
    }
    fn with_precision(fail_upload_at: usize, precision: ProviderPrecision) -> &'static Self {
        Self::with_abi(fail_upload_at, precision, false)
    }
    fn with_abi(fail_upload_at: usize, precision: ProviderPrecision, typed: bool) -> &'static Self {
        Box::leak(Box::new(Self {
            typed,
            typed_dispatches: AtomicUsize::new(0),
            precision,
            allocations: TestAllocationOwner::new(),
            fail_upload_at,
            uploads: AtomicUsize::new(0),
            dispatches: AtomicUsize::new(0),
            suspend_dispatch: AtomicBool::new(false),
            probes: Mutex::new(Vec::new()),
        }))
    }
}

impl AccelProvider for FailingProvider {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn device_info(&self) -> String {
        "elementwise-effect-fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("numeric upload must use the selected typed hook")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let ordinal = self.uploads.fetch_add(1, Ordering::SeqCst) + 1;
        if ordinal == self.fail_upload_at {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected upload unsupported",
            )
            .into());
        }
        let allocated = self.allocations.allocate_zeroed(
            host.shape.to_vec(),
            match host.data {
                runmat_accelerate_api::HostNumericDataView::F32(_) => {
                    runmat_accelerate_api::NumericElementType::F32
                }
                runmat_accelerate_api::HostNumericDataView::F64(_) => {
                    runmat_accelerate_api::NumericElementType::F64
                }
                _ => panic!("floating fixture input required"),
            },
            runmat_accelerate_api::GpuTensorStorage::Real,
        )?;
        self.probes.lock().unwrap().push(allocated.probe);
        Ok(allocated.handle)
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("failure witness must not read back")
    }

    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::supported(Default::default())
    }
    fn supports_shader_requirements(&self, requirements: ProviderShaderRequirements) -> bool {
        // The injected F64 dispatch lane models IEEE-word shader support.
        // Actual providers negotiate this separately from preferred precision.
        requirements == ProviderShaderRequirements::Basic
            || (self.precision == ProviderPrecision::F64
                && requirements == ProviderShaderRequirements::F64IeeeWords)
    }
    fn supports_fused_elementwise_abi(
        &self,
        _: runmat_accelerate_api::ProviderFusedElementwiseAbi,
    ) -> bool {
        self.typed
    }
    fn fused_elementwise_typed<'a>(
        &'a self,
        request: runmat_accelerate_api::ProviderFusedElementwiseRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            assert_eq!(request.abi.precision(), ProviderPrecision::F32);
            assert!(request
                .inputs
                .iter()
                .all(|input| input.physical().element_type()
                    == runmat_accelerate_api::NumericElementType::F32));
            assert!(request
                .output_classes
                .iter()
                .all(|class| *class == ProviderElementType::F32));
            assert!(!request.shader.contains("f64"));
            assert!(
                request.shader.contains(&format!(
                    "@binding({}) var<storage, read_write> output0",
                    request.inputs.len()
                )),
                "typed fallback must keep the prepared projection and output binding"
            );
            self.typed_dispatches.fetch_add(1, Ordering::SeqCst);
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected typed dispatch unsupported",
            )
            .into())
        })
    }

    fn fused_elementwise<'a>(
        &'a self,
        shader: &'a str,
        _: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &'a [GpuTensorHandle],
        _: &'a [usize],
        _: usize,
        _: ProviderElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if !shader.is_empty() {
                assert!(
                    shader.contains(&format!(
                        "@binding({}) var<storage, read_write> output0",
                        inputs.len()
                    )),
                    "single dispatch bindings must match the already prepared physical inputs"
                );
            }
            if self.suspend_dispatch.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            self.dispatches.fetch_add(1, Ordering::SeqCst);
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected dispatch unsupported",
            )
            .into())
        })
    }
}

fn plan() -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: Vec::new(),
            shape: ShapeInfo::Tensor(vec![Some(2)]),
            span: InstrSpan { start: 0, end: 2 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![crate::fusion::FusionOp::Primitive {
            op: crate::graph::PrimitiveOp::Add,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: Vec::new(),
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

fn invoke(provider: &'static FailingProvider) -> ProviderExecutionError {
    runmat_gc::gc_test_context(|| {
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = plan();
        let tensor = || Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![tensor(), tensor()],
            placement: None,
            runtime: None,
        };
        match futures::executor::block_on(execute_elementwise_placed(request)) {
            Err(error) => error,
            Ok(_) => panic!("injected provider hook must fail, not select a continuation"),
        }
    })
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn certified_double_control_is_not_a_physical_upload_and_empty_fusion_is_pre_effect() {
    runmat_gc::gc_test_context(|| {
        for (empty, multi) in [(false, false), (false, true), (true, false)] {
            let provider = FailingProvider::with_precision(usize::MAX, ProviderPrecision::F32);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let mut source = plan();
            source.operations = vec![crate::fusion::FusionOp::Primitive {
                op: crate::graph::PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            }];
            if multi {
                source
                    .materialized_stores
                    .push(crate::fusion::FusionStoreMaterialization {
                        value_id: 0,
                        binding: crate::graph::VarBinding {
                            kind: crate::graph::VarKind::Global,
                            index: 0,
                        },
                    });
            }
            let values = if empty { vec![] } else { vec![-0.0_f32, -2.0] };
            let shape = vec![1, values.len()];
            source.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(values.len())]);
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: vec![
                    Value::Tensor(runmat_value::Tensor::from_f32(values, shape).unwrap()),
                    Value::Num(2.0),
                ],
                placement: None,
                runtime: None,
            };
            let result = futures::executor::block_on(execute_elementwise_placed(request));
            if empty {
                assert!(matches!(result, Ok(FusionExecutionOutcome::SharedRuntime(_))), "empty source stays with the original per-operation runtime, without a fake provider execution");
                assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
                assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
            } else {
                let error = match result {
                    Err(error) => error,
                    _ => panic!("injected dispatch must be reached"),
                };
                assert_original_unsupported(&error);
                assert_eq!(
                    provider.uploads.load(Ordering::SeqCst),
                    1,
                    "original Double scalar control is embedded, never uploaded or narrowed"
                );
                assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
            }
        }
    });
}

fn assert_original_unsupported(error: &ProviderExecutionError) {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if source.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
            return;
        }
        source = source
            .source()
            .expect("original typed unsupported provider cause retained");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn f64_owner_uses_explicit_single_data_abi_without_narrowing_double_data() {
    runmat_gc::gc_test_context(|| {
        for (case, capable) in [
            (0, true),
            (0, false),
            (1, true),
            (2, true),
            (3, true),
            (4, true),
        ] {
            let provider = FailingProvider::with_abi(usize::MAX, ProviderPrecision::F64, capable);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let mut source = plan();
            let single = || {
                Value::Tensor(
                    runmat_value::Tensor::from_f32(vec![f32::from_bits(1), -2.0], vec![1, 2])
                        .unwrap(),
                )
            };
            let inputs = match case {
                0 => {
                    source.inputs = vec![0];
                    source.stack_pattern = vec![0];
                    source.operations = vec![crate::fusion::FusionOp::Primitive {
                        op: crate::graph::PrimitiveOp::UPlus,
                        inputs: vec![0],
                        output: Some(2),
                    }];
                    vec![single()]
                }
                1 | 4 => {
                    source.operations = vec![crate::fusion::FusionOp::Primitive {
                        op: crate::graph::PrimitiveOp::ElemPow,
                        inputs: vec![0, 1],
                        output: Some(2),
                    }];
                    vec![single(), Value::Num(2.0)]
                }
                2 => vec![
                    single(),
                    Value::Tensor(
                        runmat_value::Tensor::new(vec![1.0 + f64::EPSILON, 2.0], vec![1, 2])
                            .unwrap(),
                    ),
                ],
                _ => {
                    source.inputs = vec![0];
                    source.stack_pattern = vec![0];
                    source.output = Some(4);
                    source.operations = vec![
                        crate::fusion::FusionOp::Builtin {
                            origin: Default::default(),
                            name: "double".into(),
                            inputs: vec![0],
                            output: Some(2),
                        },
                        crate::fusion::FusionOp::Primitive {
                            op: crate::graph::PrimitiveOp::Add,
                            inputs: vec![2, 1],
                            output: Some(3),
                        },
                        crate::fusion::FusionOp::Builtin {
                            origin: Default::default(),
                            name: "single".into(),
                            inputs: vec![3],
                            output: Some(4),
                        },
                    ];
                    source.const_values.insert(1, Value::Num(0.1));
                    vec![single()]
                }
            };
            if case == 4 {
                source
                    .materialized_stores
                    .push(crate::fusion::FusionStoreMaterialization {
                        value_id: 0,
                        binding: crate::graph::VarBinding {
                            kind: crate::graph::VarKind::Global,
                            index: 0,
                        },
                    });
            }
            source.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(2)]);
            let result =
                futures::executor::block_on(execute_elementwise_placed(FusionExecutionRequest {
                    plan: &source,
                    inputs,
                    placement: None,
                    runtime: None,
                }));
            let error = match result {
                Err(error) => error,
                _ => panic!(
                    "selected fixture dispatch must fail: case={case}, capable={capable}, uploads={}, typed={}, legacy={}",
                    provider.uploads.load(Ordering::SeqCst),
                    provider.typed_dispatches.load(Ordering::SeqCst),
                    provider.dispatches.load(Ordering::SeqCst),
                ),
            };
            assert_original_unsupported(&error);
            let typed = capable && case != 2 && case != 3;
            assert_eq!(
                provider.typed_dispatches.load(Ordering::SeqCst),
                usize::from(typed)
            );
            assert_eq!(
                provider.dispatches.load(Ordering::SeqCst),
                usize::from(!typed)
            );
            assert_eq!(
                provider.uploads.load(Ordering::SeqCst),
                if case == 2 { 2 } else { 1 }
            );
        }
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn first_unsupported_upload_remains_pre_effect() {
    let provider = FailingProvider::new(1);
    let error = invoke(provider);
    assert!(error.is_unsupported());
    assert_original_unsupported(&error);
    assert!(error.to_string().contains("injected upload unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    assert!(provider.probes.lock().unwrap().is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn late_unsupported_upload_cannot_replay_after_first_upload() {
    let provider = FailingProvider::new(2);
    let error = invoke(provider);
    assert!(!error.is_unsupported());
    assert_original_unsupported(&error);
    assert!(error.to_string().contains("injected upload unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    let probes = provider.probes.lock().unwrap();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_dispatch_cannot_replay_after_uploads() {
    let provider = FailingProvider::new(usize::MAX);
    let error = invoke(provider);
    assert!(!error.is_unsupported());
    assert_original_unsupported(&error);
    assert!(error.to_string().contains("injected dispatch unsupported"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
    let probes = provider.probes.lock().unwrap();
    assert_eq!(probes.len(), 2);
    assert!(probes.iter().all(|probe| probe.retirements() == 1));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn fused_hook_does_not_dispatch_before_poll_or_after_unpolled_drop() {
    let provider = FailingProvider::new(usize::MAX);
    let inputs = [];
    let shape = [1, 1];
    let future = provider.fused_elementwise(
        "",
        runmat_accelerate_api::ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        ProviderElementType::F64,
    );
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    drop(future);
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    let error = futures::executor::block_on(provider.fused_elementwise(
        "",
        runmat_accelerate_api::ProviderShaderRequirements::Basic,
        &inputs,
        &shape,
        1,
        ProviderElementType::F64,
    ))
    .unwrap_err();
    assert!(error.is_unsupported());
    assert_eq!(provider.dispatches.load(Ordering::SeqCst), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_fused_hook_cancel_preserves_borrowed_inputs() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::new(usize::MAX);
        provider.suspend_dispatch.store(true, Ordering::SeqCst);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let right = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let source = plan();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::GpuTensor(left.handle.clone()),
                Value::GpuTensor(right.handle.clone()),
            ],
            placement: None,
            runtime: None,
        };
        let mut future = Box::pin(execute_elementwise_placed(request));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
        drop(future);
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(left.probe.retirements(), 0);
        assert_eq!(right.probe.retirements(), 0);
    });
}
