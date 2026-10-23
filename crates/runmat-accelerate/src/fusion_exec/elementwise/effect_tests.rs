use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec};
use crate::graph::InstrSpan;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostNumericTensorView, HostTensorView, ProviderFeasibility,
    ProviderFeasibilityQuery, ThreadProviderGuard,
};
use std::future::Future;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

#[path = "effect_tests/native_region.rs"]
mod native_region;

struct FailingProvider {
    native_quote_ns: AtomicUsize,
    quote_outputs: AtomicBool,
    typed: bool,
    typed_dispatches: AtomicUsize,
    precision: ProviderPrecision,
    allocations: TestAllocationOwner,
    fail_upload_at: usize,
    uploads: AtomicUsize,
    dispatches: AtomicUsize,
    suspend_dispatch: AtomicBool,
    probes: Mutex<Vec<RetirementProbe>>,
    successful_fusion: AtomicBool,
    allow_readback: AtomicBool,
    downloads: AtomicUsize,
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
            native_quote_ns: AtomicUsize::new(0),
            quote_outputs: AtomicBool::new(true),
            typed,
            typed_dispatches: AtomicUsize::new(0),
            precision,
            allocations: TestAllocationOwner::new(),
            fail_upload_at,
            uploads: AtomicUsize::new(0),
            dispatches: AtomicUsize::new(0),
            suspend_dispatch: AtomicBool::new(false),
            probes: Mutex::new(Vec::new()),
            successful_fusion: AtomicBool::new(false),
            allow_readback: AtomicBool::new(false),
            downloads: AtomicUsize::new(0),
        }))
    }
    async fn native_download_body(
        &self,
        handle: &GpuTensorHandle,
    ) -> anyhow::Result<runmat_accelerate_api::HostNumericTensorOwned> {
        assert!(
            self.allow_readback.load(Ordering::SeqCst),
            "unexpected private download"
        );
        assert_eq!(handle.device_id(), self.device_id());
        handle.allocation_lease().validate_live()?;
        let count = handle
            .shape()
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .unwrap();
        self.downloads.fetch_add(1, Ordering::SeqCst);
        // This provider allocates zeroed native fixture payloads. This is
        // their actual numeric content, not a fabricated device address.
        let data = match handle.physical().element_type() {
            runmat_accelerate_api::NumericElementType::F32 => {
                runmat_accelerate_api::HostNumericDataOwned::F32(vec![0.0; count])
            }
            runmat_accelerate_api::NumericElementType::F64 => {
                runmat_accelerate_api::HostNumericDataOwned::F64(vec![0.0; count])
            }
            _ => panic!("closed floating source required"),
        };
        Ok(runmat_accelerate_api::HostNumericTensorOwned {
            data,
            shape: handle.shape().to_vec(),
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
    }
}

impl AccelProvider for FailingProvider {
    fn estimate_cost(
        &self,
        _: &runmat_accelerate_api::ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        let ns = self.native_quote_ns.load(Ordering::SeqCst);
        (ns != 0).then_some(runmat_accelerate_api::ProviderCostEstimate {
            cost: ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    execution_ns: ns as u64,
                    ..Default::default()
                },
                scratch_bytes: Some(0),
                confidence: EstimateConfidence::Exact,
                source: EstimateSource::Provider,
            },
        })
    }
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
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        Box::pin(self.native_download_body(handle))
    }
    fn download_numeric_host_workspace_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        if !self.allow_readback.load(Ordering::SeqCst) {
            return runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            );
        }
        let payload =
            runmat_accelerate_api::checked_numeric_download_host_payload_bytes(handle).unwrap();
        let shape = handle
            .shape()
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .unwrap() as u64;
        let future = std::mem::size_of_val(&self.native_download_body(handle)) as u64;
        runmat_execution::ResourceByteBound::Known(
            payload
                .checked_add(shape)
                .unwrap()
                .checked_add(future)
                .unwrap(),
        )
    }
    fn download_numeric_provider_scratch_bound(
        &self,
        _: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Known(0)
    }
    fn download_numeric_queue_slots(&self, _: &GpuTensorHandle) -> u32 {
        0
    }

    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        // The mock allocates exactly the requested output representation.
        // Preserve unknown/overflow rather than inventing a zero bound; the
        // Native rejection matrix deliberately exercises that missing quote.
        let output_bytes = self
            .quote_outputs
            .load(Ordering::SeqCst)
            .then(|| {
                query.outputs.iter().try_fold(0u64, |bytes, output| {
                    bytes.checked_add(output.checked_byte_len()?)
                })
            })
            .flatten();
        ProviderFeasibility::supported(runmat_accelerate_api::ProviderResourceEstimate {
            output_bytes,
            ..Default::default()
        })
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
            // A provider's preferred/default precision does not narrow an
            // explicitly supported typed ABI (notably Single on F64).
            assert!(self.supports_fused_elementwise_abi(request.abi));
            let element = match request.abi.precision() {
                ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
            };
            let class = match request.abi.precision() {
                ProviderPrecision::F32 => ProviderElementType::F32,
                ProviderPrecision::F64 => ProviderElementType::F64,
            };
            assert!(request
                .inputs
                .iter()
                .all(|input| input.physical().element_type() == element));
            assert!(request.output_classes.iter().all(|output| *output == class));
            match request.abi.precision() {
                ProviderPrecision::F32 => assert!(!request.shader.contains("f64")),
                ProviderPrecision::F64 => assert!(request.shader.contains("f64")),
            }
            assert!(self.supports_shader_requirements(request.requirements));
            assert!(
                request.shader.contains(&format!(
                    "@binding({}) var<storage, read_write> output0",
                    request.inputs.len()
                )),
                "typed fallback must keep the prepared projection and output binding"
            );
            self.typed_dispatches.fetch_add(1, Ordering::SeqCst);
            if self.successful_fusion.load(Ordering::SeqCst) {
                return Ok(self
                    .allocations
                    .allocate_zeroed(
                        request.output_shape.to_vec(),
                        element,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )?
                    .handle);
            }
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
        output_shape: &'a [usize],
        _: usize,
        output_class: ProviderElementType,
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
            if self.successful_fusion.load(Ordering::SeqCst) {
                let element = match output_class {
                    ProviderElementType::F32 => runmat_accelerate_api::NumericElementType::F32,
                    ProviderElementType::F64 => runmat_accelerate_api::NumericElementType::F64,
                    _ => panic!("closed floating output required"),
                };
                return Ok(self
                    .allocations
                    .allocate_zeroed(
                        output_shape.to_vec(),
                        element,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )?
                    .handle);
            }
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_ineligibility_is_typed_and_never_masks_invalid_resident_authority() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::with_precision(usize::MAX, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = plan();
        let integer = || {
            Value::Tensor(
                runmat_value::Tensor::from_numeric_storage(
                    runmat_value::NumericStorage::I32(vec![1, 2]),
                    vec![2],
                )
                .unwrap(),
            )
        };
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![integer(), Value::Num(2.0)],
            placement: None,
            runtime: None,
        };
        assert_eq!(
            elementwise_floating_admission(&request).unwrap(),
            ElementwiseFloatingAdmission::NativeIntegerIneligible
        );
        let error = match futures::executor::block_on(execute_elementwise_placed(request)) {
            Err(error) => error,
            Ok(_) => panic!("direct floating fusion must still reject native integer values"),
        };
        assert!(error
            .to_string()
            .contains("dedicated typed integer fusion kernel"));

        let stale = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        stale.probe.mark_unavailable();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![integer(), Value::GpuTensor(stale.handle)],
            placement: None,
            runtime: None,
        };
        let error = elementwise_floating_admission(&request)
            .expect_err("integer ineligibility cannot mask a later stale resident input");
        assert!(error
            .to_string()
            .contains("invalid floating metadata or provider ownership"));
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unavailable_fused_shader_defers_explicit_source_to_original_provider_operations() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::with_precision(usize::MAX, ProviderPrecision::F32);
        let allocation = provider
            .allocations
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let explicit = allocation
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let source = plan();
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![Value::GpuTensor(explicit.clone()), Value::Num(2.0)],
            placement: None,
            runtime: None,
        };
        assert!(host_candidate::unfused_provider_required(&request));
        let mut captured = source.clone();
        captured.inputs = vec![1];
        captured.const_values.insert(0, Value::GpuTensor(explicit));
        let request = FusionExecutionRequest {
            plan: &captured,
            inputs: vec![Value::Num(2.0)],
            placement: None,
            runtime: None,
        };
        assert!(host_candidate::unfused_provider_required(&request));
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(allocation.probe.retirements(), 0);
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn valid_ineligible_inputs_decline_only_after_all_resident_authority_is_validated() {
    runmat_gc::gc_test_context(|| {
        let provider = FailingProvider::with_precision(usize::MAX, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let logical = provider
            .allocations
            .allocate_logical_zeroed(vec![2], ProviderPrecision::F32)
            .unwrap();
        let complex = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let stale = provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        stale.probe.mark_unavailable();
        let source = plan();
        for input in [logical.handle.clone(), complex.handle] {
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: vec![Value::GpuTensor(input), Value::Num(2.0)],
                placement: None,
                runtime: None,
            };
            assert!(matches!(
                futures::executor::block_on(execute_elementwise_placed(request)),
                Ok(FusionExecutionOutcome::SharedRuntime(_))
            ));
        }
        for first in [
            Value::GpuTensor(logical.handle),
            Value::String("host-only".into()),
        ] {
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: vec![first, Value::GpuTensor(stale.handle.clone())],
                placement: None,
                runtime: None,
            };
            let error = match futures::executor::block_on(execute_elementwise_placed(request)) {
                Err(error) => error,
                Ok(_) => panic!("a prior ineligible input must not mask a stale resident input"),
            };
            assert!(error
                .to_string()
                .contains("invalid floating metadata or provider ownership"));
        }
        let foreign_provider = FailingProvider::with_precision(usize::MAX, ProviderPrecision::F32);
        // SAFETY: both providers are leaked fixtures with privately minted,
        // distinct device identities. Registration does not select the foreign
        // provider for this invocation.
        unsafe { runmat_accelerate_api::register_device_provider(foreign_provider) };
        let foreign = foreign_provider
            .allocations
            .allocate_zeroed(
                vec![2],
                runmat_accelerate_api::NumericElementType::F32,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        for first in [
            Value::GpuTensor(
                provider
                    .allocations
                    .allocate_logical_zeroed(vec![2], ProviderPrecision::F32)
                    .unwrap()
                    .handle,
            ),
            Value::String("host-only".into()),
        ] {
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: vec![first, Value::GpuTensor(foreign.handle.clone())],
                placement: None,
                runtime: None,
            };
            let error = match futures::executor::block_on(execute_elementwise_placed(request)) {
                Err(error) => error,
                Ok(_) => panic!("a prior ineligible input must not mask a foreign resident input"),
            };
            assert!(error
                .to_string()
                .contains("invalid floating metadata or provider ownership"));
        }
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(foreign_provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(foreign_provider.dispatches.load(Ordering::SeqCst), 0);
    });
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

struct FinitePreparationOwner(runmat_runtime::context::host_effect::ManagedHostEffectLedger);

impl runmat_runtime::context::RuntimeParallelService for FinitePreparationOwner {
    fn supports(&self, _: runmat_runtime::context::ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> runmat_runtime::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        runmat_runtime::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..Default::default()
        }
    }

    fn managed_host_effect_ledger(
        &self,
    ) -> Option<runmat_runtime::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn finite_scalar_preparation_denies_before_provider_work() {
    runmat_gc::gc_test_context(|| {
        let source = plan();
        let tensor = || Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        let inputs = vec![tensor(), tensor()];
        let semantic_bytes =
            crate::fusion::semantics::preparation_resources::PureSourceGeometry::admit(
                &source, &inputs,
            )
            .unwrap()
            .overlapping_preparation_bytes(None)
            .unwrap();
        for budget in [1, semantic_bytes + 1] {
            let provider = FailingProvider::new(usize::MAX);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let ledger =
                runmat_runtime::context::host_effect::ManagedHostEffectLedger::new_with_epoch(
                    budget, 29,
                );
            let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            ))
            .with_service_ports(
                runmat_runtime::context::RuntimeServicePorts::default()
                    .with_parallel(Rc::new(FinitePreparationOwner(ledger.clone()))),
            );
            let request = FusionExecutionRequest {
                plan: &source,
                inputs: inputs.clone(),
                placement: None,
                runtime: Some(runtime),
            };
            let error = match futures::executor::block_on(execute_elementwise_placed(request)) {
                Err(error) => error,
                Ok(_) => {
                    panic!("finite host authority must reject insufficient preparation capacity")
                }
            };
            assert!(
                !error.is_unsupported(),
                "quota denial cannot authorize host replay"
            );
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
            let mut denied_by_ledger = false;
            while let Some(current) = cause {
                denied_by_ledger |= matches!(
                    current.downcast_ref::<
                        runmat_runtime::context::host_allocation::HostAllocationError,
                    >(),
                    Some(runmat_runtime::context::host_allocation::HostAllocationError::Resource(_))
                );
                cause = current.source();
            }
            assert!(
                denied_by_ledger,
                "the managed finite owner must deny this reservation"
            );
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    });
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
        // A Double scalar without an exact binary32 product receipt is real
        // Double computation, not an F32 shader recipe. Finite preparation
        // must retain the original CPU operation before any provider work.
        let provider = FailingProvider::with_precision(usize::MAX, ProviderPrecision::F32);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let ledger = runmat_runtime::context::host_effect::ManagedHostEffectLedger::new_with_epoch(
            1 << 20,
            31,
        );
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_parallel(Rc::new(FinitePreparationOwner(ledger.clone()))),
        );
        let mut source = plan();
        source.operations = vec![crate::fusion::FusionOp::Primitive {
            op: crate::graph::PrimitiveOp::ElemMul,
            inputs: vec![0, 1],
            output: Some(2),
        }];
        let request = FusionExecutionRequest {
            plan: &source,
            inputs: vec![
                Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![2]).unwrap()),
                Value::Num(1.1),
            ],
            placement: None,
            runtime: Some(runtime),
        };
        assert!(matches!(
            futures::executor::block_on(execute_elementwise_placed(request)),
            Ok(FusionExecutionOutcome::SharedRuntime(_))
        ));
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.dispatches.load(Ordering::SeqCst), 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
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
