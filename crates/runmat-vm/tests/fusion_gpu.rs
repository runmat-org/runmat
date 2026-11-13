#![allow(clippy::result_large_err)]

#[path = "support/mod.rs"]
mod test_helpers;

#[path = "support/placement_feedback.rs"]
mod placement_feedback;

#[path = "fusion_gpu/persistent.rs"]
mod persistent;

use anyhow::{anyhow, bail, Context};
use futures::executor::block_on;
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
use runmat_accelerate::fusion_exec::{
    execute_elementwise, execute_elementwise_placed, FusionExecutionOutcome, FusionExecutionRequest,
};
use runmat_accelerate::{
    configure_auto_offload, fusion_residency, AutoOffloadLogLevel, AutoOffloadOptions, FusionKind,
};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, AccelProviderFuture,
    ApiDeviceInfo, CorrcoefOptions, CovNormalization, CovRows, CovarianceOptions, FspecialRequest,
    GpuTensorHandle, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorOwned, HostTensorView, ImageNormalizeDescriptor,
    PagefunRequest, PowerStepEpilogue, ProviderCapabilityOperation, ProviderCapabilitySnapshot,
    ProviderConcurrencyCapabilities, ProviderCondNorm, ProviderConvMode, ProviderCostEstimate,
    ProviderCostQuery, ProviderEigResult, ProviderElementType, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderLinsolveOptions, ProviderLinsolveResult, ProviderNormOrder,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderPinvOptions, ProviderPrecision,
    ProviderResourceEstimate, ProviderStorage, UniqueOptions, UniqueResult,
    PROVIDER_CAPABILITY_SCHEMA_VERSION,
};
use runmat_execution::{
    EstimateConfidence, EstimateSource, ExecutionCostComponents, ExecutionCostEstimate,
};
use runmat_gc::gc_test_context;
use runmat_runtime::builtins::image::filters::fspecial::spec_from_request as test_fspecial_spec_from_request;
use runmat_runtime::builtins::math::linalg::ops::mrdivide_host_real_for_provider;
use runmat_runtime::{gather_if_needed, gather_if_needed_async, RuntimeError};
#[cfg(target_arch = "wasm32")]
use runmat_thread_local::runmat_thread_local;
use runmat_value::{Tensor, Value};
use runmat_vm::{
    interpret as interpret_async, interpret_function as interpret_function_async, Instr,
};
use std::collections::HashMap;
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

#[path = "support/fusion_diagnostics.rs"]
mod fusion_diagnostics;

fn interpret(
    bytecode: &runmat_vm::Bytecode,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    block_on(interpret_async(bytecode))
}

fn interpret_function(
    bytecode: &runmat_vm::Bytecode,
    vars: Vec<Value>,
) -> Result<runmat_runtime::execution::RootedValues, RuntimeError> {
    block_on(interpret_function_async(bytecode, vars))
}
use runmat_time::Instant;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[path = "fusion_gpu/ownership.rs"]
mod ownership;
use ownership::{BufferPayload, BufferRecord, BufferStore};

#[path = "fusion_gpu/staged_image.rs"]
mod staged_image;

fn compile_semantic(source: &str) -> runmat_vm::Bytecode {
    test_helpers::compile_source(source).expect("compile source")
}

fn graph_for_fusion_test(
    bytecode: &runmat_vm::Bytecode,
) -> Option<runmat_accelerate::graph::AccelGraph> {
    bytecode
        .reconciled_runtime_fusion_graph()
        .map(|(graph, _)| graph)
}

#[test]
fn fusion_graph_helper_ignores_stale_compile_graph_metadata() {
    gc_test_context(|| {
        ensure_provider_registered();

        let mut bytecode = compile_semantic(
            r#"
            x = [-5.5, -1.25, 5.5];
            y = mod(x, 2) + rem(x, 2);
            "#,
        );
        let stale_bytecode = compile_semantic(
            r#"
            x = [0.25, 0.5, 0.75];
            y = asinh(x) + atanh(x) + acosh(x + 1);
            "#,
        );
        let stale_groups = stale_bytecode.runtime_fusion_groups();
        assert!(
            !stale_groups.is_empty(),
            "expected stale fixture to produce runtime fusion groups"
        );
        let stale_graph = stale_bytecode
            .runtime_accel_graph_for_fusion(&stale_groups)
            .expect("stale runtime graph");
        bytecode.accel_graph = Some(stale_graph.clone());

        let runtime_groups = bytecode.runtime_fusion_groups();
        let (runtime_graph, source) =
            bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
        assert_eq!(source.as_str(), "runtime_materialized_from_instructions");

        let runtime_graph = runtime_graph.expect("runtime graph for fusion");
        assert!(
            !runtime_graph.nodes.is_empty(),
            "expected runtime-materialized graph nodes"
        );
        assert_ne!(
            runtime_graph.nodes.len(),
            stale_graph.nodes.len(),
            "runtime graph should not reuse stale compile graph metadata"
        );

        let graph = graph_for_fusion_test(&bytecode).expect("fusion helper graph");
        let groups = graph.detect_fusion_groups();
        assert!(
            groups
                .iter()
                .any(|g| matches!(g.kind, FusionKind::ElementwiseChain)),
            "expected elementwise fusion group from runtime graph, got {:?}",
            groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
        );
    });
}

struct TestProvider {
    next_id: AtomicU64,
    buffers: Mutex<BufferStore>,
    frees: Mutex<HashMap<u64, usize>>,
    inspect_roots_on_free: AtomicBool,
    fail_free_buffer: AtomicU64,
    single_inputs: Mutex<Vec<u64>>,
    reject_fusion: AtomicBool,
    fused_dispatches: AtomicU64,
    fail_next_fused_single: AtomicBool,
    numeric_downloads: AtomicU64,
    report_f32: AtomicBool,
}

impl TestProvider {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            buffers: Mutex::new(HashMap::new()),
            frees: Mutex::new(HashMap::new()),
            inspect_roots_on_free: AtomicBool::new(false),
            fail_free_buffer: AtomicU64::new(0),
            single_inputs: Mutex::new(Vec::new()),
            reject_fusion: AtomicBool::new(false),
            fused_dispatches: AtomicU64::new(0),
            fail_next_fused_single: AtomicBool::new(false),
            numeric_downloads: AtomicU64::new(0),
            report_f32: AtomicBool::new(false),
        }
    }

    fn pull(&self, handle: &GpuTensorHandle) -> anyhow::Result<(Vec<f64>, Vec<usize>)> {
        let guard = self.buffers.lock().unwrap();
        guard
            .get(&handle.buffer_id())
            .filter(|record| record.owns(handle))
            .and_then(|record| {
                record
                    .data
                    .to_f64()
                    .map(|data| (data, handle.shape().to_vec()))
            })
            .ok_or_else(|| anyhow!("buffer not found: {}", handle.buffer_id()))
    }

    fn push_complex(
        &self,
        data: BufferPayload,
        shape: Vec<usize>,
        element: runmat_accelerate_api::NumericElementType,
    ) -> GpuTensorHandle {
        use runmat_accelerate_api::allocation::{AllocationIdentity, AllocationLease};
        assert!(matches!(
            data,
            BufferPayload::ComplexF32(_) | BufferPayload::ComplexF64(_)
        ));
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let physical = ownership::complex_physical_for(data.len(), element);
        self.buffers
            .lock()
            .unwrap()
            .insert(id, BufferRecord::new(data, shape.clone(), physical, false));
        let registered = registered_provider().expect("registered fixture provider");
        assert!(std::ptr::eq(self, registered.as_ref()));
        let owner: runmat_accelerate_api::allocation::AllocationOwnerHandle = registered.clone();
        let lease = AllocationLease::register(AllocationIdentity::new(0, id, 1), owner)
            .expect("fixture complex allocation registration");
        let handle = GpuTensorHandle::from_allocation(
            lease,
            runmat_accelerate_api::GpuTensorView::contiguous(shape),
        )
        .expect("fixture complex tensor view admission");
        runmat_accelerate_api::mark_residency(&handle);
        handle
    }

    fn push(&self, data: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        self.push_typed(data, shape, runmat_accelerate_api::NumericElementType::F64)
    }

    fn push_typed(
        &self,
        data: Vec<f64>,
        shape: Vec<usize>,
        element: runmat_accelerate_api::NumericElementType,
    ) -> GpuTensorHandle {
        self.push_with_class(data, shape, element, false)
    }

    fn push_logical(&self, data: Vec<bool>, shape: Vec<usize>) -> GpuTensorHandle {
        self.push_with_class(
            data.into_iter()
                .map(|value| u8::from(value) as f64)
                .collect(),
            shape,
            runmat_accelerate_api::NumericElementType::U8,
            true,
        )
    }

    fn push_with_class(
        &self,
        data: Vec<f64>,
        shape: Vec<usize>,
        element: runmat_accelerate_api::NumericElementType,
        logical: bool,
    ) -> GpuTensorHandle {
        use runmat_accelerate_api::allocation::{AllocationIdentity, AllocationLease};
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let data = BufferPayload::from_values(data, element);
        let physical = ownership::physical_for(data.len(), element);
        self.buffers.lock().unwrap().insert(
            id,
            BufferRecord::new(data, shape.clone(), physical, logical),
        );
        let registered = registered_provider().expect("registered fixture provider");
        assert!(std::ptr::eq(self, registered.as_ref()));
        let owner: runmat_accelerate_api::allocation::AllocationOwnerHandle = registered.clone();
        let lease = AllocationLease::register(AllocationIdentity::new(0, id, 1), owner)
            .expect("fixture allocation registration");
        let handle = GpuTensorHandle::from_allocation(
            lease,
            runmat_accelerate_api::GpuTensorView::contiguous(shape).with_logical(logical),
        )
        .expect("fixture tensor view admission");
        runmat_accelerate_api::mark_residency(&handle);
        handle
    }
}

impl AccelProvider for TestProvider {
    fn propose_power_pair_geometry(
        &self,
        inputs: &[runmat_accelerate_api::ProviderRepresentation; 2],
        descriptions: &[runmat_accelerate_api::PowerInputDescription; 2],
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerPairPhysicalProjection>,
    > {
        self.staged_power_pair_geometry(inputs, descriptions, frozen)
    }

    fn propose_power_probe(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    > {
        self.staged_power_probe(base, exponent, frozen)
    }

    fn execute_power_probe_recipe<'a>(
        &'a self,
        recipe: &'a runmat_accelerate_api::power_service::PowerProbeRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::power_service::PowerProbeObserved,
    > {
        Box::pin(async move { self.staged_power_probe_execute(recipe) })
    }

    fn propose_power_evaluation(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: runmat_accelerate_api::PowerResultDomain,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerEvaluationPhysicalProposal>,
    > {
        self.staged_power_evaluation(base, exponent, domain)
    }

    fn execute_power_evaluation_recipe<'a>(
        &'a self,
        recipe: &'a runmat_accelerate_api::power_service::PowerEvaluationRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealElementwise,
    > {
        Box::pin(async move { self.staged_power_evaluation_execute(recipe) })
    }

    fn propose_real_binary(
        &self,
        request: &runmat_accelerate_api::ProviderRealBinaryRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::ProviderRealBinaryPhysicalProposal>,
    > {
        self.staged_binary_proposal(request)
    }

    fn execute_real_binary_recipe<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        recipe: &'a runmat_accelerate_api::ProviderRealBinaryRecipe,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealBinary,
    > {
        Box::pin(async move {
            staged_image::require_unmanaged_metadata(admission)?;
            self.staged_binary_execute(lhs, rhs, recipe)
        })
    }

    fn propose_real_elementwise(
        &self,
        request: &runmat_accelerate_api::ProviderRealElementwiseRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::ProviderRealElementwisePhysicalProposal>,
    > {
        self.staged_real_proposal(request)
    }

    fn execute_real_elementwise_recipe<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        recipe: &'a runmat_accelerate_api::ProviderRealElementwiseRecipe,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealElementwise,
    > {
        Box::pin(async move {
            staged_image::require_unmanaged_metadata(admission)?;
            self.staged_real_execute(input, recipe)
        })
    }

    fn plan_dense_real_output_buffer_bytes(
        &self,
        output: &runmat_accelerate_api::ProviderRepresentation,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<u64>> {
        if output.storage != ProviderStorage::DenseReal
            || output.layout != runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
            || output.residency != runmat_accelerate_api::ProviderResidency::Device
            || !matches!(
                output.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
        {
            return Ok(None);
        }
        Ok(output.checked_byte_len())
    }

    fn plan_power_output_buffer_bytes(
        &self,
        planned: &runmat_accelerate_api::power_service::PowerSymbolicPlan,
        domain: runmat_accelerate_api::PowerResultDomain,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<u64>> {
        Ok(planned
            .physical_work(domain)
            .and_then(|work| work.output.checked_byte_len()))
    }

    fn plan_ordered_real_mean(
        &self,
        source: &runmat_accelerate_api::ProviderRepresentation,
        source_buffer_bytes: u64,
        axes_zero_based: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: runmat_accelerate_api::ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<runmat_accelerate_api::ProviderOrderedRealMeanRecipe>> {
        self.staged_mean_plan(
            source,
            source_buffer_bytes,
            axes_zero_based,
            output_precision,
            nan_mode,
        )
    }

    fn prepare_ordered_real_mean_against_recipe(
        &self,
        source: &GpuTensorHandle,
        recipe: &runmat_accelerate_api::ProviderOrderedRealMeanRecipe,
    ) -> anyhow::Result<Option<runmat_accelerate_api::ProviderOrderedRealMean>> {
        self.staged_mean_prepare(source, recipe)
    }

    fn execute_ordered_real_mean(
        &self,
        selected: runmat_accelerate_api::ProviderOrderedRealMean,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_accelerate_api::ProviderExecutedOrderedRealMean,
    > {
        staged_image::require_unmanaged_metadata(admission)?;
        self.staged_mean_execute(selected)
    }

    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        const OPERATIONS: &[(&str, ProviderOperationFamily)] = &[
            ("transfer.upload", ProviderOperationFamily::Upload),
            ("transfer.download", ProviderOperationFamily::Download),
            ("sin", ProviderOperationFamily::Elementwise),
            ("minus", ProviderOperationFamily::Elementwise),
            ("times", ProviderOperationFamily::Elementwise),
            ("rdivide", ProviderOperationFamily::Elementwise),
            ("single", ProviderOperationFamily::Elementwise),
            ("double", ProviderOperationFamily::Elementwise),
            (
                runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
                ProviderOperationFamily::Elementwise,
            ),
            ("sum", ProviderOperationFamily::Reduction),
            ("fusion.elementwise", ProviderOperationFamily::Fusion),
            ("fusion.matmul_epilogue", ProviderOperationFamily::Fusion),
            (
                "fusion.power_step_normalize",
                ProviderOperationFamily::Fusion,
            ),
            ("fusion.explained_variance", ProviderOperationFamily::Fusion),
            ("fusion.image_normalize", ProviderOperationFamily::Fusion),
        ];
        ProviderCapabilitySnapshot {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 1,
            device: self.device_info_struct(),
            operations: OPERATIONS
                .iter()
                .map(|(operation, family)| ProviderCapabilityOperation {
                    identity: ProviderOperationIdentity::new(*operation),
                    family: *family,
                })
                .collect(),
            element_types: vec![
                ProviderElementType::F32,
                ProviderElementType::F64,
                ProviderElementType::ComplexF32,
                ProviderElementType::ComplexF64,
            ],
            max_rank: None,
            max_allocation_bytes: None,
            sparse: None,
            concurrency: ProviderConcurrencyCapabilities {
                spawn_handles: self.spawn_handle_concurrency(),
                concurrent_dispatch: false,
                cancellation: false,
                transactional_results: true,
            },
        }
    }

    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        if self.reject_fusion.load(Ordering::Relaxed)
            && query.family == ProviderOperationFamily::Fusion
        {
            return ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "test_provider.operation.unsupported",
            );
        }
        let capability = self.capability_snapshot();
        if query.operation.0 == runmat_builtins::POWER_CATALOG_ENTRY.identity.name
            && query.family == ProviderOperationFamily::Elementwise
            && query.outputs.is_empty()
            && query.inputs.len() == 2
            && query.inputs[0].element_type == ProviderElementType::F32
            && query.inputs[0].storage == ProviderStorage::DenseReal
            && query.inputs[0].residency == runmat_accelerate_api::ProviderResidency::Device
            && query.inputs[1].element_type == ProviderElementType::F64
            && query.inputs[1].residency == runmat_accelerate_api::ProviderResidency::Host
            && query.workload.numerical_domain
                == runmat_accelerate_api::NumericalDomainEvidence::RealPower(
                    runmat_accelerate_api::PowerResultDomain::Real,
                )
        {
            if let Some(work) = &query.workload.real_power_scalar {
                if runmat_accelerate_api::ExactSingleBasePower::for_control(work.control).is_some()
                {
                    if let Some(bytes) = work.input.as_ref().map(|input| input.requested_bytes()) {
                        if let Some(request) =
                            runmat_accelerate_api::power_service::exact_single_base_request(
                                &query.inputs[0],
                                work.control,
                                runmat_accelerate_api::RealPowerOperand::Base,
                                bytes,
                            )
                        {
                            if let Ok(Some(physical)) = self.staged_real_proposal(&request) {
                                return ProviderFeasibility::supported(ProviderResourceEstimate {
                                    transient_bytes: Some(physical.uniform_buffer_bytes),
                                    output_bytes: Some(physical.output_buffer_bytes),
                                    dispatches: Some(physical.dispatches),
                                });
                            }
                        }
                    }
                }
            }
        }
        if query.operation.0 == runmat_builtins::POWER_CATALOG_ENTRY.identity.name
            && query.family == ProviderOperationFamily::Elementwise
            && query.outputs.is_empty()
            && query.inputs.len() == 2
            && query.workload.real_power_mode.is_some()
            && matches!(
                query.workload.numerical_domain,
                runmat_accelerate_api::NumericalDomainEvidence::RealPower(_)
            )
            && query.inputs.iter().all(|input| {
                input.storage == ProviderStorage::DenseReal
                    && input.layout == runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
                    && matches!(
                        input.element_type,
                        ProviderElementType::F32 | ProviderElementType::F64
                    )
                    && input.checked_byte_len().is_some()
            })
            && query
                .inputs
                .iter()
                .filter(|input| input.residency == runmat_accelerate_api::ProviderResidency::Device)
                .count()
                == 1
            && query
                .inputs
                .iter()
                .filter(|input| {
                    input.residency == runmat_accelerate_api::ProviderResidency::Host
                        && input.checked_element_count() == Some(1)
                })
                .count()
                == 1
        {
            let base = &query.inputs[0];
            let exponent = &query.inputs[1];
            let output_shape = base
                .shape
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()
                .zip(
                    exponent
                        .shape
                        .iter()
                        .copied()
                        .map(usize::try_from)
                        .collect::<Result<Vec<_>, _>>()
                        .ok(),
                )
                .and_then(|(base, exponent)| {
                    runmat_types::geometry::broadcast_concrete_shape(&base, &exponent).ok()
                });
            if let Some(shape) = output_shape {
                let elements = shape.iter().try_fold(1_u64, |n, &extent| {
                    n.checked_mul(u64::try_from(extent).ok()?)
                });
                // Symbolic feasibility has one device operand and one host
                // scalar. The paired physical hook sees two device operands
                // only after the exact scalar upload (and any widening).
                let element_width = match query.workload.real_power_mode {
                    Some(runmat_accelerate_api::RealPowerArithmeticMode::NativeSingle) => 4,
                    Some(_) => 8,
                    None => {
                        return ProviderFeasibility::rejected(
                            runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                            "test_provider.power_mode.unavailable",
                        )
                    }
                };
                let complex_width = if query.workload.numerical_domain
                    == runmat_accelerate_api::NumericalDomainEvidence::RealPower(
                        runmat_accelerate_api::PowerResultDomain::Complex,
                    ) {
                    2
                } else {
                    1
                };
                if let Some(output_bytes) = elements
                    .and_then(|n| n.checked_mul(element_width))
                    .and_then(|n| n.checked_mul(complex_width))
                {
                    return ProviderFeasibility::supported(ProviderResourceEstimate {
                        transient_bytes: Some(0),
                        output_bytes: Some(output_bytes),
                        dispatches: Some(1),
                    });
                }
            }
        }
        if query.operation.0 == "transfer.upload"
            && query.family == ProviderOperationFamily::Upload
            && query.outputs.len() == 1
            && runmat_accelerate_api::class_preserving_upload_query(query.outputs.clone()).as_ref()
                == Some(query)
            && matches!(
                query.outputs[0].element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
        {
            return ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: Some(0),
                output_bytes: query.outputs[0].checked_byte_len(),
                dispatches: Some(1),
            });
        }
        if matches!(query.operation.0.as_str(), "single" | "double")
            && query.family == ProviderOperationFamily::Elementwise
            && query.inputs.len() == 1
            && query.outputs.len() == 1
            && query.inputs[0].shape == query.outputs[0].shape
            && query.inputs[0].storage == ProviderStorage::DenseReal
            && query.outputs[0].storage == ProviderStorage::DenseReal
            && query.inputs[0].residency == runmat_accelerate_api::ProviderResidency::Device
            && query.outputs[0].residency == runmat_accelerate_api::ProviderResidency::Device
            && matches!(
                query.inputs[0].element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
        {
            let target = if query.operation.0 == "single" {
                ProviderPrecision::F32
            } else {
                ProviderPrecision::F64
            };
            if query.outputs[0].element_type == ProviderElementType::from(target)
                && query.workload
                    == (runmat_accelerate_api::ProviderWorkload {
                        preparation_route: Some(
                            runmat_accelerate_api::ProviderPreparationRoute::FloatingCast {
                                target,
                            },
                        ),
                        ..runmat_accelerate_api::ProviderWorkload::default()
                    })
            {
                let changing = query.inputs[0].element_type != query.outputs[0].element_type;
                let output_bytes = query.outputs[0].checked_byte_len();
                return ProviderFeasibility::supported(ProviderResourceEstimate {
                    transient_bytes: output_bytes,
                    output_bytes,
                    dispatches: Some(u32::from(changing)),
                });
            }
        }
        let native_single_image = query.operation
            == ProviderOperationIdentity::new(
                runmat_accelerate::placement::fusion_operation_token(&FusionKind::ImageNormalize),
            );
        let native_single_elementwise_output = query.operation
            == ProviderOperationIdentity::new(
                runmat_accelerate::placement::fusion_operation_token(&FusionKind::ElementwiseChain),
            );
        let ordinary = match (
            query.operation.0.as_str(),
            query.family,
            query.inputs.as_slice(),
        ) {
            ("minus" | "times" | "rdivide", ProviderOperationFamily::Elementwise, [lhs, rhs])
                if query.outputs.is_empty()
                    && lhs.element_type == rhs.element_type
                    && matches!(
                        lhs.element_type,
                        ProviderElementType::F32 | ProviderElementType::F64
                    )
                    && [lhs, rhs].iter().all(|input| {
                        input.storage == ProviderStorage::DenseReal
                            && input.layout
                                == runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
                            && matches!(
                                input.residency,
                                runmat_accelerate_api::ProviderResidency::Device
                                    | runmat_accelerate_api::ProviderResidency::Host
                            )
                    })
                    && [lhs, rhs].iter().any(|input| {
                        input.residency == runmat_accelerate_api::ProviderResidency::Device
                    }) =>
            {
                let shapes = lhs
                    .shape
                    .iter()
                    .copied()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    .zip(
                        rhs.shape
                            .iter()
                            .copied()
                            .map(usize::try_from)
                            .collect::<Result<Vec<_>, _>>()
                            .ok(),
                    );
                shapes.and_then(|(lhs_shape, rhs_shape)| {
                    runmat_types::geometry::broadcast_concrete_shape(&lhs_shape, &rhs_shape)
                        .ok()
                        .and_then(|shape| {
                            let expansions =
                                u64::from(lhs_shape != shape) + u64::from(rhs_shape != shape);
                            shape
                                .iter()
                                .copied()
                                .try_fold(1_u64, |n, dim| n.checked_mul(u64::try_from(dim).ok()?))
                                .and_then(|count| {
                                    count.checked_mul(
                                        if lhs.element_type == ProviderElementType::F32 {
                                            4
                                        } else {
                                            8
                                        },
                                    )
                                })
                                .and_then(|output| {
                                    let uploads = [lhs, rhs]
                                        .iter()
                                        .filter(|input| {
                                            input.residency
                                                == runmat_accelerate_api::ProviderResidency::Host
                                        })
                                        .try_fold(0_u64, |bytes, input| {
                                            bytes.checked_add(input.checked_byte_len()?)
                                        })?;
                                    let dispatches = u32::try_from(expansions)
                                        .ok()?
                                        .checked_add(u32::from(uploads > 0))?
                                        .checked_add(1)?;
                                    output
                                        .checked_mul(expansions)
                                        .and_then(|scratch| scratch.checked_add(uploads))
                                        .map(|scratch| (output, scratch, dispatches))
                                })
                        })
                })
            }
            ("sin", ProviderOperationFamily::Elementwise, [input])
                if input.element_type == ProviderElementType::F64
                    && input.storage == ProviderStorage::DenseReal
                    && query.outputs.is_empty() =>
            {
                input.checked_byte_len().map(|bytes| (bytes, bytes, 1))
            }
            ("single", ProviderOperationFamily::Elementwise, [input])
                if matches!(
                    input.element_type,
                    ProviderElementType::F32 | ProviderElementType::F64
                ) && input.storage == ProviderStorage::DenseReal
                    && query.outputs.is_empty() =>
            {
                input
                    .checked_element_count()
                    .and_then(|count| count.checked_mul(4))
                    .map(|bytes| (bytes, bytes, 1))
            }
            ("double", ProviderOperationFamily::Elementwise, [input])
                if matches!(
                    input.element_type,
                    ProviderElementType::F32 | ProviderElementType::F64
                ) && input.storage == ProviderStorage::DenseReal
                    && input.layout
                        == runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
                    && input.residency == runmat_accelerate_api::ProviderResidency::Device
                    && query.outputs.is_empty() =>
            {
                input
                    .checked_element_count()
                    .and_then(|count| count.checked_mul(8))
                    .map(|bytes| (bytes, bytes, 1))
            }
            ("sum", ProviderOperationFamily::Reduction, [input])
                if input.element_type == ProviderElementType::F64
                    && input.storage == ProviderStorage::DenseReal
                    && input.shape.len() == 2
                    && query.outputs.is_empty() =>
            {
                query.workload.sum.as_ref().and_then(|route| {
                    if route.axes.iter().any(|axis| *axis > 1) {
                        return None;
                    }
                    let mut shape = input.shape.clone();
                    for &axis in &route.axes {
                        *shape.get_mut(usize::try_from(axis).ok()?)? = 1;
                    }
                    shape
                        .into_iter()
                        .try_fold(8_u64, |bytes, extent| bytes.checked_mul(extent))
                        .map(|bytes| (bytes, bytes, 1))
                })
            }
            _ => None,
        };
        if let Some((bytes, scratch, dispatches)) = ordinary {
            return ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: Some(scratch),
                output_bytes: Some(bytes),
                dispatches: Some(dispatches),
            });
        }
        let supported = capability.supports_operation(&query.operation, query.family)
            && query.inputs.iter().all(|value| {
                (value.element_type == ProviderElementType::F64
                    || (native_single_image && value.element_type == ProviderElementType::F32))
                    && value.storage == ProviderStorage::DenseReal
                    && value.checked_byte_len().is_some()
            })
            && query.outputs.iter().all(|value| {
                (value.element_type == ProviderElementType::F64
                    || (value.element_type == ProviderElementType::F32
                        && (native_single_image || native_single_elementwise_output)))
                    && value.storage == ProviderStorage::DenseReal
                    && value.checked_byte_len().is_some()
            });
        if supported {
            ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: None,
                output_bytes: query.outputs.iter().try_fold(0_u64, |total, value| {
                    total.checked_add(value.checked_byte_len()?)
                }),
                dispatches: Some(1),
            })
        } else {
            ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "test_provider.operation.unsupported",
            )
        }
    }

    fn estimate_cost(&self, query: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
        let elements = query.operation.workload.elements.unwrap_or(0);
        let transfer_dominated_tiny_elementwise =
            query.operation.operation.0 == "fusion.elementwise" && elements <= 2;
        Some(ProviderCostEstimate {
            cost: ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    upload_ns: if transfer_dominated_tiny_elementwise {
                        1_000_000
                    } else {
                        0
                    },
                    execution_ns: 1,
                    ..ExecutionCostComponents::default()
                },
                scratch_bytes: Some(0),
                confidence: EstimateConfidence::Exact,
                source: EstimateSource::Provider,
            },
        })
    }

    fn upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderUploadPhysicalEvidence> {
        if query.operation.0 != "transfer.upload" || query.family != ProviderOperationFamily::Upload
        {
            return None;
        }
        let ProviderFeasibility::Supported { estimate } = self.query_feasibility(query) else {
            return None;
        };
        // The F32 software upload first widens the borrowed source into a
        // temporary f64 Vec before constructing the retained F32 buffer.
        // Count that transient and the copied shape separately from output.
        let output = &query.outputs[0];
        let host_workspace_bytes = match output.element_type {
            ProviderElementType::F32 => output.checked_element_count()?.checked_mul(8)?,
            ProviderElementType::F64 => 0,
            _ => return None,
        }
        .checked_add(
            u64::try_from(output.shape.len())
                .ok()?
                .checked_mul(u64::try_from(std::mem::size_of::<usize>()).ok()?)?,
        )?;
        Some(runmat_accelerate_api::ProviderUploadPhysicalEvidence {
            output_requested_bytes: estimate.output_bytes?,
            host_workspace_bytes: runmat_execution::ResourceByteBound::Known(host_workspace_bytes),
            provider_scratch_bytes: runmat_execution::ResourceByteBound::Known(0),
            completion_queue_slots: 1,
        })
    }

    fn numeric_upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence> {
        let upload = self.upload_physical_evidence(query)?;
        Some(
            runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence {
                output_requested_bytes: upload.output_requested_bytes,
                provider_scratch_bytes: upload.provider_scratch_bytes,
                completion_queue_slots: upload.completion_queue_slots,
            },
        )
    }

    fn floating_cast_physical_evidence(
        &self,
        input: &runmat_accelerate_api::ProviderRepresentation,
        target: ProviderPrecision,
    ) -> Option<runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence> {
        let (source_precision, component_lanes) = match (input.element_type, input.storage) {
            (ProviderElementType::F32, ProviderStorage::DenseReal)
            | (ProviderElementType::F64, ProviderStorage::DenseReal) => (input.element_type, 1),
            (ProviderElementType::ComplexF32, ProviderStorage::DenseComplexInterleaved) => {
                (ProviderElementType::F32, 2)
            }
            (ProviderElementType::ComplexF64, ProviderStorage::DenseComplexInterleaved) => {
                (ProviderElementType::F64, 2)
            }
            _ => return None,
        };
        if source_precision == ProviderElementType::from(target)
            || input.layout != runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
            || input.residency != runmat_accelerate_api::ProviderResidency::Device
        {
            return None;
        }
        let width = if target == ProviderPrecision::F32 {
            4
        } else {
            8
        };
        Some(
            runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence {
                output_requested_bytes: input
                    .checked_element_count()?
                    .checked_mul(width)?
                    .checked_mul(component_lanes)?,
                uniform_buffer_bytes: 0,
                dispatches: 1,
                queue_slots: 1,
            },
        )
    }

    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let (data, _) = self.pull(source)?;
        let mut out = Vec::with_capacity(indices.len());
        for (pos, &idx) in indices.iter().enumerate() {
            let lin = idx as usize;
            if lin >= data.len() {
                return Err(anyhow!(
                    "gather_linear: index {} (pos {}) out of bounds for buffer {}",
                    lin,
                    pos,
                    source.buffer_id()
                ));
            }
            out.push(data[lin]);
        }
        Ok(self.push_typed(out, output_shape.to_vec(), source.physical().element_type()))
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        let (vals, _) = self.pull(values)?;
        if vals.len() != indices.len() {
            return Err(anyhow!(
                "scatter_linear: values length {} mismatch indices {}",
                vals.len(),
                indices.len()
            ));
        }
        let mut guard = self.buffers.lock().unwrap();
        let target_buf = guard
            .get_mut(&target.buffer_id())
            .ok_or_else(|| anyhow!("scatter_linear: unknown target {}", target.buffer_id()))?;
        if !target_buf.owns(target) {
            bail!("scatter_linear: target allocation is not owned by this provider");
        }
        for (pos, &idx) in indices.iter().enumerate() {
            let lin = idx as usize;
            if lin >= target_buf.data.len() {
                return Err(anyhow!(
                    "scatter_linear: index {} (pos {}) out of bounds for buffer {}",
                    lin,
                    pos,
                    target.buffer_id()
                ));
            }
            target_buf
                .data
                .write(lin, vals[pos])
                .map_err(|error| anyhow!(error))?;
        }
        Ok(())
    }

    fn precision(&self) -> ProviderPrecision {
        if self.report_f32.load(Ordering::Relaxed) {
            ProviderPrecision::F32
        } else {
            ProviderPrecision::F64
        }
    }

    fn supports_shader_requirements(
        &self,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
    ) -> bool {
        // This software fixture executes the selected F64 math/IEEE forms;
        // the WGPU provider instead checks enabled device features.
        matches!(
            requirements,
            runmat_accelerate_api::ProviderShaderRequirements::Basic
                | runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords
        )
    }

    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.push(host.data.to_vec(), host.shape.to_vec()))
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        match (host.data, host.storage) {
            (HostNumericDataView::F64(data), runmat_accelerate_api::GpuTensorStorage::Real) => {
                Ok(self.upload(&HostTensorView {
                    data,
                    shape: host.shape,
                })?)
            }
            (HostNumericDataView::F32(data), runmat_accelerate_api::GpuTensorStorage::Real) => {
                let handle = self.push_typed(
                    data.iter().copied().map(f64::from).collect(),
                    host.shape.to_vec(),
                    runmat_accelerate_api::NumericElementType::F32,
                );
                Ok(handle)
            }
            _ => Err(
                anyhow::anyhow!("test provider only supports real floating-point uploads").into(),
            ),
        }
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            suspension::pending_download_if_armed(handle).await;
            let (data, shape) = self.pull(handle)?;
            Ok(HostTensorOwned {
                data,
                shape,
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            suspension::pending_download_if_armed(handle).await;
            self.numeric_downloads.fetch_add(1, Ordering::Relaxed);
            let data = {
                let guard = self.buffers.lock().unwrap();
                let record = guard
                    .get(&handle.buffer_id())
                    .filter(|record| record.owns(handle))
                    .ok_or_else(|| anyhow!("numeric download source is not live"))?;
                match &record.data {
                    BufferPayload::F32(values) => HostNumericDataOwned::F32(values.clone()),
                    BufferPayload::F64(values) => HostNumericDataOwned::F64(values.clone()),
                    BufferPayload::U8(values) => HostNumericDataOwned::U8(values.clone()),
                    BufferPayload::ComplexF32(values) => HostNumericDataOwned::F32(
                        values
                            .iter()
                            .flat_map(|(real, imag)| [*real, *imag])
                            .collect(),
                    ),
                    BufferPayload::ComplexF64(values) => HostNumericDataOwned::F64(
                        values
                            .iter()
                            .flat_map(|(real, imag)| [*real, *imag])
                            .collect(),
                    ),
                }
            };
            Ok(HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle
                    .descriptor()
                    .storage
                    .unwrap_or(runmat_accelerate_api::GpuTensorStorage::Real),
            })
        })
    }

    fn device_info(&self) -> String {
        "test-provider".to_string()
    }

    fn query_numeric_download_feasibility(
        &self,
        source: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        // The deterministic fixture's pull implements only its exact owned
        // contiguous view, not arbitrary strided aliases.
        if !query.matches_numeric_download(source)
            || !matches!(
                source.physical().element_type(),
                runmat_accelerate_api::NumericElementType::F32
                    | runmat_accelerate_api::NumericElementType::F64
            )
            || !self
                .buffers
                .lock()
                .unwrap()
                .get(&source.buffer_id())
                .is_some_and(|record| record.owns(source) && record.shape == source.shape())
            || source.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous
        {
            return ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "test_provider.exact_native_download",
            );
        }
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: Some(0),
            output_bytes: runmat_accelerate_api::checked_numeric_download_host_payload_bytes(
                source,
            ),
            dispatches: Some(0),
        })
    }

    fn download_numeric_host_workspace_bound(
        &self,
        source: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        let bytes = Some(source.physical().payload_bytes());
        match bytes.and_then(|bytes| {
            bytes.checked_add(
                u64::try_from(source.shape().len())
                    .ok()?
                    .checked_mul(std::mem::size_of::<usize>() as u64)?,
            )
        }) {
            Some(bytes) => runmat_execution::ResourceByteBound::Known(bytes),
            None => runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoPeakMemoryBound,
            ),
        }
    }

    fn download_numeric_provider_scratch_bound(
        &self,
        _: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Known(0)
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: 0,
            name: "TestProvider".into(),
            vendor: "RunMat".into(),
            memory_bytes: None,
            backend: Some("test".into()),
        }
    }

    fn logical_isreal(&self, handle: &GpuTensorHandle) -> anyhow::Result<bool> {
        let _ = self.pull(handle)?;
        Ok(true)
    }

    fn logical_isnan(&self, handle: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        let (data, shape) = self.pull(handle)?;
        Ok(self.push_logical(
            data.into_iter().map(|value| value.is_nan()).collect(),
            shape,
        ))
    }

    fn pagefun(&self, _request: &PagefunRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!("pagefun not supported by test provider"))
    }

    fn fspecial(&self, request: &FspecialRequest) -> anyhow::Result<GpuTensorHandle> {
        let spec = test_fspecial_spec_from_request(&request.filter);
        let tensor = spec.generate_tensor().map_err(|error| anyhow!(error))?;
        Ok(self.push(tensor.materialize_f64().clone(), tensor.shape().to_vec()))
    }

    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = async move {
                if second.is_some() {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "test provider: covariance secondary input unsupported",
                    ));
                }
                if weights.is_some() || options.has_weight_vector {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "test provider: covariance weights unsupported",
                    ));
                }
                if options.rows != CovRows::All {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        format!(
                            "test provider: covariance row option {:?} unsupported",
                            options.rows
                        ),
                    ));
                }
                let (data, shape) = self.pull(matrix)?;
                let rows = shape.first().copied().unwrap_or(1);
                let cols = if shape.len() > 1 { shape[1] } else { 1 };
                if rows == 0 || cols == 0 {
                    return Ok(self.push(Vec::new(), vec![cols, cols]));
                }
                let mut means = vec![0.0f64; cols];
                for (c, mean) in means.iter_mut().enumerate().take(cols) {
                    for r in 0..rows {
                        let idx = r + rows * c;
                        if let Some(value) = data.get(idx) {
                            *mean += *value;
                        }
                    }
                    *mean /= rows as f64;
                }
                let denom = match options.normalization {
                    CovNormalization::Unbiased => (rows as f64) - 1.0,
                    CovNormalization::Biased => rows as f64,
                };
                let denom = if denom <= 0.0 { 1.0 } else { denom };
                let mut cov = vec![0.0f64; cols * cols];
                for c1 in 0..cols {
                    for c2 in 0..cols {
                        let mut acc = 0.0f64;
                        for r in 0..rows {
                            let idx1 = r + rows * c1;
                            let idx2 = r + rows * c2;
                            let val1 = data.get(idx1).copied().unwrap_or(0.0);
                            let val2 = data.get(idx2).copied().unwrap_or(0.0);
                            acc += (val1 - means[c1]) * (val2 - means[c2]);
                        }
                        let idx = c1 + cols * c2;
                        cov[idx] = acc / denom;
                    }
                }
                Ok(self.push(cov, vec![cols, cols]))
            }
            .await;
            result.map_err(Into::into)
        })
    }

    fn image_normalize<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        desc: &'a ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let (data, shape) = self.pull(input)?;
            if shape.len() != 3 {
                bail!("test provider: image_normalize expects 3-D tensor");
            }
            let batch = shape[0];
            let height = shape[1];
            let width = shape[2];
            if batch != desc.batch || height != desc.height || width != desc.width {
                bail!(
                    "test provider: image_normalize descriptor mismatch tensor {:?} vs {:?}",
                    shape,
                    (desc.batch, desc.height, desc.width)
                );
            }
            let plane = height * width;
            if plane == 0 {
                return Ok(self.push_typed(Vec::new(), shape, input.physical().element_type()));
            }

            if input.physical().element_type() == runmat_accelerate_api::NumericElementType::F32 {
                let data = data
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>();
                let mut out = vec![0.0f32; data.len()];
                for b in 0..batch {
                    let mut sum = 0.0f32;
                    for idx in 0..plane {
                        sum += data[b + batch * idx];
                    }
                    let mean = sum / plane as f32;
                    let mut sq_sum = 0.0f32;
                    for idx in 0..plane {
                        let diff = data[b + batch * idx] - mean;
                        sq_sum += diff * diff;
                    }
                    let sigma = (sq_sum / plane as f32 + desc.epsilon as f32).sqrt();
                    let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };
                    for idx in 0..plane {
                        let offset = b + batch * idx;
                        let mut value = (data[offset] - mean) * inv_sigma;
                        if let Some(gain) = desc.gain {
                            value *= gain as f32;
                        }
                        if let Some(bias) = desc.bias {
                            value += bias as f32;
                        }
                        if desc.clamp_zero {
                            value = value.max(0.0);
                        }
                        if let Some(gamma) = desc.gamma {
                            value = value.powf(gamma as f32);
                        }
                        out[offset] = value;
                    }
                }
                return Ok(self.push_typed(
                    out.into_iter().map(f64::from).collect(),
                    shape,
                    runmat_accelerate_api::NumericElementType::F32,
                ));
            }

            let mut out = data.clone();
            for b in 0..batch {
                let mut sum = 0.0f64;
                for idx in 0..plane {
                    let offset = b + batch * idx;
                    sum += data[offset];
                }
                let mean = sum / plane as f64;

                let mut sq_sum = 0.0f64;
                for idx in 0..plane {
                    let offset = b + batch * idx;
                    let diff = data[offset] - mean;
                    sq_sum += diff * diff;
                }
                let variance = sq_sum / plane as f64;
                let sigma = (variance + desc.epsilon).sqrt();
                let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };

                for idx in 0..plane {
                    let offset = b + batch * idx;
                    let mut value = (data[offset] - mean) * inv_sigma;
                    if let Some(g) = desc.gain {
                        value *= g;
                    }
                    if let Some(bias) = desc.bias {
                        value += bias;
                    }
                    if desc.clamp_zero {
                        value = value.max(0.0);
                    }
                    if let Some(gamma) = desc.gamma {
                        value = value.powf(gamma);
                    }
                    out[offset] = value;
                }
            }

            Ok(self.push(out, shape))
        })
    }

    fn matmul_power_step<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        epilogue: &'a PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let (lhs_data, lhs_shape) = self.pull(lhs)?;
            let (rhs_data, rhs_shape) = self.pull(rhs)?;
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                bail!("test provider: matmul_power_step expects 2D inputs");
            }
            let m = lhs_shape[0];
            let k = lhs_shape[1];
            if rhs_shape[0] != k {
                bail!("test provider: matmul_power_step inner dimensions mismatch");
            }
            let n = rhs_shape[1];
            let mut product = vec![0.0f64; m * n];
            for col in 0..n {
                for row in 0..m {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        let lhs_idx = row + m * kk;
                        let rhs_idx = kk + k * col;
                        acc += lhs_data[lhs_idx] * rhs_data[rhs_idx];
                    }
                    product[row + m * col] = acc;
                }
            }
            let mut norms = vec![0.0f64; n];
            for (col, norm) in norms.iter_mut().enumerate().take(n) {
                let mut acc = 0.0f64;
                for row in 0..m {
                    let val = product[row + m * col];
                    acc += val * val;
                }
                acc += epilogue.epsilon;
                *norm = acc.sqrt();
            }
            for (col, norm) in norms.iter().copied().enumerate().take(n) {
                for row in 0..m {
                    let idx = row + m * col;
                    product[idx] /= norm;
                }
            }
            Ok(self.push(product, vec![m, n]))
        })
    }

    fn eig<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _compute_left: bool,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderEigResult> {
        Box::pin(async move {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "eig not supported by test provider",
            )
            .into())
        })
    }

    fn linsolve<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        _options: &'a ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderLinsolveResult> {
        Box::pin(async move {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "linsolve not supported by test provider",
            )
            .into())
        })
    }

    fn pinv<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: ProviderPinvOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "pinv not supported by test provider",
            )
            .into())
        })
    }

    fn cond<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _norm: ProviderCondNorm,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "cond not supported by test provider",
            )
            .into())
        })
    }

    fn norm<'a>(
        &'a self,
        _tensor: &'a GpuTensorHandle,
        _order: ProviderNormOrder,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "norm not supported by test provider",
            )
            .into())
        })
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = async {
                let (data, shape) = self.pull(a)?;
                if shape.len() != 2 {
                    bail!("reduce_sum_dim: only 2D supported in test provider");
                }
                let rows = shape[0];
                let cols = shape[1];
                match dim {
                    0 => {
                        // Column-wise: sum over rows -> shape [1, cols]
                        let mut out = vec![0.0f64; cols];
                        for c in 0..cols {
                            let mut acc = 0.0f64;
                            let mut saw_nan = false;
                            for r in 0..rows {
                                let v = data[r + c * rows];
                                if v.is_nan() {
                                    saw_nan = true;
                                    break;
                                }
                                acc += v;
                            }
                            out[c] = if saw_nan { f64::NAN } else { acc };
                        }
                        Ok(self.push(out, vec![1, cols]))
                    }
                    1 => {
                        // Row-wise: sum over cols -> shape [rows, 1]
                        let mut out = vec![0.0f64; rows];
                        for r in 0..rows {
                            let mut acc = 0.0f64;
                            let mut saw_nan = false;
                            for c in 0..cols {
                                let v = data[r + c * rows];
                                if v.is_nan() {
                                    saw_nan = true;
                                    break;
                                }
                                acc += v;
                            }
                            out[r] = if saw_nan { f64::NAN } else { acc };
                        }
                        Ok(self.push(out, vec![rows, 1]))
                    }
                    _ => bail!("reduce_sum_dim: only dims 0 or 1 supported in test provider"),
                }
            }
            .await;
            result.map_err(Into::into)
        })
    }

    fn map_nan_to_zero(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let (data, shape) = self.pull(input)?;
        Ok(self.push(
            data.into_iter()
                .map(|value| if value.is_nan() { 0.0 } else { value })
                .collect(),
            shape,
        ))
    }

    fn reduce_any_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { bail!("reduce_any_dim not supported by test provider") })
    }

    fn reduce_any<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if !runmat_accelerate_api::handle_is_logical(a) {
                bail!("reduce_any requires logical input");
            }
            let (data, _) = self.pull(a)?;
            Ok(self.push_logical(vec![data.iter().any(|value| *value != 0.0)], vec![1, 1]))
        })
    }

    fn reduce_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let (data, _) = self.pull(a)?;
            let minimum = data
                .into_iter()
                .filter(|value| !value.is_nan())
                .reduce(f64::min)
                .ok_or_else(|| anyhow!("reduce_min requires a nonempty numeric input"))?;
            Ok(self.push_typed(vec![minimum], vec![1, 1], a.physical().element_type()))
        })
    }

    fn reduce_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let (data, _) = self.pull(a)?;
            let maximum = data
                .into_iter()
                .filter(|value| !value.is_nan())
                .reduce(f64::max)
                .ok_or_else(|| anyhow!("reduce_max requires a nonempty numeric input"))?;
            Ok(self.push_typed(vec![maximum], vec![1, 1], a.physical().element_type()))
        })
    }

    fn reduce_all_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { bail!("reduce_all_dim not supported by test provider") })
    }

    fn reduce_all<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { bail!("reduce_all not supported by test provider") })
    }

    fn conv2d(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        bail!("conv2d not supported by test provider")
    }

    fn mrdivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let (lhs_data, lhs_shape) = self.pull(lhs)?;
            let (rhs_data, rhs_shape) = self.pull(rhs)?;

            let lhs_tensor = Tensor::new(lhs_data, lhs_shape)
                .map_err(|e| anyhow!("mrdivide (test provider): {e}"))?;
            let rhs_tensor = Tensor::new(rhs_data, rhs_shape)
                .map_err(|e| anyhow!("mrdivide (test provider): {e}"))?;
            let result = mrdivide_host_real_for_provider(&lhs_tensor, &rhs_tensor)
                .map_err(|e| anyhow!("{e}"))?;
            let data = result.materialize_f64();
            Ok(self.push(data, result.shape().to_vec()))
        })
    }

    fn sym_rcm<'a>(&'a self, _matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move { bail!("symrcm not supported by test provider") })
    }

    fn unique<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: &'a UniqueOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<UniqueResult>> {
        let pending = async move {
            let (data, shape) = self.pull(handle)?;
            let tensor =
                Tensor::new(data, shape).map_err(|e| anyhow!("unique (test provider): {e}"))?;
            let eval =
                runmat_runtime::builtins::array::sorting_sets::unique::unique_numeric_from_tensor(
                    tensor, options,
                )
                .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
            eval.into_numeric_unique_result()
                .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
        };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }

    fn corrcoef<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: &'a CorrcoefOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "corrcoef (test provider): not implemented",
            )
            .into())
        })
    }

    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if runmat_accelerate_api::handle_precision(input) != Some(ProviderPrecision::F64)
                || runmat_accelerate_api::handle_storage(input)
                    != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "test sine hook admits only real Double operands",
                ));
            }
            let (data, shape) = self.pull(input)?;
            Ok(self.push(data.into_iter().map(f64::sin).collect(), shape))
        })
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            suspension::pending_single_if_armed().await;
            self.single_inputs.lock().unwrap().push(input.buffer_id());
            if input.physical().element_type() == runmat_accelerate_api::NumericElementType::F32 {
                let live_same_class = self
                    .buffers
                    .lock()
                    .unwrap()
                    .get(&input.buffer_id())
                    .is_some_and(|record| {
                        record.owns(input)
                            && record.shape == input.shape()
                            && matches!(
                                &record.data,
                                BufferPayload::F32(_) | BufferPayload::ComplexF32(_)
                            )
                    });
                if !live_same_class {
                    return Err(runmat_accelerate_api::ProviderExecutionError::from(
                        anyhow!("same-class Single source is not live"),
                    ));
                }
                // The declared zero-dispatch cast is an exact owned view of
                // the input, not a new physical allocation.
                return Ok(input.clone());
            }
            if input.physical().storage()
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                let data = {
                    let guard = self.buffers.lock().unwrap();
                    let record = guard
                        .get(&input.buffer_id())
                        .filter(|record| record.owns(input))
                        .ok_or_else(|| anyhow!("complex Single source is not live"))?;
                    match &record.data {
                        BufferPayload::ComplexF32(values) => values.clone(),
                        BufferPayload::ComplexF64(values) => values
                            .iter()
                            .map(|(real, imag)| (*real as f32, *imag as f32))
                            .collect(),
                        _ => {
                            return Err(runmat_accelerate_api::ProviderExecutionError::from(
                                anyhow!("complex Single source payload is not complex"),
                            ))
                        }
                    }
                };
                return Ok(self.push_complex(
                    BufferPayload::ComplexF32(data),
                    input.shape().to_vec(),
                    runmat_accelerate_api::NumericElementType::F32,
                ));
            }
            let (data, shape) = self.pull(input)?;
            let single = data
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            runmat_accelerate_api::upload_numeric_protecting(
                self,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(&single),
                    shape: &shape,
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                },
                &[input],
            )
            .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)
        })
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if input.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "test Double cast admits only real floating operands",
                )
                .into());
            }
            if input.physical().element_type() == runmat_accelerate_api::NumericElementType::F64 {
                let owned_real_double = self
                    .buffers
                    .lock()
                    .unwrap()
                    .get(&input.buffer_id())
                    .is_some_and(|record| {
                        record.owns(input)
                            && record.shape == input.shape()
                            && matches!(&record.data, BufferPayload::F64(_))
                    });
                if !owned_real_double {
                    return Err(runmat_accelerate_api::ProviderExecutionError::from(
                        anyhow!("same-class Double source is not owned and live"),
                    ));
                }
                input
                    .allocation_lease()
                    .validate_live()
                    .map_err(runmat_accelerate_api::ProviderExecutionError::from_source)?;
                return Ok(input.clone());
            }
            if input.physical().element_type() != runmat_accelerate_api::NumericElementType::F32 {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "test Double cast admits only Single or Double operands",
                )
                .into());
            }
            let (values, shape) = self
                .pull(input)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            Ok(self.push_typed(
                values,
                shape,
                runmat_accelerate_api::NumericElementType::F64,
            ))
        })
    }

    fn elem_sub<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.ordinary_real_binary(
                lhs,
                rhs,
                runmat_accelerate_api::ProviderRealBinaryOperation::Sub,
            )
        })
    }

    fn elem_mul<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.ordinary_real_arithmetic(
                lhs,
                rhs,
                staged_image::real_binary::RealBinaryArithmetic::Mul,
            )
        })
    }

    fn elem_div<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.ordinary_real_binary(
                lhs,
                rhs,
                runmat_accelerate_api::ProviderRealBinaryOperation::Div,
            )
        })
    }

    fn repmat(
        &self,
        input: &GpuTensorHandle,
        repetitions: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.ordinary_real_repmat(input, repetitions)
    }

    fn fused_elementwise<'a>(
        &'a self,
        shader: &'a str,
        _: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &'a [GpuTensorHandle],
        output_shape: &'a [usize],
        len: usize,
        output_class: ProviderElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output_element = ownership::element_for(output_class)?;
            self.fused_dispatches.fetch_add(1, Ordering::Relaxed);
            if self.fail_next_fused_single.swap(false, Ordering::Relaxed) {
                return Err(runmat_accelerate_api::ProviderExecutionError::from_source(
                    runmat_runtime::runtime_error::build_runtime_error(
                        "selected provider rejected single-output execution",
                    )
                    .with_identifier("RunMat:FusionFixture:ProviderFailure")
                    .build(),
                ));
            }
            let execute = || -> anyhow::Result<GpuTensorHandle> {
                let parsed = ParsedShader::parse(shader)?;
                if parsed.tmp_exprs.len() > 1024 {
                    bail!(
                        "test provider: excessive temporary count {}",
                        parsed.tmp_exprs.len()
                    );
                }

                let mut input_values: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());
                let mut input_shapes: Vec<Vec<usize>> = Vec::with_capacity(inputs.len());
                for handle in inputs {
                    let (data, shape) = self.pull(handle)?;
                    input_values.push(data);
                    input_shapes.push(shape);
                }

                let max_input_len = input_values
                    .iter()
                    .map(|data| data.len())
                    .max()
                    .unwrap_or(0);
                let output_elements = output_shape.iter().product::<usize>();
                let total = output_elements.max(len).max(max_input_len).max(1);

                let mut out = Vec::with_capacity(total);
                for idx in 0..total {
                    let mut tmp_values: Vec<Option<f64>> = vec![None; parsed.tmp_exprs.len()];
                    for (slot, expr) in parsed.tmp_exprs.iter().enumerate() {
                        let value = evaluate_expression(
                            expr,
                            idx,
                            &input_values,
                            &input_shapes,
                            output_shape,
                            &tmp_values,
                        )
                        .with_context(|| format!("evaluating tmp{slot} for idx {idx}"))?;
                        tmp_values[slot] = Some(value);
                    }
                    let result = evaluate_expression(
                        &parsed.output_expr,
                        idx,
                        &input_values,
                        &input_shapes,
                        output_shape,
                        &tmp_values,
                    )
                    .with_context(|| format!("evaluating output expression for idx {idx}"))?;
                    out.push(result);
                }

                let mut shape = if output_elements == 0 {
                    if let Some(first) = inputs.first() {
                        first.shape().to_vec()
                    } else {
                        vec![total]
                    }
                } else {
                    output_shape.to_vec()
                };
                if shape.iter().product::<usize>() != total {
                    shape = vec![total];
                }
                Ok(self.push_typed(out, shape, output_element))
            };
            execute().map_err(Into::into)
        })
    }

    fn fused_elementwise_multi<'a>(
        &'a self,
        shader: &'a str,
        _: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &'a [GpuTensorHandle],
        output_shape: &'a [usize],
        len: usize,
        output_classes: &'a [ProviderElementType],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move {
            let execute = || -> anyhow::Result<Vec<GpuTensorHandle>> {
                let output_element_types = output_classes
                    .iter()
                    .copied()
                    .map(ownership::element_for)
                    .collect::<anyhow::Result<Vec<_>>>()?;
                self.fused_dispatches.fetch_add(1, Ordering::Relaxed);
                let parsed = ParsedShader::parse(shader)?;
                if parsed.output_exprs.len() != output_classes.len() {
                    bail!(
                        "test provider: multi-output shader has {} output expressions, expected {}",
                        parsed.output_exprs.len(),
                        output_classes.len()
                    );
                }
                if parsed.tmp_exprs.len() > 1024 {
                    bail!(
                        "test provider: excessive temporary count {}",
                        parsed.tmp_exprs.len()
                    );
                }

                let mut input_values: Vec<Vec<f64>> = Vec::with_capacity(inputs.len());
                let mut input_shapes: Vec<Vec<usize>> = Vec::with_capacity(inputs.len());
                for handle in inputs {
                    let (data, shape) = self.pull(handle)?;
                    input_values.push(data);
                    input_shapes.push(shape);
                }

                let max_input_len = input_values.iter().map(|d| d.len()).max().unwrap_or(0);
                let output_elements = output_shape.iter().product::<usize>();
                let total = output_elements.max(len).max(max_input_len).max(1);

                let shape = if output_elements == 0 {
                    inputs
                        .first()
                        .map(|h| h.shape().to_vec())
                        .unwrap_or_else(|| vec![total])
                } else {
                    output_shape.to_vec()
                };

                let mut per_output: Vec<Vec<f64>> =
                    vec![Vec::with_capacity(total); output_classes.len()];
                for idx in 0..total {
                    let mut tmp_values: Vec<Option<f64>> = vec![None; parsed.tmp_exprs.len()];
                    for (slot, expr) in parsed.tmp_exprs.iter().enumerate() {
                        let value = evaluate_expression(
                            expr,
                            idx,
                            &input_values,
                            &input_shapes,
                            output_shape,
                            &tmp_values,
                        )
                        .with_context(|| format!("evaluating tmp{slot} for idx {idx}"))?;
                        tmp_values[slot] = Some(value);
                    }
                    for (k, out_expr) in parsed.output_exprs.iter().enumerate() {
                        let result = evaluate_expression(
                            out_expr,
                            idx,
                            &input_values,
                            &input_shapes,
                            output_shape,
                            &tmp_values,
                        )
                        .with_context(|| {
                            format!("evaluating output{k} expression for idx {idx}")
                        })?;
                        per_output[k].push(result);
                    }
                }

                let mut handles = Vec::with_capacity(output_classes.len());
                for (out_data, output_element) in per_output.into_iter().zip(output_element_types) {
                    let mut out_shape = shape.clone();
                    if out_shape.iter().product::<usize>() != total {
                        out_shape = vec![total];
                    }
                    handles.push(self.push_typed(out_data, out_shape, output_element));
                }
                Ok(handles)
            };
            execute().map_err(Into::into)
        })
    }

    fn transpose(&self, handle: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        let (data, shape) = self.pull(handle)?;
        if shape.len() < 2 {
            return Ok(self.push_typed(data, shape, handle.physical().element_type()));
        }
        let rows = shape[0];
        let cols = shape[1];
        let mut out = vec![0.0f64; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = r + rows * c;
                let dst = c + cols * r;
                out[dst] = data[src];
            }
        }
        let mut new_shape = shape.clone();
        new_shape[0] = cols;
        new_shape[1] = rows;
        Ok(self.push_typed(out, new_shape, handle.physical().element_type()))
    }

    fn matmul<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let (lhs_data, lhs_shape) = self.pull(lhs)?;
            let (rhs_data, rhs_shape) = self.pull(rhs)?;
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                bail!("test provider: matmul expects 2D inputs");
            }
            let m = lhs_shape[0];
            let k = lhs_shape[1];
            if rhs_shape[0] != k {
                bail!("test provider: matmul inner dimensions mismatch");
            }
            let n = rhs_shape[1];
            let mut out = vec![0.0f64; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0.0;
                    for inner in 0..k {
                        let lhs_idx = row + m * inner;
                        let rhs_idx = inner + k * col;
                        acc += lhs_data[lhs_idx] * rhs_data[rhs_idx];
                    }
                    let dst = row + m * col;
                    out[dst] = acc;
                }
            }
            Ok(self.push(out, vec![m, n]))
        })
    }

    fn diag_extract(
        &self,
        matrix: &GpuTensorHandle,
        offset: isize,
    ) -> anyhow::Result<GpuTensorHandle> {
        if offset != 0 {
            bail!("test provider: diag_extract offset {offset} unsupported");
        }
        let (data, shape) = self.pull(matrix)?;
        if shape.len() != 2 {
            bail!("test provider: diag_extract expects 2D input");
        }
        let rows = shape[0];
        let cols = shape[1];
        let len = rows.min(cols);
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let idx = i + rows * i;
            out.push(data[idx]);
        }
        Ok(self.push(out, vec![len, 1]))
    }

    fn reshape(
        &self,
        handle: &GpuTensorHandle,
        new_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let len = new_shape
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or_else(|| anyhow!("reshape: dimensions exceed addressable length"))?;
        let buffers = self.buffers.lock().unwrap();
        let entry = buffers
            .get(&handle.buffer_id())
            .ok_or_else(|| anyhow!("reshape: unknown buffer {}", handle.buffer_id()))?;
        if entry.data.len() != len {
            bail!(
                "reshape: product of new dimensions ({}) must equal existing length ({})",
                len,
                entry.data.len()
            );
        }
        drop(buffers);
        Ok(handle.with_view(handle.view().clone().with_shape(new_shape.to_vec()))?)
    }
}

struct ParsedShader {
    tmp_exprs: Vec<String>,
    /// Single-output shaders populate `output_expr`; multi-output shaders use `output_exprs`.
    output_expr: String,
    /// Non-empty only for multi-output shaders (`output0.data[g] = …`, `output1.data[g] = …`, …).
    output_exprs: Vec<String>,
}

impl ParsedShader {
    fn parse(shader: &str) -> anyhow::Result<Self> {
        let mut tmp_exprs = Vec::new();
        let mut output_expr: Option<String> = None;
        let mut output_exprs_map: std::collections::BTreeMap<usize, String> =
            std::collections::BTreeMap::new();
        for line in shader.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("let ") {
                if let Some(eq_idx) = rest.find('=') {
                    let (lhs, rhs) = rest.split_at(eq_idx);
                    let name = lhs.split(':').next().map(str::trim).unwrap_or("");
                    if name.starts_with("tmp") || name.starts_with("typed") {
                        let expr = rhs.trim_start_matches('=').trim();
                        let expr = expr.trim_end_matches(';').trim().to_string();
                        tmp_exprs.push(
                            if lhs.split(':').nth(1).is_some_and(|ty| ty.trim() == "f32") {
                                format!("f32({expr})")
                            } else {
                                expr
                            },
                        );
                    }
                }
            } else if let Some(rest) = trimmed
                .strip_prefix("output.data[idx] =")
                .or_else(|| trimmed.strip_prefix("output.data[g] ="))
            {
                let expr = rest.trim().trim_end_matches(';').trim().to_string();
                output_expr = Some(expr);
            } else {
                // Multi-output: `outputK.data[g] = <expr>;`
                if let Some(rest) = trimmed.strip_prefix("output") {
                    if let Some(dot_pos) = rest.find(".data[") {
                        let idx_str = &rest[..dot_pos];
                        if let Ok(k) = idx_str.parse::<usize>() {
                            let after = &rest[dot_pos + ".data[".len()..];
                            // find `] =`
                            if let Some(bracket_pos) = after.find("] =") {
                                let rhs = after[bracket_pos + "] =".len()..].trim();
                                let expr = rhs.trim_end_matches(';').trim().to_string();
                                output_exprs_map.insert(k, expr);
                            }
                        }
                    }
                }
            }
        }

        let output_exprs: Vec<String> = output_exprs_map.into_values().collect();
        let output_expr =
            output_expr.unwrap_or_else(|| output_exprs.first().cloned().unwrap_or_default());
        if output_expr.is_empty() && output_exprs.is_empty() {
            bail!("failed to locate fused output expression");
        }
        Ok(Self {
            tmp_exprs,
            output_expr,
            output_exprs,
        })
    }
}

fn broadcast_linear_position(
    output_linear: usize,
    output_shape: &[usize],
    input_shape: &[usize],
) -> Option<usize> {
    if input_shape.is_empty() || input_shape.iter().product::<usize>() == 0 {
        return Some(0);
    }
    if output_shape.is_empty() {
        return Some(output_linear % input_shape.iter().product::<usize>().max(1));
    }
    let rank = output_shape.len().max(input_shape.len());
    let mut out_dims = vec![1usize; rank];
    let mut in_dims = vec![1usize; rank];
    for (i, dim) in output_shape.iter().copied().enumerate() {
        out_dims[i] = dim.max(1);
    }
    for (i, dim) in input_shape.iter().copied().enumerate() {
        in_dims[i] = dim.max(1);
    }

    let mut coord = vec![0usize; rank];
    let mut tmp = output_linear;
    for d in 0..rank {
        let dim = out_dims[d];
        coord[d] = if dim == 0 { 0 } else { tmp % dim };
        tmp = if dim == 0 { 0 } else { tmp / dim };
    }

    let mut stride = 1usize;
    let mut linear = 0usize;
    for d in 0..rank {
        let dim = in_dims[d];
        let c = if dim == 1 { 0 } else { coord[d].min(dim - 1) };
        linear += c * stride;
        stride = stride.saturating_mul(dim.max(1));
    }
    Some(linear)
}

fn evaluate_expression(
    expr: &str,
    idx: usize,
    inputs: &[Vec<f64>],
    input_shapes: &[Vec<usize>],
    output_shape: &[usize],
    tmp_values: &[Option<f64>],
) -> anyhow::Result<f64> {
    (|| {
        let tokens = tokenize(expr)?;
        let mut parser =
            ExprParser::new(tokens, idx, inputs, input_shapes, output_shape, tmp_values);
        let value = parser.parse_expression()?;
        parser.expect_end()?;
        Ok(value)
    })()
    .map_err(suspension::record_provider_error)
}

#[test]
fn fused_shader_fixture_uses_canonical_remainder_math_and_boolean_literals() {
    let evaluate = |expression| evaluate_expression(expression, 0, &[], &[], &[], &[]).unwrap();
    assert_eq!(evaluate("rm_remainder_f64(-5.5, 2.0, true)"), 0.5);
    assert_eq!(evaluate("rm_remainder_f64(-5.5, 2.0, false)"), -1.5);
    assert_eq!(evaluate("rm_remainder_f32(-5.5, 2.0, true)"), 0.5);
    assert_eq!(evaluate("rm_remainder_f32(-5.5, 2.0, false)"), -1.5);
}

#[derive(Clone, Debug)]
enum Token {
    Number(f64),
    Ident(String),
    Symbol(char),
}

fn tokenize(input: &str) -> anyhow::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_ascii_whitespace() {
            chars.next();
            continue;
        }
        if ch.is_ascii_digit()
            || (ch == '.'
                && chars
                    .clone()
                    .nth(1)
                    .map(|next| next.is_ascii_digit())
                    .unwrap_or(false))
        {
            let mut buf = String::new();
            let mut has_dot = false;
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    buf.push(c);
                    chars.next();
                } else if c == '.' && !has_dot {
                    if buf.is_empty() {
                        buf.push('0');
                    }
                    has_dot = true;
                    buf.push(c);
                    chars.next();
                } else if (c == 'e' || c == 'E') && !buf.is_empty() {
                    buf.push(c);
                    chars.next();
                    if let Some(&sign) = chars.peek() {
                        if sign == '+' || sign == '-' {
                            buf.push(sign);
                            chars.next();
                        }
                    }
                } else {
                    break;
                }
            }
            let value = buf
                .parse::<f64>()
                .map_err(|e| anyhow!("invalid number '{buf}': {e}"))?;
            tokens.push(Token::Number(value));
            continue;
        }
        if ch.is_ascii_alphabetic() || ch == '_' {
            let mut buf = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    buf.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Token::Ident(buf));
            continue;
        }
        tokens.push(Token::Symbol(ch));
        chars.next();
    }
    Ok(tokens)
}

struct ExprParser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    idx: usize,
    inputs: &'a [Vec<f64>],
    input_shapes: &'a [Vec<usize>],
    output_shape: &'a [usize],
    tmp_values: &'a [Option<f64>],
}

impl<'a> ExprParser<'a> {
    fn new(
        tokens: Vec<Token>,
        idx: usize,
        inputs: &'a [Vec<f64>],
        input_shapes: &'a [Vec<usize>],
        output_shape: &'a [usize],
        tmp_values: &'a [Option<f64>],
    ) -> Self {
        Self {
            tokens,
            pos: 0,
            idx,
            inputs,
            input_shapes,
            output_shape,
            tmp_values,
        }
    }

    fn parse_expression(&mut self) -> anyhow::Result<f64> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_logical_and()?;
        while self.match_symbol('|') {
            self.expect_symbol('|')?;
            let rhs = self.parse_logical_and()?;
            value = if value != 0.0 || rhs != 0.0 { 1.0 } else { 0.0 };
        }
        Ok(value)
    }

    fn parse_logical_and(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_additive()?;
        while self.match_symbol('&') {
            self.expect_symbol('&')?;
            let rhs = self.parse_additive()?;
            value = if value != 0.0 && rhs != 0.0 { 1.0 } else { 0.0 };
        }
        Ok(value)
    }

    fn parse_additive(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_comparison()?;
        while let Some(op) = self.match_symbols(&['+', '-']) {
            let rhs = self.parse_comparison()?;
            value = if op == '+' { value + rhs } else { value - rhs };
        }
        Ok(value)
    }

    fn parse_comparison(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_term()?;
        loop {
            if self.match_symbol('=') {
                self.expect_symbol('=')?;
                let rhs = self.parse_term()?;
                value = if value == rhs { 1.0 } else { 0.0 };
                continue;
            }
            if self.match_symbol('>') {
                let inclusive = self.match_symbol('=');
                let rhs = self.parse_term()?;
                let passed = if inclusive { value >= rhs } else { value > rhs };
                value = if passed { 1.0 } else { 0.0 };
                continue;
            }
            if self.match_symbol('<') {
                let inclusive = self.match_symbol('=');
                let rhs = self.parse_term()?;
                let passed = if inclusive { value <= rhs } else { value < rhs };
                value = if passed { 1.0 } else { 0.0 };
                continue;
            }
            break;
        }
        Ok(value)
    }

    fn parse_term(&mut self) -> anyhow::Result<f64> {
        let mut value = self.parse_factor()?;
        while let Some(op) = self.match_symbols(&['*', '/']) {
            let rhs = self.parse_factor()?;
            value = if op == '*' { value * rhs } else { value / rhs };
        }
        Ok(value)
    }

    fn parse_factor(&mut self) -> anyhow::Result<f64> {
        if self.match_symbol('-') {
            return Ok(-self.parse_factor()?);
        }
        if self.match_symbol('+') {
            return self.parse_factor();
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> anyhow::Result<f64> {
        if let Some(token) = self.peek() {
            match token.clone() {
                Token::Number(n) => {
                    self.advance();
                    Ok(n)
                }
                Token::Symbol('(') => {
                    self.advance();
                    let value = self.parse_expression()?;
                    self.expect_symbol(')')?;
                    Ok(value)
                }
                Token::Ident(name) => self.parse_identifier(name),
                Token::Symbol(sym) => Err(anyhow!("unexpected symbol '{sym}' in expression")),
            }
        } else {
            Err(anyhow!("unexpected end of expression"))
        }
    }

    fn parse_identifier(&mut self, name: String) -> anyhow::Result<f64> {
        self.advance();
        if name == "true" {
            return Ok(1.0);
        }
        if name == "false" {
            return Ok(0.0);
        }
        if let Some(rest) = name
            .strip_prefix("tmp")
            .or_else(|| name.strip_prefix("typed"))
        {
            let idx: usize = rest
                .parse()
                .map_err(|e| anyhow!("invalid tmp identifier '{name}': {e}"))?;
            let value = self
                .tmp_values
                .get(idx)
                .ok_or_else(|| anyhow!("tmp index {idx} out of range"))?
                .ok_or_else(|| anyhow!("tmp{idx} used before assignment"))?;
            return Ok(value);
        }
        if name.starts_with("input") && self.check_symbol('.') {
            let rest = name
                .strip_prefix("input")
                .ok_or_else(|| anyhow!("invalid input identifier '{name}'"))?;
            let input_idx: usize = rest
                .parse()
                .map_err(|e| anyhow!("invalid input identifier '{name}': {e}"))?;
            self.expect_symbol('.')?;
            let ident = self.expect_ident()?;
            if ident != "data" {
                bail!("expected 'data' after input identifier, found '{ident}'");
            }
            self.expect_symbol('[')?;
            let idx_ident = self.expect_ident()?;
            let is_input_index = idx_ident == "idx"
                || idx_ident == "g"
                || idx_ident
                    .strip_prefix('i')
                    .map(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
                    .unwrap_or(false);
            if !is_input_index {
                bail!("expected 'idx' access, found '{idx_ident}'");
            }
            self.expect_symbol(']')?;
            let data = self
                .inputs
                .get(input_idx)
                .ok_or_else(|| anyhow!("input index {input_idx} out of range"))?;
            if data.is_empty() {
                bail!("input{input_idx} has no data");
            }
            let shape = self
                .input_shapes
                .get(input_idx)
                .ok_or_else(|| anyhow!("input shape index {input_idx} out of range"))?;
            let pos = broadcast_linear_position(self.idx, self.output_shape, shape)
                .unwrap_or(self.idx % data.len());
            return Ok(data[pos]);
        }

        if self.match_symbol('(') {
            let mut args = Vec::new();
            if !self.match_symbol(')') {
                loop {
                    args.push(self.parse_expression()?);
                    if self.match_symbol(')') {
                        break;
                    }
                    self.expect_symbol(',')?;
                }
            }
            return self.call_function(&name, args);
        }

        if name == "idx"
            || name == "g"
            || name
                .strip_prefix('i')
                .map(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
                .unwrap_or(false)
        {
            return Ok(self.idx as f64);
        }

        Err(anyhow!("unrecognised identifier '{name}'"))
    }

    fn call_function(&self, name: &str, args: Vec<f64>) -> anyhow::Result<f64> {
        let arg = |i: usize| -> anyhow::Result<f64> {
            args.get(i)
                .copied()
                .ok_or_else(|| anyhow!("function '{name}' missing argument {i}"))
        };
        match name {
            "sin" => Ok(arg(0)?.sin()),
            "cos" => Ok(arg(0)?.cos()),
            "tan" => Ok(arg(0)?.tan()),
            "asin" => Ok(arg(0)?.asin()),
            "acos" => Ok(arg(0)?.acos()),
            "atan" => Ok(arg(0)?.atan()),
            "atan2" => Ok(arg(0)?.atan2(arg(1)?)),
            "hypot" => Ok(arg(0)?.hypot(arg(1)?)),
            "sinh" => Ok(arg(0)?.sinh()),
            "cosh" => Ok(arg(0)?.cosh()),
            "tanh" => Ok(arg(0)?.tanh()),
            "asinh" => Ok(arg(0)?.asinh()),
            "acosh" => Ok(arg(0)?.acosh()),
            "atanh" => Ok(arg(0)?.atanh()),
            "exp" => Ok(arg(0)?.exp()),
            "exp2" => Ok(arg(0)?.exp2()),
            "log" => Ok(arg(0)?.ln()),
            "log2" => Ok(arg(0)?.log2()),
            "sqrt" => Ok(arg(0)?.sqrt()),
            "abs" => Ok(arg(0)?.abs()),
            "sign" => Ok(arg(0)?.signum()),
            "floor" => Ok(arg(0)?.floor()),
            "ceil" => Ok(arg(0)?.ceil()),
            "round" => Ok(arg(0)?.round()),
            "trunc" => Ok(arg(0)?.trunc()),
            "expm1" => Ok(arg(0)?.exp_m1()),
            "log1p" => Ok(arg(0)?.ln_1p()),
            "pow" => Ok(arg(0)?.powf(arg(1)?)),
            "isNan" => Ok(if arg(0)?.is_nan() { 1.0 } else { 0.0 }),
            "isFinite" => Ok(if arg(0)?.is_finite() { 1.0 } else { 0.0 }),
            "isInf" => Ok(if arg(0)?.is_infinite() { 1.0 } else { 0.0 }),
            "select" => Ok(if arg(2)? != 0.0 { arg(1)? } else { arg(0)? }),
            "rm_remainder_f32" => {
                let kind = if arg(2)? != 0.0 {
                    runmat_types::FloatingRemainderKind::Modulus
                } else {
                    runmat_types::FloatingRemainderKind::Remainder
                };
                Ok(f64::from(
                    runmat_runtime::builtins::math::rounding::floating_remainder::apply_f32(
                        arg(0)? as f32,
                        arg(1)? as f32,
                        kind,
                    ),
                ))
            }
            "rm_remainder_f64" => {
                let kind = if arg(2)? != 0.0 {
                    runmat_types::FloatingRemainderKind::Modulus
                } else {
                    runmat_types::FloatingRemainderKind::Remainder
                };
                Ok(
                    runmat_runtime::builtins::math::rounding::floating_remainder::apply_f64(
                        arg(0)?,
                        arg(1)?,
                        kind,
                    ),
                )
            }
            "f32" => Ok(f64::from(arg(0)? as f32)),
            "f64" => Ok(arg(0)?),
            _ => Err(anyhow!("unsupported function '{name}' in fused shader")),
        }
    }

    fn expect_end(&self) -> anyhow::Result<()> {
        if self.pos >= self.tokens.len() {
            Ok(())
        } else {
            Err(anyhow!("unexpected trailing tokens in expression"))
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn match_symbol(&mut self, symbol: char) -> bool {
        if let Some(Token::Symbol(ch)) = self.peek() {
            if *ch == symbol {
                self.advance();
                return true;
            }
        }
        false
    }

    fn match_symbols(&mut self, symbols: &[char]) -> Option<char> {
        if let Some(Token::Symbol(ch)) = self.peek() {
            if symbols.contains(ch) {
                let symbol = *ch;
                self.advance();
                return Some(symbol);
            }
        }
        None
    }

    fn check_symbol(&self, symbol: char) -> bool {
        matches!(self.peek(), Some(Token::Symbol(ch)) if *ch == symbol)
    }

    fn expect_symbol(&mut self, symbol: char) -> anyhow::Result<()> {
        if self.match_symbol(symbol) {
            Ok(())
        } else {
            Err(anyhow!("expected symbol '{symbol}'"))
        }
    }

    fn expect_ident(&mut self) -> anyhow::Result<String> {
        if let Some(Token::Ident(name)) = self.peek() {
            let ident = name.clone();
            self.advance();
            Ok(ident)
        } else {
            Err(anyhow!("expected identifier"))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
type TestProviderHandle = Arc<TestProvider>;
#[cfg(target_arch = "wasm32")]
type TestProviderHandle = Rc<TestProvider>;

#[cfg(not(target_arch = "wasm32"))]
static PROVIDER: OnceCell<TestProviderHandle> = OnceCell::new();

#[cfg(target_arch = "wasm32")]
runmat_thread_local! {
    static WORKER_PROVIDER: std::cell::Cell<Option<&'static TestProviderHandle>> = const {
        std::cell::Cell::new(None)
    };
}

fn registered_provider() -> Option<&'static TestProviderHandle> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        PROVIDER.get()
    }
    #[cfg(target_arch = "wasm32")]
    {
        WORKER_PROVIDER.with(std::cell::Cell::get)
    }
}

fn ensure_provider_registered() {
    #[cfg(not(target_arch = "wasm32"))]
    let provider: &'static TestProvider = PROVIDER.get_or_init(|| Arc::new(TestProvider::new()));
    #[cfg(target_arch = "wasm32")]
    let provider: &'static TestProvider = {
        if registered_provider().is_none() {
            let provider: &'static TestProviderHandle =
                Box::leak(Box::new(Rc::new(TestProvider::new())));
            WORKER_PROVIDER.with(|slot| slot.set(Some(provider)));
        }
        registered_provider().expect("worker provider initialized")
    };
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    configure_auto_offload(AutoOffloadOptions {
        enabled: true,
        calibrate: false,
        log_level: AutoOffloadLogLevel::Trace,
    });
}

fn accel_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    configure_auto_offload(AutoOffloadOptions {
        enabled: false,
        calibrate: false,
        log_level: AutoOffloadLogLevel::Trace,
    });
    runmat_accelerate_api::set_thread_provider(None);
    runmat_accelerate_api::clear_provider();
    guard
}

struct RejectFusion(&'static TestProvider);

impl Drop for RejectFusion {
    fn drop(&mut self) {
        self.0.reject_fusion.store(false, Ordering::Relaxed);
        self.0.report_f32.store(false, Ordering::Relaxed);
    }
}

#[path = "fusion_gpu/completion.rs"]
mod completion;
#[path = "fusion_gpu/consumed_values.rs"]
mod consumed_values;
#[path = "fusion_gpu/direct.rs"]
mod direct;
#[path = "fusion_gpu/observation.rs"]
mod observation;
#[path = "fusion_gpu/suspension.rs"]
mod suspension;
#[path = "fusion_gpu/terminal.rs"]
mod terminal;

#[test]
fn fusion_feasibility_rejection_resumes_shared_runtime_before_provider_side_effects() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        configure_auto_offload(AutoOffloadOptions {
            enabled: false,
            calibrate: false,
            log_level: AutoOffloadLogLevel::Trace,
        });
        let provider = registered_provider().expect("test provider");
        provider.reject_fusion.store(true, Ordering::Relaxed);
        let _reject = RejectFusion(provider);
        runmat_accelerate::placement::reset();

        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_placement(placement.clone()),
        );
        let allocations_before = provider.next_id.load(Ordering::Relaxed);
        let dispatches_before = provider.fused_dispatches.load(Ordering::Relaxed);

        let bytecode = compile_semantic(
            r#"
            x = [0.25, 0.5, 0.75];
            y = sin(x) .* x + 2;
            "#,
        );
        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                None,
                runtime,
                None,
                None,
            ))
            .outcome
            .expect("shared-runtime fusion after provider rejection");
        let output_slot = *bytecode
            .var_names
            .iter()
            .find(|(_, name)| name.as_str() == "y")
            .expect("y output slot")
            .0;
        let output_value = completion
            .values
            .get(output_slot)
            .expect("completed output slot");
        let Value::Tensor(output) = &*output_value else {
            panic!(
                "shared runtime must return a host tensor: {:?}",
                &*output_value
            );
        };
        assert_eq!(output.shape(), vec![1, 3]);
        for (actual, input) in output
            .materialize_f64()
            .into_iter()
            .zip([0.25_f64, 0.5, 0.75])
        {
            assert!((actual - (input.sin() * input + 2.0)).abs() < 1e-12);
        }
        assert_eq!(provider.next_id.load(Ordering::Relaxed), allocations_before);
        assert_eq!(
            provider.fused_dispatches.load(Ordering::Relaxed),
            dispatches_before
        );
        let feedback = placement.profile_snapshot().feedback;
        let report = runmat_accelerate::placement_report();
        let rejected = report
            .traces
            .iter()
            .filter(|trace| {
                trace.operation == "fusion.elementwise"
                    && trace.events.iter().any(|event| {
                        event.kind == runmat_accelerate::placement::PlacementEventKind::Selected
                            && event.variant
                                == Some(
                                    runmat_accelerate::placement::PlacementVariant::SharedRuntime,
                                )
                            && event.reason.as_deref() == Some("provider_operation_unsupported")
                    })
            })
            .collect::<Vec<_>>();
        assert!(!rejected.is_empty(), "expected a rejected fusion region");
        for trace in &rejected {
            assert!(trace.complete);
            assert!(
                trace.events.iter().any(|event| {
                    event.kind == runmat_accelerate::placement::PlacementEventKind::Prepare
                        && event.reason.as_deref() == Some("generate_wgsl")
                        && event.duration_ns.is_some()
                }),
                "host shader-source preparation remains observed"
            );
            assert!(trace.events.iter().any(|event| {
                event.kind == runmat_accelerate::placement::PlacementEventKind::Complete
                    && event.variant
                        == Some(runmat_accelerate::placement::PlacementVariant::SharedRuntime)
            }));
            assert!(!trace.events.iter().any(|event| {
                event.kind == runmat_accelerate::placement::PlacementEventKind::Fallback
                    && event.reason.as_deref() == Some("fusion_execution_failed")
            }));
            assert!(
                !trace.events.iter().any(|event| {
                    matches!(
                        event.kind,
                        runmat_accelerate::placement::PlacementEventKind::Upload
                            | runmat_accelerate::placement::PlacementEventKind::Compile
                            | runmat_accelerate::placement::PlacementEventKind::Queue
                            | runmat_accelerate::placement::PlacementEventKind::Kernel
                    )
                }),
                "rejected CPU trace contains provider stages: {trace:#?}"
            );
        }
        let cpu_samples = feedback
            .iter()
            .filter(|series| series.candidate == "cpu.shared.fusion.elementwise")
            .map(|series| {
                assert_eq!(series.failures, 0);
                series.samples
            })
            .sum::<u64>();
        assert_eq!(
            cpu_samples, 0,
            "the rejected CPU-only route has no complete-call observation permit"
        );
        assert!(feedback
            .iter()
            .all(|series| { !series.candidate.starts_with("provider.") || series.samples == 0 }));
    });
}

#[test]
fn ordinary_gpu_sum_survives_unclaimed_fused_reduction_without_fusion_upload() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        let provider = registered_provider().expect("test provider");
        assert!(
            !provider.capability_snapshot().supports_operation(
                &ProviderOperationIdentity::new("fusion.reduction"),
                ProviderOperationFamily::Fusion,
            ),
            "fixture must not claim a fused hook it does not implement"
        );

        let bytecode = compile_semantic("x = [1, 2; 3, 4]; S = sum(x, 1);");
        let output_slot = *bytecode
            .var_names
            .iter()
            .find(|(_, name)| name.as_str() == "S")
            .expect("S output slot")
            .0;
        let input_slot = *bytecode
            .var_names
            .iter()
            .find(|(_, name)| name.as_str() == "x")
            .unwrap()
            .0;
        let store = bytecode
            .instructions
            .iter()
            .position(
                |instruction| matches!(instruction, Instr::StoreVar(index) if *index == input_slot),
            )
            .expect("literal input store");
        let mut vars = vec![Value::Num(0.0); bytecode.var_count];
        vars[input_slot] = Value::GpuTensor(provider.push(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        let resume = runmat_vm::InterpreterResumeState {
            pc: store + 1,
            completion_boundary: None,
            vars: runmat_vm::InterpreterVars::from_owned(vars),
            assigned_slots: std::collections::HashSet::from([input_slot]),
            supplied_inputs: 0,
            requested_outputs: 0,
            missing_input_slots: Default::default(),
            global_aliases: Default::default(),
            persistent_aliases: Default::default(),
            side_effect_epoch: 0,
        };
        let allocations_before = provider.next_id.load(Ordering::Relaxed);
        let runmat_vm::InterpreterOutcome::Completed(completion) =
            block_on(runmat_vm::interpret_resume_in_context(
                &bytecode,
                resume,
                None,
                runmat_runtime::context::RuntimeContext::new(Rc::new(
                    runmat_runtime::execution::RuntimeExecutionService::new(),
                )),
            ))
            .expect("ordinary resident GPU sum");
        let result_value = completion
            .values
            .get(output_slot)
            .expect("ordinary resident output");
        let Value::GpuTensor(result) = &*result_value else {
            panic!("ordinary sum must preserve resident output");
        };
        assert_eq!(provider.pull(result).unwrap(), (vec![3.0, 7.0], vec![1, 2]));
        assert_eq!(
            provider.next_id.load(Ordering::Relaxed),
            allocations_before + 1,
            "unsupported fused reduction must not stage an additional upload"
        );
    });
}

#[test]
#[ignore]
fn fused_elementwise_then_reduction_sum_rows_profiled() {
    gc_test_context(|| {
        // Generate a reasonably large matrix; try a couple sizes to exercise GPU path
        let sizes = &[(512usize, 512usize), (1024, 1024)];
        for &(rows, cols) in sizes {
            let data: Vec<f64> = (0..rows * cols)
                .map(|i| (i as f64).sin() * 0.5 + 0.5)
                .collect();
            let source = format!(
                r#"
                X = zeros({rows}, {cols});
                for c = 1:{cols}
                    for r = 1:{rows}
                        X(r, c) = c * 10 + r;
                    end
                end
                Y = sin(X) .* X + 2;
                S = sum(Y, 2);
                "#
            );

            let bytecode = compile_semantic(&source);

            // Initialize vars and insert tensor for 'X' by locating first LoadVar use
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let x_index = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(idx) => Some(*idx),
                    _ => None,
                })
                .next()
                .unwrap_or(0);
            let tensor = runmat_value::Tensor::new(data, vec![rows, cols]).unwrap();
            vars[x_index] = Value::Tensor(tensor);

            // CPU path (no provider registered yet)
            let _ = interpret_function(&bytecode, vars.clone());
            let start_cpu = Instant::now();
            let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("interpret");
            let elapsed_cpu = start_cpu.elapsed();
            let s_index = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(i) => Some(*i),
                    _ => None,
                })
                .next_back()
                .expect("store var for S");
            let s_value_cpu = vars_cpu.get(s_index).expect("value for S (cpu)");
            let gathered_cpu = gather_if_needed(s_value_cpu).expect("gather cpu");
            let out_cpu = match gathered_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor S (cpu), got {other:?}"),
            };
            assert_eq!(
                out_cpu.materialize_f64().len(),
                rows,
                "CPU output elements should equal rows"
            );

            // GPU path: register provider and run again
            ensure_provider_registered();
            let _ = interpret_function(&bytecode, vars.clone());
            let start_gpu = Instant::now();
            let vars_gpu = interpret_function(&bytecode, vars).expect("interpret");
            let elapsed_gpu = start_gpu.elapsed();
            // Attempt to gather S at the computed index; if shape doesn't match expectation, scan for a length==rows tensor
            let out_gpu = {
                let s_value_gpu = vars_gpu.get(s_index).expect("value for S (gpu)");
                let gathered = gather_if_needed(s_value_gpu).expect("gather gpu");
                let pick_alt = match &gathered {
                    Value::Tensor(t) => t.materialize_f64().len() != rows,
                    _ => true,
                };
                if !pick_alt {
                    match gathered {
                        Value::Tensor(t) => t,
                        _ => unreachable!(),
                    }
                } else {
                    // Fallback: search all variables for a tensor of length==rows
                    let mut found: Option<runmat_value::Tensor> = None;
                    for v in vars_gpu.values() {
                        if let Ok(Value::Tensor(t)) = gather_if_needed(v) {
                            if t.materialize_f64().len() == rows {
                                found = Some(t);
                                break;
                            }
                        }
                    }
                    found.expect("expected to find reduced tensor of length rows")
                }
            };
            assert_eq!(
                out_gpu.materialize_f64().len(),
                rows,
                "GPU output elements should equal rows"
            );

            // Log basic perf info
            eprintln!(
                "fused sin.* + sum rows: {rows}x{cols} -> [{rows}x1] CPU: {:?} GPU: {:?}",
                elapsed_cpu, elapsed_gpu
            );
        }
    });
}

#[test]
fn reduction_sum_omitnan_vs_include_dim2_gpu_cpu() {
    gc_test_context(|| {
        // Matrix with NaNs in each row
        let rows = 4usize;
        let cols = 5usize;
        let mut data = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[r + c * rows] = (r as f64) + (c as f64);
            }
            // place a NaN in each row
            data[r + ((r % cols) * rows)] = f64::NAN;
        }

        // Program computes include-nan and omit-nan variants along rows (dim=2)
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            for c = 1:cols
                for r = 1:rows
                    X(r, c) = c * 10 + r;
                end
            end
            % Place a NaN in the first column of each row
            for r = 1:rows
                X(r, 1) = NaN;
            end
            SI = sum(X, 2);           % include nan (default)
            SO = sum(X, 2, "omitnan");
        "#
        );

        let bytecode = compile_semantic(&source);
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            let groups = graph.detect_fusion_groups();
            let reduction_count = groups
                .iter()
                .filter(|g| matches!(g.kind, runmat_accelerate::fusion::FusionKind::Reduction))
                .count();
            assert!(
                reduction_count >= 2,
                "expected at least two reduction groups for include/omit nan sums, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
        } else {
            panic!("bytecode missing accel graph");
        }

        // Helper to run with/without GPU
        let run_with = |use_gpu: bool| -> runmat_runtime::execution::RootedValues {
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            if use_gpu {
                ensure_provider_registered();
            }
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Warmups
        let _ = run_with(false);
        let _ = run_with(true);

        // CPU and GPU results
        let cpu = run_with(false);
        let gpu = run_with(true);

        let var_index = |name: &str| {
            bytecode
                .var_names
                .iter()
                .find_map(|(idx, candidate)| (candidate == name).then_some(*idx))
                .unwrap_or_else(|| panic!("missing variable {name}"))
        };
        let si_idx = var_index("SI");
        let so_idx = var_index("SO");

        let gather_numvec = |value: &Value| -> Vec<f64> {
            match value {
                Value::GpuTensor(_handle) => {
                    let v = gather_if_needed(value).unwrap();
                    match v {
                        Value::Tensor(t) => t.materialize_f64(),
                        _ => panic!("expected tensor"),
                    }
                }
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected tensor, got {other:?}"),
            }
        };

        // Compare CPU vs GPU for include-nan and omit-nan
        let si_cpu = gather_numvec(cpu.get(si_idx).unwrap());
        let si_gpu = gather_numvec(gpu.get(si_idx).unwrap());
        assert_eq!(si_cpu.len(), rows);
        assert_eq!(si_gpu.len(), rows);
        for (a, b) in si_cpu.iter().zip(si_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }

        let so_cpu = gather_numvec(cpu.get(so_idx).unwrap());
        let so_gpu = gather_numvec(gpu.get(so_idx).unwrap());
        assert_eq!(so_cpu.len(), rows);
        assert_eq!(so_gpu.len(), rows);
        for (a, b) in so_cpu.iter().zip(so_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
fn reduction_sum_include_omit_dim1_dim2_gpu_cpu() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        // Program defines X so we can locate its var slot; we inject contents after compile
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            SI1 = sum(X, 1);
            SO1 = sum(X, 1, 'omitnan');
            SI2 = sum(X, 2);
            SO2 = sum(X, 2, 'omitnan');
            "#
        );

        let bytecode = compile_semantic(&source);

        // Determine slot for X (first StoreVar in the program)
        let x_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next()
            .unwrap_or(0);

        // Build deterministic host matrix with NaNs
        let mut data = vec![0.0f64; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                data[r + c * rows] = (c as f64) * 10.0 + (r as f64 + 1.0);
            }
        }
        for c in 0..cols {
            data[c * rows] = f64::NAN; // first row
        }
        for value in data.iter_mut().take(rows) {
            *value = f64::NAN; // first column
        }

        // Run CPU path with host tensor injected
        let cpu = {
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let t = runmat_value::Tensor::new(data.clone(), vec![rows, cols]).unwrap();
            vars[x_index] = Value::Tensor(t);
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Run GPU path with uploaded tensor injected
        let gpu = {
            ensure_provider_registered();
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &[rows, cols],
            };
            let provider = runmat_accelerate_api::provider().expect("provider");
            let handle = provider.upload(&view).expect("upload");
            vars[x_index] = Value::GpuTensor(handle);
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Find last 4 stores (SI1, SO1, SI2, SO2)
        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 4);
        let so2 = stores.pop().unwrap();
        let si2 = stores.pop().unwrap();
        let so1 = stores.pop().unwrap();
        let si1 = stores.pop().unwrap();

        let gather_vec = |v: &Value| -> Vec<f64> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                    Value::Tensor(t) => t.materialize_f64(),
                    Value::Num(n) => vec![n],
                    other => panic!("expected tensor, got {other:?}"),
                },
                Value::Tensor(t) => t.materialize_f64().clone(),
                Value::Num(n) => vec![*n],
                other => panic!("expected tensor, got {other:?}"),
            }
        };

        // Include-nan compare
        let si1_cpu = gather_vec(cpu.get(si1).unwrap());
        let si1_gpu = gather_vec(gpu.get(si1).unwrap());
        eprintln!("SI1 len: cpu={} gpu={}", si1_cpu.len(), si1_gpu.len());
        eprintln!("SI1 CPU: {:?}", si1_cpu);
        eprintln!("SI1 GPU: {:?}", si1_gpu);
        assert_eq!(si1_cpu.len(), cols);
        assert_eq!(si1_gpu.len(), cols);
        for (i, (a, b)) in si1_cpu.iter().zip(si1_gpu.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                if (a - b).abs() >= 1e-9 {
                    eprintln!(
                        "SI1 mismatch at col {}: cpu={} gpu={} rows={} cols={}",
                        i, a, b, rows, cols
                    );
                }
                assert!((a - b).abs() < 1e-9);
            }
        }

        let si2_cpu = gather_vec(cpu.get(si2).unwrap());
        let si2_gpu = gather_vec(gpu.get(si2).unwrap());
        eprintln!("SI2 len: cpu={} gpu={}", si2_cpu.len(), si2_gpu.len());
        eprintln!("SI2 CPU: {:?}", si2_cpu);
        eprintln!("SI2 GPU: {:?}", si2_gpu);
        assert_eq!(si2_cpu.len(), rows);
        assert_eq!(si2_gpu.len(), rows);
        for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }

        // Omit-nan compare
        let so1_cpu = gather_vec(cpu.get(so1).unwrap());
        let so1_gpu = gather_vec(gpu.get(so1).unwrap());
        eprintln!("SO1 len: cpu={} gpu={}", so1_cpu.len(), so1_gpu.len());
        eprintln!("SO1 CPU: {:?}", so1_cpu);
        eprintln!("SO1 GPU: {:?}", so1_gpu);
        assert_eq!(so1_cpu.len(), cols);
        assert_eq!(so1_gpu.len(), cols);
        for (i, (a, b)) in so1_cpu.iter().zip(so1_gpu.iter()).enumerate() {
            if (a - b).abs() >= 1e-9 {
                eprintln!("SO1 mismatch at col {}: cpu={} gpu={}", i, a, b);
            }
            assert!((a - b).abs() < 1e-9);
        }

        let so2_cpu = gather_vec(cpu.get(so2).unwrap());
        let so2_gpu = gather_vec(gpu.get(so2).unwrap());
        eprintln!("SO2 len: cpu={} gpu={}", so2_cpu.len(), so2_gpu.len());
        eprintln!("SO2 CPU: {:?}", so2_cpu);
        eprintln!("SO2 GPU: {:?}", so2_gpu);
        assert_eq!(so2_cpu.len(), rows);
        assert_eq!(so2_gpu.len(), rows);
        for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
fn reduction_sum_include_omit_dim1_dim2_degenerate_gpu_cpu() {
    gc_test_context(|| {
        let cases = vec![(1usize, 7usize), (8usize, 1usize)];
        for (rows, cols) in cases {
            let source = format!(
                r#"
                rows = {rows}; cols = {cols};
                X = zeros(rows, cols);
                SI1 = sum(X, 1);
                SO1 = sum(X, 1, 'omitnan');
                SI2 = sum(X, 2);
                SO2 = sum(X, 2, 'omitnan');
                "#
            );

            let bytecode = compile_semantic(&source);

            // Determine slot for X (first StoreVar in the program)
            let x_index = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(idx) => Some(*idx),
                    _ => None,
                })
                .next()
                .unwrap_or(0);

            // Build deterministic host matrix with NaNs on first row/col if present
            let mut data = vec![0.0f64; rows * cols];
            for c in 0..cols {
                for r in 0..rows {
                    data[r + c * rows] = (c as f64) * 10.0 + (r as f64 + 1.0);
                }
            }
            if rows > 0 {
                for c in 0..cols {
                    data[c * rows] = f64::NAN;
                }
            }
            if cols > 0 {
                for value in data.iter_mut().take(rows) {
                    *value = f64::NAN;
                }
            }

            // CPU path
            let cpu = {
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let t = runmat_value::Tensor::new(data.clone(), vec![rows, cols]).unwrap();
                vars[x_index] = Value::Tensor(t);
                interpret_function(&bytecode, vars).expect("interpret")
            };

            // GPU path
            let gpu = {
                ensure_provider_registered();
                let mut vars = vec![Value::Num(0.0); bytecode.var_count];
                let view = runmat_accelerate_api::HostTensorView {
                    data: &data,
                    shape: &[rows, cols],
                };
                let provider = runmat_accelerate_api::provider().expect("provider");
                let handle = provider.upload(&view).expect("upload");
                vars[x_index] = Value::GpuTensor(handle);
                interpret_function(&bytecode, vars).expect("interpret")
            };

            // Collect last 4 stores (SI1, SO1, SI2, SO2)
            let mut stores: Vec<usize> = bytecode
                .instructions
                .iter()
                .filter_map(|instr| match instr {
                    Instr::StoreVar(idx) => Some(*idx),
                    _ => None,
                })
                .collect();
            assert!(stores.len() >= 4);
            let so2 = stores.pop().unwrap();
            let si2 = stores.pop().unwrap();
            let so1 = stores.pop().unwrap();
            let si1 = stores.pop().unwrap();

            let gather_vec = |v: &Value| -> Vec<f64> {
                match v {
                    Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                        Value::Tensor(t) => t.materialize_f64(),
                        Value::Num(n) => vec![n],
                        other => panic!("expected tensor, got {other:?}"),
                    },
                    Value::Tensor(t) => t.materialize_f64().clone(),
                    Value::Num(n) => vec![*n],
                    other => panic!("expected tensor, got {other:?}"),
                }
            };

            // Compare include for both dims
            let si1_cpu = gather_vec(cpu.get(si1).unwrap());
            let si1_gpu = gather_vec(gpu.get(si1).unwrap());
            assert_eq!(si1_cpu.len(), cols);
            assert_eq!(si1_gpu.len(), cols);
            for (a, b) in si1_cpu.iter().zip(si1_gpu.iter()) {
                if a.is_nan() || b.is_nan() {
                    assert!(a.is_nan() && b.is_nan());
                } else {
                    assert!((a - b).abs() < 1e-9);
                }
            }
            let si2_cpu = gather_vec(cpu.get(si2).unwrap());
            let si2_gpu = gather_vec(gpu.get(si2).unwrap());
            assert_eq!(si2_cpu.len(), rows);
            assert_eq!(si2_gpu.len(), rows);
            for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
                if a.is_nan() || b.is_nan() {
                    assert!(a.is_nan() && b.is_nan());
                } else {
                    assert!((a - b).abs() < 1e-9);
                }
            }

            // Compare omit for both dims
            let so1_cpu = gather_vec(cpu.get(so1).unwrap());
            let so1_gpu = gather_vec(gpu.get(so1).unwrap());
            assert_eq!(so1_cpu.len(), cols);
            assert_eq!(so1_gpu.len(), cols);
            for (a, b) in so1_cpu.iter().zip(so1_gpu.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
            let so2_cpu = gather_vec(cpu.get(so2).unwrap());
            let so2_gpu = gather_vec(gpu.get(so2).unwrap());
            assert_eq!(so2_cpu.len(), rows);
            assert_eq!(so2_gpu.len(), rows);
            for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    });
}

#[test]
fn fused_elementwise_then_reduction_sum_dim1_dim2_include_gpu_cpu_small() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            Y = sin(X) + X + 2;    % elementwise producer
            SI1 = sum(Y, 1);
            SI2 = sum(Y, 2);
            "#
        );
        let bytecode = compile_semantic(&source);

        let cpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("cpu");
        ensure_provider_registered();
        let gpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("gpu");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 2);
        let si2 = stores.pop().unwrap();
        let si1 = stores.pop().unwrap();

        let gather_vec = |v: &Value| -> Vec<f64> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                    Value::Tensor(t) => t.materialize_f64(),
                    _ => panic!("expected tensor"),
                },
                Value::Tensor(t) => t.materialize_f64().clone(),
                _ => panic!("expected tensor"),
            }
        };

        let si1_cpu = gather_vec(cpu.get(si1).unwrap());
        let si1_gpu = gather_vec(gpu.get(si1).unwrap());
        assert_eq!(si1_cpu.len(), cols);
        assert_eq!(si1_gpu.len(), cols);
        for (a, b) in si1_cpu.iter().zip(si1_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }

        let si2_cpu = gather_vec(cpu.get(si2).unwrap());
        let si2_gpu = gather_vec(gpu.get(si2).unwrap());
        assert_eq!(si2_cpu.len(), rows);
        assert_eq!(si2_gpu.len(), rows);
        for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
fn fused_elementwise_then_reduction_sum_dim1_dim2_omit_gpu_cpu_small() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            Y = sin(X) + X + 2;
            % inject NaNs in first row/col
            for c = 1:cols, X(1, c) = NaN; end
            for r = 1:rows, X(r, 1) = NaN; end
            Z = Y + X;              % keep NaNs aligned with Y
            SO1 = sum(Z, 1, 'omitnan');
            SO2 = sum(Z, 2, 'omitnan');
            "#
        );
        let bytecode = compile_semantic(&source);

        let cpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("cpu");
        ensure_provider_registered();
        let gpu =
            interpret_function(&bytecode, vec![Value::Num(0.0); bytecode.var_count]).expect("gpu");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 2);
        let so2 = stores.pop().unwrap();
        let so1 = stores.pop().unwrap();

        let gather_vec = |v: &Value| -> Vec<f64> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).unwrap() {
                    Value::Tensor(t) => t.materialize_f64(),
                    _ => panic!("expected tensor"),
                },
                Value::Tensor(t) => t.materialize_f64().clone(),
                _ => panic!("expected tensor"),
            }
        };

        let so1_cpu = gather_vec(cpu.get(so1).unwrap());
        let so1_gpu = gather_vec(gpu.get(so1).unwrap());
        assert_eq!(so1_cpu.len(), cols);
        assert_eq!(so1_gpu.len(), cols);
        for (a, b) in so1_cpu.iter().zip(so1_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }

        let so2_cpu = gather_vec(cpu.get(so2).unwrap());
        let so2_gpu = gather_vec(gpu.get(so2).unwrap());
        assert_eq!(so2_cpu.len(), rows);
        assert_eq!(so2_gpu.len(), rows);
        for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}

#[test]
// keep: parity-focused provider check
fn provider_reduce_sum_dim_parity_simple() {
    gc_test_context(|| {
        ensure_provider_registered();
        let provider = runmat_accelerate_api::provider().expect("provider");

        let rows = 3usize;
        let cols = 4usize;
        // Column-major layout: data[r + c*rows]
        let mut data = vec![0.0f64; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                data[r + c * rows] = (c as f64) * 10.0 + (r as f64 + 1.0);
            }
        }
        // Insert NaNs: one in column 1, one in row 2
        data[rows] = f64::NAN; // (r=0,c=1)
        data[2 + 3 * rows] = f64::NAN; // (r=2,c=3)

        let view = runmat_accelerate_api::HostTensorView {
            data: &data,
            shape: &[rows, cols],
        };
        let gpu = provider.upload(&view).expect("upload");

        // CPU include-nan semantics
        let mut col_sums_cpu = vec![0.0f64; cols];
        for c in 0..cols {
            let mut acc = 0.0;
            let mut saw_nan = false;
            for r in 0..rows {
                let v = data[r + c * rows];
                if v.is_nan() {
                    saw_nan = true;
                    break;
                } else {
                    acc += v;
                }
            }
            col_sums_cpu[c] = if saw_nan { f64::NAN } else { acc };
        }

        let mut row_sums_cpu = vec![0.0f64; rows];
        for r in 0..rows {
            let mut acc = 0.0;
            let mut saw_nan = false;
            for c in 0..cols {
                let v = data[r + c * rows];
                if v.is_nan() {
                    saw_nan = true;
                    break;
                } else {
                    acc += v;
                }
            }
            row_sums_cpu[r] = if saw_nan { f64::NAN } else { acc };
        }

        // Provider dim=0 (MATLAB dim=1): column-wise
        let col_gpu = block_on(provider.reduce_sum_dim(&gpu, 0)).expect("reduce dim=0 (cols)");
        let host_col = block_on(provider.download(&col_gpu)).expect("download col");
        assert_eq!(host_col.shape, vec![1, cols]);
        assert_eq!(host_col.data.len(), cols);
        for (i, (a, b)) in col_sums_cpu.iter().zip(host_col.data.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(
                    a.is_nan() && b.is_nan(),
                    "col {}: cpu={:?} gpu={:?}",
                    i,
                    a,
                    b
                );
            } else {
                assert!((a - b).abs() < 1e-9, "col {}: {} vs {}", i, a, b);
            }
        }

        // Provider dim=1 (MATLAB dim=2): row-wise
        let row_gpu = block_on(provider.reduce_sum_dim(&gpu, 1)).expect("reduce dim=1 (rows)");
        let host_row = block_on(provider.download(&row_gpu)).expect("download row");
        assert_eq!(host_row.shape, vec![rows, 1]);
        assert_eq!(host_row.data.len(), rows);
        for (i, (a, b)) in row_sums_cpu.iter().zip(host_row.data.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(
                    a.is_nan() && b.is_nan(),
                    "row {}: cpu={:?} gpu={:?}",
                    i,
                    a,
                    b
                );
            } else {
                assert!((a - b).abs() < 1e-9, "row {}: {} vs {}", i, a, b);
            }
        }
    });
}

#[test]
fn fused_reduction_sum_dim1_dim2_include_gpu_cpu() {
    gc_test_context(|| {
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            Y = sin(X) .* X + 2;
            SI1 = sum(Y, 1);
            SI2 = sum(Y, 2);
            "#
        );

        let bytecode = compile_semantic(&source);

        let run_with = |use_gpu: bool| -> runmat_runtime::execution::RootedValues {
            let vars = vec![Value::Num(0.0); bytecode.var_count];
            if use_gpu {
                ensure_provider_registered();
            }
            interpret_function(&bytecode, vars).expect("interpret")
        };

        // Warmups
        let _ = run_with(false);
        let _ = run_with(true);

        let cpu = run_with(false);
        let gpu = run_with(true);

        // Helper: gather value to vec
        let _gather_vec = |v: &Value| -> Option<Vec<f64>> {
            match v {
                Value::GpuTensor(_) => match gather_if_needed(v).ok()? {
                    Value::Tensor(t) => Some(t.materialize_f64()),
                    Value::Num(n) => Some(vec![n]),
                    _ => None,
                },
                Value::Tensor(t) => Some(t.materialize_f64().clone()),
                Value::Num(n) => Some(vec![*n]),
                _ => None,
            }
        };

        // Robustly locate SI1 (1 x cols) and SI2 (rows x 1) by shape
        let find_by_shape =
            |vars: &[Value], want_rows: usize, want_cols: usize| -> Option<Vec<f64>> {
                for v in vars {
                    if let Ok(Value::Tensor(t)) = gather_if_needed(v) {
                        if t.shape().len() == 2
                            && t.shape()[0] == want_rows
                            && t.shape()[1] == want_cols
                        {
                            return Some(t.materialize_f64());
                        }
                    }
                }
                None
            };

        let si1_cpu = find_by_shape(&cpu, 1, cols).expect("cpu si1 1xcols");
        let si1_gpu = find_by_shape(&gpu, 1, cols).expect("gpu si1 1xcols");
        assert_eq!(si1_cpu.len(), cols);
        assert_eq!(si1_gpu.len(), cols);
        for (a, b) in si1_cpu.iter().zip(si1_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }

        let si2_cpu = find_by_shape(&cpu, rows, 1).expect("cpu si2 rowsx1");
        let si2_gpu = find_by_shape(&gpu, rows, 1).expect("gpu si2 rowsx1");
        assert_eq!(si2_cpu.len(), rows);
        assert_eq!(si2_gpu.len(), rows);
        for (a, b) in si2_cpu.iter().zip(si2_gpu.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-9);
            }
        }
    });
}

#[test]
fn fused_reduction_sum_dim1_dim2_omit_gpu_cpu() {
    gc_test_context(|| {
        let _accel_guard = accel_test_lock();
        let rows = 8usize;
        let cols = 7usize;
        let source = format!(
            r#"
            rows = {rows}; cols = {cols};
            X = zeros(rows, cols);
            % Insert NaNs in first row and first column
            for c = 1:cols
                X(1, c) = NaN;
            end
            for r = 1:rows
                X(r, 1) = NaN;
            end
            Y = sin(X) .* X + 2;
            SO1 = sum(Y, 1, 'omitnan');
            SO2 = sum(Y, 2, 'omitnan');
            "#
        );

        let bytecode = compile_semantic(&source);

        let x_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next()
            .unwrap_or(0);

        let mut data = vec![0.0f64; rows * cols];
        for c in 0..cols {
            data[c * rows] = f64::NAN;
        }
        for value in data.iter_mut().take(rows) {
            *value = f64::NAN;
        }

        let cpu = {
            runmat_accelerate_api::set_thread_provider(None);
            runmat_accelerate_api::clear_provider();
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let t = runmat_value::Tensor::new(data.clone(), vec![rows, cols]).unwrap();
            vars[x_index] = Value::Tensor(t);
            interpret_function(&bytecode, vars).expect("interpret")
        };

        let gpu = {
            ensure_provider_registered();
            let mut vars = vec![Value::Num(0.0); bytecode.var_count];
            let view = runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &[rows, cols],
            };
            let provider = runmat_accelerate_api::provider().expect("provider");
            let handle = provider.upload(&view).expect("upload");
            vars[x_index] = Value::GpuTensor(handle);
            interpret_function(&bytecode, vars).expect("interpret")
        };

        if std::env::var("RUNMAT_DUMP_OMITNAN_VALUES").as_deref() == Ok("1") {
            let dump = |vars: &[Value]| {
                vars.iter()
                    .enumerate()
                    .map(|(idx, v)| match gather_if_needed(v) {
                        Ok(Value::Tensor(t)) => format!("{idx}:tensor{:?}", t.shape()),
                        Ok(Value::Num(n)) => format!("{idx}:num({n})"),
                        Ok(other) => format!("{idx}:{other:?}"),
                        Err(err) => format!("{idx}:err({err})"),
                    })
                    .collect::<Vec<_>>()
            };
            panic!("cpu={:?} gpu={:?}", dump(&cpu), dump(&gpu));
        }

        let stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 2, "expected stores for SO1/SO2");
        let so2_idx = stores[stores.len() - 1];
        let so1_idx = stores[stores.len() - 2];

        let gather_numvec = |value: &Value| -> Vec<f64> {
            match value {
                Value::GpuTensor(_) => match gather_if_needed(value).expect("gather gpu tensor") {
                    Value::Tensor(t) => t.materialize_f64(),
                    other => panic!("expected gathered tensor, got {other:?}"),
                },
                Value::Tensor(t) => t.materialize_f64().clone(),
                other => panic!("expected tensor value, got {other:?}"),
            }
        };

        let so1_cpu = gather_numvec(cpu.get(so1_idx).expect("cpu so1 value"));
        let so1_gpu = gather_numvec(gpu.get(so1_idx).expect("gpu so1 value"));
        assert_eq!(so1_cpu.len(), cols);
        assert_eq!(so1_gpu.len(), cols);
        for (a, b) in so1_cpu.iter().zip(so1_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }

        let so2_cpu = gather_numvec(cpu.get(so2_idx).expect("cpu so2 value"));
        let so2_gpu = gather_numvec(gpu.get(so2_idx).expect("gpu so2 value"));
        assert_eq!(so2_cpu.len(), rows);
        assert_eq!(so2_gpu.len(), rows);
        for (a, b) in so2_cpu.iter().zip(so2_gpu.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    });
}
#[allow(dead_code)]
fn fused_elementwise_residency_and_gather() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [1, 2, 3];
        b = 2;
        y = sin(x) .* x + b;
        "#;

        let bytecode = compile_semantic(source);
        let vars = interpret(&bytecode).expect("interpret");

        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");

        let handle = match y_value {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        assert!(
            fusion_residency::is_resident(handle),
            "GPU handle should be marked resident before gather"
        );

        let gathered = gather_if_needed(y_value).expect("gather");
        let tensor = match gathered {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected gathered tensor, got {other:?}"),
        };
        assert!(
            !fusion_residency::is_resident(handle),
            "Residency should be cleared after gather"
        );

        let expected: Vec<f64> = [1.0f64, 2.0, 3.0]
            .iter()
            .map(|x| x.sin() * x + 2.0)
            .collect();
        assert_eq!(tensor.materialize_f64().len(), expected.len());
        for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}

#[test]
fn fused_literal_constant_and_extended_builtins() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [4, 5, 6];
        y = sin(x) .* x + 2;
        u = exp(x);
        v = sqrt(x);
        "#;

        let bytecode = compile_semantic(source);
        let vars = interpret(&bytecode).expect("interpret");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 4, "expected stores for x, y, u, v");
        let v_index = stores.pop().unwrap();
        let u_index = stores.pop().unwrap();
        let y_index = stores.pop().unwrap();

        let check_tensor = |value: &Value| -> Vec<f64> {
            match value {
                Value::GpuTensor(handle) => {
                    assert!(fusion_residency::is_resident(handle));
                    let gathered = gather_if_needed(value).expect("gather");
                    let tensor = match gathered {
                        Value::Tensor(tensor) => tensor,
                        other => panic!("expected gathered tensor, got {other:?}"),
                    };
                    assert!(
                        !fusion_residency::is_resident(handle),
                        "Residency should be cleared after gather"
                    );
                    tensor.materialize_f64()
                }
                Value::Tensor(tensor) => tensor.materialize_f64().clone(),
                other => panic!("expected tensor, got {other:?}"),
            }
        };

        let y_data = check_tensor(vars.get(y_index).expect("value for y"));
        let y_expected: Vec<f64> = [4.0f64, 5.0, 6.0]
            .iter()
            .map(|x| x.sin() * x + 2.0)
            .collect();
        assert_eq!(y_data, y_expected);

        let u_data = check_tensor(vars.get(u_index).expect("value for u"));
        let u_expected: Vec<f64> = [4.0f64, 5.0, 6.0].iter().map(|x| x.exp()).collect();
        assert_eq!(u_data, u_expected);

        let v_data = check_tensor(vars.get(v_index).expect("value for v"));
        let v_expected: Vec<f64> = [4.0f64, 5.0, 6.0].iter().map(|x| x.sqrt()).collect();
        assert_eq!(v_data, v_expected);
    });
}

#[test]
fn fused_heaviside_preserves_nan_and_zero_semantics() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [-2, -0, 0, 3, 0/0];
        y = heaviside(x) + 0;
        "#;

        let bytecode = compile_semantic(source);
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ElementwiseChain)),
                "expected elementwise fusion group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
            assert!(
                plan.groups.iter().any(|g| g.kernel.supported),
                "expected supported fusion plan, groups={:?} supported={:?}",
                &bytecode.fusion_groups,
                plan.groups
                    .iter()
                    .map(|g| g.kernel.supported)
                    .collect::<Vec<_>>()
            );
        }

        let vars = interpret(&bytecode).expect("interpret");
        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");
        if let Value::GpuTensor(handle) = y_value {
            assert!(fusion_residency::is_resident(handle));
        }

        let tensor = match gather_if_needed(y_value).expect("gather y") {
            Value::Tensor(tensor) => tensor,
            Value::GpuTensor(_) => panic!("expected gathered tensor after gather_if_needed"),
            other => panic!("expected gathered tensor, got {other:?}"),
        };

        assert_real_sequence_matches(&tensor.materialize_f64(), &[0.0, 0.5, 0.5, 1.0, f64::NAN]);
    });
}

#[test]
fn fused_safe_followup_builtins_remain_resident() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [-2.5, -1.25, 0.5];
        y = sign(x) + fix(x) + pow2(x) + hypot(x, 2);
        "#;

        let bytecode = compile_semantic(source);
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ElementwiseChain)),
                "expected elementwise fusion group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
            assert!(
                plan.groups.iter().any(|g| g.kernel.supported),
                "expected supported fusion plan, groups={:?} supported={:?}",
                &bytecode.fusion_groups,
                plan.groups
                    .iter()
                    .map(|g| g.kernel.supported)
                    .collect::<Vec<_>>()
            );
        }
        let vars = interpret(&bytecode).expect("interpret");

        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store var for y");

        let tensor =
            match gather_if_needed(vars.get(y_index).expect("value for y")).expect("gather y") {
                Value::Tensor(tensor) => tensor,
                Value::GpuTensor(_) => panic!("expected gathered tensor after gather_if_needed"),
                other => panic!("expected gathered tensor, got {other:?}"),
            };

        let expected: Vec<f64> = [-2.5f64, -1.25, 0.5]
            .iter()
            .map(|x| x.signum() + x.trunc() + x.exp2() + x.hypot(2.0))
            .collect();
        assert_eq!(tensor.materialize_f64().len(), expected.len());
        for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}

#[test]
fn fused_inverse_hyperbolic_builtins_are_plannable() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [0.25, 0.5, 0.75];
        y = asinh(x) + atanh(x) + acosh(x + 1);
        "#;

        let bytecode = compile_semantic(source);
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ElementwiseChain)),
                "expected elementwise fusion group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
            assert!(
                plan.groups.iter().any(|g| g.kernel.supported),
                "expected supported fusion plan, groups={:?} supported={:?}",
                &bytecode.fusion_groups,
                plan.groups
                    .iter()
                    .map(|g| g.kernel.supported)
                    .collect::<Vec<_>>()
            );
        }

        let vars = interpret(&bytecode).expect("interpret");
        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");
        if let Value::GpuTensor(handle) = y_value {
            assert!(fusion_residency::is_resident(handle));
        }

        let tensor = match gather_if_needed(y_value).expect("gather y") {
            Value::Tensor(tensor) => tensor,
            Value::GpuTensor(_) => panic!("expected gathered tensor after gather_if_needed"),
            other => panic!("expected gathered tensor, got {other:?}"),
        };

        let expected: Vec<f64> = [0.25f64, 0.5, 0.75]
            .iter()
            .map(|x| x.asinh() + x.atanh() + (x + 1.0).acosh())
            .collect();
        assert_eq!(tensor.materialize_f64().len(), expected.len());
        for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}

#[test]
fn direct_execution_of_safe_followup_group_returns_gpu_tensor() {
    gc_test_context(|| {
        ensure_provider_registered();
        let provider = registered_provider().expect("fixture provider");

        let source = r#"
        x = [-2.5, -1.25, 0.5];
        y = sign(x) + fix(x) + pow2(x) + hypot(x, 2);
        "#;

        let bytecode = compile_semantic(source);
        let graph = graph_for_fusion_test(&bytecode).expect("accel graph");
        let groups = graph.detect_fusion_groups();
        let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
        let group_plan = plan
            .groups
            .iter()
            .find(|g| matches!(g.group.kind, FusionKind::ElementwiseChain) && g.kernel.supported)
            .expect("supported elementwise fusion group");

        assert_eq!(
            group_plan.inputs.len(),
            2,
            "expected x and scalar runtime inputs"
        );

        let x = provider
            .push(vec![-2.5, -1.25, 0.5], vec![1, 3])
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit resident source");
        let result = block_on(execute_elementwise(FusionExecutionRequest {
            plan: group_plan,
            inputs: vec![Value::GpuTensor(x), Value::Num(2.0)],
            placement: None,
            runtime: Some(direct::selected_runtime_for_plan(group_plan)),
        }))
        .expect("direct fusion execution");

        match result.final_value {
            Value::GpuTensor(handle) => {
                assert!(fusion_residency::is_resident(&handle));
            }
            other => panic!("expected direct execution to return GPU tensor, got {other:?}"),
        }
    });
}

#[test]
fn direct_execution_of_inverse_hyperbolic_group_returns_gpu_tensor() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [0.25, 0.5, 0.75];
        y = asinh(x) + atanh(x) + acosh(x + 1);
        "#;

        let bytecode = compile_semantic(source);
        let graph = graph_for_fusion_test(&bytecode).expect("accel graph");
        let groups = graph.detect_fusion_groups();
        let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
        let group_plan = plan
            .groups
            .iter()
            .find(|g| matches!(g.group.kind, FusionKind::ElementwiseChain) && g.kernel.supported)
            .expect("supported elementwise fusion group");

        let x = Tensor::new(vec![0.25, 0.5, 0.75], vec![1, 3]).expect("tensor x");
        let result = block_on(execute_elementwise(FusionExecutionRequest {
            plan: group_plan,
            inputs: vec![Value::Tensor(x), Value::Num(1.0)],
            placement: None,
            runtime: Some(direct::selected_runtime_for_plan(group_plan)),
        }))
        .unwrap_or_else(|error| {
            panic!(
                "inverse fusion execution: {error}; plan={group_plan:#?}; placement={:#?}",
                runmat_accelerate::placement_report()
            )
        });

        match result.final_value {
            Value::GpuTensor(handle) => {
                assert!(fusion_residency::is_resident(&handle));
            }
            other => panic!("expected direct execution to return GPU tensor, got {other:?}"),
        }
    });
}

#[test]
fn explicit_gpu_inverse_hyperbolic_group_uses_owned_domain_probes() {
    gc_test_context(|| {
        ensure_provider_registered();
        let bytecode =
            compile_semantic("x = [0.25, 0.5, 0.75]; y = asinh(x) + atanh(x) + acosh(x + 1);");
        let graph = graph_for_fusion_test(&bytecode).expect("accel graph");
        let groups = graph.detect_fusion_groups();
        let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
        let group_plan = plan
            .groups
            .iter()
            .find(|group| {
                matches!(group.group.kind, FusionKind::ElementwiseChain) && group.kernel.supported
            })
            .expect("supported guarded fusion group");
        assert_eq!(group_plan.inputs.len(), 2);

        let provider = registered_provider().expect("registered fixture provider");
        let input = provider
            .push(vec![0.25, 0.5, 0.75], vec![1, 3])
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit fixture input");
        let trace = runmat_accelerate::placement::begin_trace("fusion.inverse_fixture", None);
        let result = block_on(execute_elementwise(FusionExecutionRequest {
            plan: group_plan,
            inputs: vec![Value::GpuTensor(input), Value::Num(1.0)],
            placement: Some(trace.correlation()),
            runtime: Some(direct::selected_runtime_for_plan(group_plan)),
        }));
        trace.complete(None, None);
        let result = result.unwrap_or_else(|error| {
            panic!(
                "guarded GPU fusion: {error}; placement={:#?}",
                runmat_accelerate::placement_report()
            )
        });
        let Value::GpuTensor(output) = result.final_value else {
            panic!("expected explicitly resident fused output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        let (actual, shape) = provider.pull(&output).expect("fixture output");
        assert_eq!(shape, vec![1, 3]);
        for (actual, input) in actual.into_iter().zip([0.25_f64, 0.5, 0.75]) {
            let expected = input.asinh() + input.atanh() + (input + 1.0).acosh();
            assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
        }
    });
}

#[test]
fn guarded_inverse_hyperbolic_fusion_defers_unsafe_or_unknown_gpu_inputs() {
    gc_test_context(|| {
        ensure_provider_registered();
        let bytecode =
            compile_semantic("x = [0.25, 0.5, 0.75]; y = asinh(x) + atanh(x) + acosh(x + 1);");
        let graph = graph_for_fusion_test(&bytecode).expect("accel graph");
        let groups = graph.detect_fusion_groups();
        let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
        let group_plan = plan
            .groups
            .iter()
            .find(|group| {
                matches!(group.group.kind, FusionKind::ElementwiseChain) && group.kernel.supported
            })
            .expect("guarded group");
        let provider = registered_provider().expect("fixture provider");
        for data in [vec![-1.25, 0.5, 0.75], vec![0.25, f64::NAN, 0.75], vec![]] {
            let input = provider.push(data.clone(), vec![1, data.len()]);
            let outcome = block_on(execute_elementwise_placed(FusionExecutionRequest {
                plan: group_plan,
                inputs: vec![Value::GpuTensor(input), Value::Num(1.0)],
                placement: None,
                runtime: Some(direct::selected_runtime_for_plan(group_plan)),
            }))
            .expect("unknown real domain is not a provider failure");
            assert!(matches!(outcome, FusionExecutionOutcome::SharedRuntime(_)));
        }
    });
}

#[test]
fn direct_execution_of_mod_and_rem_group_returns_gpu_tensor() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [-5.5, -1.25, 5.5];
        y = mod(x, 2) + rem(x, 2);
        "#;

        let bytecode = compile_semantic(source);
        let graph = graph_for_fusion_test(&bytecode).expect("accel graph");
        let groups = graph.detect_fusion_groups();
        let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
        let group_plan = plan
            .groups
            .iter()
            .find(|g| matches!(g.group.kind, FusionKind::ElementwiseChain) && g.kernel.supported)
            .expect("supported elementwise fusion group");

        let x = Tensor::new(vec![-5.5, -1.25, 5.5], vec![1, 3]).expect("tensor x");
        let result = block_on(execute_elementwise(FusionExecutionRequest {
            plan: group_plan,
            inputs: vec![Value::Tensor(x), Value::Num(2.0), Value::Num(2.0)],
            placement: None,
            runtime: None,
        }))
        .expect("direct fusion execution");

        match result.final_value {
            Value::GpuTensor(handle) => {
                assert!(fusion_residency::is_resident(&handle));
            }
            other => panic!("expected direct execution to return GPU tensor, got {other:?}"),
        }
    });
}

#[test]
fn fused_mod_and_rem_are_correct_under_local_placement() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [-5.5, -1.25, 5.5];
        y = mod(x, 2) + rem(x, 2);
        "#;

        let bytecode = compile_semantic(source);
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ElementwiseChain)),
                "expected elementwise fusion group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
            assert!(
                plan.groups.iter().any(|g| g.kernel.supported),
                "expected supported fusion plan, groups={:?} supported={:?}",
                &bytecode.fusion_groups,
                plan.groups
                    .iter()
                    .map(|g| g.kernel.supported)
                    .collect::<Vec<_>>()
            );
        }

        let vars = interpret(&bytecode).expect("interpret");
        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");
        if let Value::GpuTensor(handle) = y_value {
            assert!(fusion_residency::is_resident(handle));
        }

        let tensor = match gather_if_needed(y_value).expect("gather y") {
            Value::Tensor(tensor) => tensor,
            Value::GpuTensor(_) => panic!("expected gathered tensor after gather_if_needed"),
            other => panic!("expected gathered tensor, got {other:?}"),
        };

        let expected: Vec<f64> = [-5.5f64, -1.25, 5.5]
            .iter()
            .map(|x| x.rem_euclid(2.0) + (x - 2.0 * (x / 2.0).trunc()))
            .collect();
        assert_eq!(tensor.materialize_f64().len(), expected.len());
        for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}

#[test]
fn fused_mod_and_rem_broadcast_variants_are_correct_under_local_placement() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = [-5.5, -1.25, 5.5];
        d = [2, 3, 4];
        y = mod(x, d) + rem(6, d);
        "#;

        let bytecode = compile_semantic(source);
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ElementwiseChain)),
                "expected elementwise fusion group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
            assert!(
                plan.groups.iter().any(|g| g.kernel.supported),
                "expected supported fusion plan, groups={:?} supported={:?}",
                &bytecode.fusion_groups,
                plan.groups
                    .iter()
                    .map(|g| g.kernel.supported)
                    .collect::<Vec<_>>()
            );
        }

        let vars = interpret(&bytecode).expect("interpret");
        let y_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store var for y");

        let y_value = vars.get(y_index).expect("value for y");
        if let Value::GpuTensor(handle) = y_value {
            assert!(fusion_residency::is_resident(handle));
        }

        let tensor = match gather_if_needed(y_value).expect("gather y") {
            Value::Tensor(tensor) => tensor,
            Value::GpuTensor(_) => panic!("expected gathered tensor after gather_if_needed"),
            other => panic!("expected gathered tensor, got {other:?}"),
        };

        let xs = [-5.5f64, -1.25, 5.5];
        let ds = [2.0f64, 3.0, 4.0];
        let expected: Vec<f64> = xs
            .iter()
            .zip(ds.iter())
            .map(|(x, d)| x.rem_euclid(*d) + (6.0 - d * (6.0 / d).trunc()))
            .collect();
        assert_eq!(tensor.materialize_f64().len(), expected.len());
        for (actual, expect) in tensor.materialize_f64().iter().zip(expected.iter()) {
            assert!(
                (actual - expect).abs() < 1e-9,
                "mismatch: {actual} vs {expect}"
            );
        }
    });
}

fn mod_real_expected(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if b == 0.0 {
        return a;
    }
    if !a.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    let quotient = (a / b).floor();
    let mut remainder = a - b * quotient;
    if remainder == 0.0 {
        remainder = 0.0;
    }
    if b.is_infinite() && a.is_finite() {
        // MATLAB sign-correction: same-sign → a, different-sign → b (±Inf).
        if a == 0.0 {
            return 0.0;
        }
        return if a.signum() == b.signum() { a } else { b };
    }
    if !remainder.is_finite() && !a.is_finite() {
        return f64::NAN;
    }
    let same_sign = remainder == 0.0 || remainder.signum() == b.signum();
    if !same_sign {
        remainder += b;
    }
    if remainder == -0.0 {
        0.0
    } else {
        remainder
    }
}

fn rem_real_expected(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if b == 0.0 {
        return f64::NAN;
    }
    if !a.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    if b.is_infinite() && a.is_finite() {
        return if a == -0.0 { 0.0 } else { a };
    }
    let quotient = (a / b).trunc();
    if !quotient.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    let remainder = a - b * quotient;
    if remainder == -0.0 {
        0.0
    } else {
        remainder
    }
}

fn assert_real_sequence_matches(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN, got {actual}");
        } else {
            assert!(
                actual.to_bits() == expected.to_bits() || (actual - expected).abs() < 1e-9,
                "expected {expected}, got {actual}"
            );
        }
    }
}

fn compile_and_interpret_value(source: &str) -> (runmat_vm::Bytecode, Value) {
    let bytecode = compile_semantic(source);
    let vars = interpret(&bytecode).expect("interpret");
    let y_index = bytecode
        .instructions
        .iter()
        .filter_map(|instr| match instr {
            Instr::StoreVar(idx) => Some(*idx),
            _ => None,
        })
        .next_back()
        .expect("store var for y");
    (bytecode, vars.get(y_index).expect("value for y").clone())
}

#[test]
fn mod_real_parity_matrix_matches_expected_values() {
    gc_test_context(|| {
        ensure_provider_registered();

        struct Case {
            source: &'static str,
            expected: Vec<f64>,
            expect_resident: bool,
        }

        let cases = vec![
            Case {
                source: "y = mod(-5.5, 2) + 0;",
                expected: vec![mod_real_expected(-5.5, 2.0)],
                expect_resident: false,
            },
            Case {
                source: "x = [-5.5, -1.25, 5.5]; y = mod(x, 2) + 0;",
                expected: vec![
                    mod_real_expected(-5.5, 2.0),
                    mod_real_expected(-1.25, 2.0),
                    mod_real_expected(5.5, 2.0),
                ],
                expect_resident: true,
            },
            Case {
                source: "d = [2, 3, 4]; y = mod(6, d) + 0;",
                expected: vec![
                    mod_real_expected(6.0, 2.0),
                    mod_real_expected(6.0, 3.0),
                    mod_real_expected(6.0, 4.0),
                ],
                expect_resident: true,
            },
            Case {
                source: "x = [-5.5, -1.25, 5.5]; d = [2, 3, 4]; y = mod(x, d) + 0;",
                expected: vec![
                    mod_real_expected(-5.5, 2.0),
                    mod_real_expected(-1.25, 3.0),
                    mod_real_expected(5.5, 4.0),
                ],
                expect_resident: true,
            },
            Case {
                source: "x = [6, 6, 6, 0/0, 5, 1/0, 4]; d = [0, 2, 1/0, 2, 1/0, 2, 0]; y = mod(x, d) + 0;",
                expected: vec![
                    mod_real_expected(6.0, 0.0),
                    mod_real_expected(6.0, 2.0),
                    mod_real_expected(6.0, f64::INFINITY),
                    mod_real_expected(f64::NAN, 2.0),
                    mod_real_expected(5.0, f64::INFINITY),
                    mod_real_expected(f64::INFINITY, 2.0),
                    mod_real_expected(4.0, 0.0),
                ],
                expect_resident: true,
            },
            // (finite, -Inf) sign-correction: positive lhs → -Inf, negative lhs → -Inf matches.
            Case {
                source: "x = [5, -5, 5, -5, 0]; d = [-1/0, -1/0, 1/0, 1/0, -1/0]; y = mod(x, d) + 0;",
                expected: vec![
                    mod_real_expected(5.0, f64::NEG_INFINITY),
                    mod_real_expected(-5.0, f64::NEG_INFINITY),
                    mod_real_expected(5.0, f64::INFINITY),
                    mod_real_expected(-5.0, f64::INFINITY),
                    mod_real_expected(0.0, f64::NEG_INFINITY),
                ],
                expect_resident: true,
            },
        ];

        for case in cases {
            let (_, value) = compile_and_interpret_value(case.source);
            match &value {
                Value::GpuTensor(handle) if case.expect_resident => {
                    assert!(fusion_residency::is_resident(handle));
                }
                Value::GpuTensor(_) | Value::Tensor(_) | Value::Num(_) => {}
                other => panic!("unexpected result kind: {other:?}"),
            }
            let gathered = gather_if_needed(&value).expect("gather result");
            match gathered {
                Value::Tensor(tensor) => {
                    assert_real_sequence_matches(&tensor.materialize_f64(), &case.expected)
                }
                Value::Num(num) => assert_real_sequence_matches(&[num], &case.expected),
                other => panic!("expected real result, got {other:?}"),
            }
        }
    });
}

#[test]
fn rem_real_parity_matrix_matches_expected_values() {
    gc_test_context(|| {
        ensure_provider_registered();

        struct Case {
            source: &'static str,
            expected: Vec<f64>,
            expect_resident: bool,
        }

        let cases = vec![
            Case {
                source: "y = rem(-5.5, 2) + 0;",
                expected: vec![rem_real_expected(-5.5, 2.0)],
                expect_resident: false,
            },
            Case {
                source: "x = [-5.5, -1.25, 5.5]; y = rem(x, 2) + 0;",
                expected: vec![
                    rem_real_expected(-5.5, 2.0),
                    rem_real_expected(-1.25, 2.0),
                    rem_real_expected(5.5, 2.0),
                ],
                expect_resident: true,
            },
            Case {
                source: "d = [2, 3, 4]; y = rem(6, d) + 0;",
                expected: vec![
                    rem_real_expected(6.0, 2.0),
                    rem_real_expected(6.0, 3.0),
                    rem_real_expected(6.0, 4.0),
                ],
                expect_resident: true,
            },
            Case {
                source: "x = [-5.5, -1.25, 5.5]; d = [2, 3, 4]; y = rem(x, d) + 0;",
                expected: vec![
                    rem_real_expected(-5.5, 2.0),
                    rem_real_expected(-1.25, 3.0),
                    rem_real_expected(5.5, 4.0),
                ],
                expect_resident: true,
            },
            Case {
                source: "x = [6, 6, 6, 0/0, 5, 1/0, 4]; d = [0, 2, 1/0, 2, 1/0, 2, 0]; y = rem(x, d) + 0;",
                expected: vec![
                    rem_real_expected(6.0, 0.0),
                    rem_real_expected(6.0, 2.0),
                    rem_real_expected(6.0, f64::INFINITY),
                    rem_real_expected(f64::NAN, 2.0),
                    rem_real_expected(5.0, f64::INFINITY),
                    rem_real_expected(f64::INFINITY, 2.0),
                    rem_real_expected(4.0, 0.0),
                ],
                expect_resident: true,
            },
        ];

        for case in cases {
            let (_, value) = compile_and_interpret_value(case.source);
            match &value {
                Value::GpuTensor(handle) if case.expect_resident => {
                    assert!(fusion_residency::is_resident(handle));
                }
                Value::GpuTensor(_) | Value::Tensor(_) | Value::Num(_) => {}
                other => panic!("unexpected result kind: {other:?}"),
            }
            let gathered = gather_if_needed(&value).expect("gather result");
            match gathered {
                Value::Tensor(tensor) => {
                    assert_real_sequence_matches(&tensor.materialize_f64(), &case.expected)
                }
                Value::Num(num) => assert_real_sequence_matches(&[num], &case.expected),
                other => panic!("expected real result, got {other:?}"),
            }
        }
    });
}

#[test]
fn fused_reduction_scalar_report_tail_matches_expected() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = ones(1000000, 1, 'single');
        y = max(x - single(0.5), 0);
        z = mean(y, 'all') * exp(-single(0.1));
        s = char(num2str(double(z)));
        "#;

        let bytecode = compile_semantic(source);
        let vars = interpret(&bytecode).expect("interpret");

        let expected = 0.5f64 * (-0.1f64).exp();
        let tol = 1e-6;
        let rendered_value = vars
            .iter()
            .filter_map(|value| match value {
                Value::CharArray(chars) => runmat_value::decode_utf16_code_units(chars.data())
                    .expect("valid UTF-16 text fixture")
                    .trim()
                    .parse()
                    .ok(),
                _ => None,
            })
            .find(|value: &f64| (*value - expected).abs() <= tol)
            .expect("find rendered scalar report");
        let scalar_value = vars
            .iter()
            .filter_map(|value| match gather_if_needed(value).ok()? {
                Value::Num(n) => Some(n),
                Value::Tensor(tensor) if tensor.materialize_f64().len() == 1 => {
                    Some(tensor.materialize_f64()[0])
                }
                _ => None,
            })
            .find(|value| (*value - expected).abs() <= tol)
            .expect("find scalar z value");
        assert!(
            (rendered_value - expected).abs() <= tol,
            "rendered report mismatch: got {rendered_value}, expected {expected}"
        );
        assert!(
            (scalar_value - expected).abs() <= tol,
            "stored scalar mismatch: got {scalar_value}, expected {expected}"
        );
    });
}

#[test]
fn fused_reduction_scalar_single_report_tail_matches_expected() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = ones(1000000, 1, 'single');
        y = max(x - single(0.5), 0);
        z = mean(y, 'all') * exp(-single(0.1));
        z1 = single(z);
        s = char(num2str(double(z1)));
        "#;

        let bytecode = compile_semantic(source);
        let vars = interpret(&bytecode).expect("interpret");

        let expected = (0.5f64 * (-0.1f64).exp()) as f32 as f64;
        let tol = 1e-6;
        let rendered_value = vars
            .iter()
            .filter_map(|value| match value {
                Value::CharArray(chars) => runmat_value::decode_utf16_code_units(chars.data())
                    .expect("valid UTF-16 text fixture")
                    .trim()
                    .parse()
                    .ok(),
                _ => None,
            })
            .find(|value: &f64| (*value - expected).abs() <= tol)
            .expect("find rendered scalar report");
        let scalar_value = vars
            .iter()
            .filter_map(|value| match gather_if_needed(value).ok()? {
                Value::Num(n) => Some(n),
                Value::Tensor(tensor) if tensor.materialize_f64().len() == 1 => {
                    Some(tensor.materialize_f64()[0])
                }
                _ => None,
            })
            .find(|value| (*value - expected).abs() <= tol)
            .expect("find scalar z1 value");
        assert!(
            (rendered_value - expected).abs() <= tol,
            "rendered report mismatch: got {rendered_value}, expected {expected}"
        );
        assert!(
            (scalar_value - expected).abs() <= tol,
            "stored scalar mismatch: got {scalar_value}, expected {expected}"
        );
    });
}

#[test]
fn fused_reduction_scalar_branch_report_tail_matches_expected() {
    gc_test_context(|| {
        ensure_provider_registered();

        let source = r#"
        x = ones(1000000, 1, 'single');
        y = max(x - single(0.5), 0);
        z = mean(y, 'all') * exp(-single(0.1));
        if z > single(0)
            s = char(num2str(double(z)));
        else
            s = char('bad');
        end
        "#;

        let bytecode = compile_semantic(source);
        let vars = interpret(&bytecode).expect("interpret");

        let mut stores: Vec<usize> = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .collect();
        assert!(stores.len() >= 4, "expected stores for x, y, z, s");
        let s_index = stores.pop().expect("store index for s");

        let rendered = match vars.get(s_index).expect("value for s") {
            Value::CharArray(chars) => runmat_value::decode_utf16_code_units(chars.data())
                .expect("valid UTF-16 text fixture"),
            other => panic!("expected char array report, got {other:?}"),
        };
        let rendered_value: f64 = rendered.trim().parse().expect("parse rendered scalar");

        let expected = 0.5f64 * (-0.1f64).exp();
        let scalar_value = vars
            .iter()
            .filter_map(|value| match gather_if_needed(value).ok()? {
                Value::Num(n) => Some(n),
                Value::Tensor(tensor) if tensor.materialize_f64().len() == 1 => {
                    Some(tensor.materialize_f64()[0])
                }
                _ => None,
            })
            .find(|value| (*value - expected).abs() <= 1e-6)
            .expect("find scalar z value");
        let tol = 1e-6;
        assert!(
            (rendered_value - expected).abs() <= tol,
            "rendered report mismatch: got {rendered_value}, expected {expected}"
        );
        assert!(
            (scalar_value - expected).abs() <= tol,
            "stored scalar mismatch: got {scalar_value}, expected {expected}"
        );
    });
}

#[test]
fn original_centered_product_matches_cpu_with_provider_available() {
    gc_test_context(|| {
        let rows = 5usize;
        let cols = 3usize;
        let source = r#"
            rows = 5;
            A = [
                1.0, 2.0, 3.0;
                4.0, 5.0, 6.0;
                7.0, 8.0, 9.0;
                -1.0, 0.5, 2.0;
                3.0, -2.0, 4.0
            ];
            mu = mean(A, 1);
            centered = A - mu;
            cov = (centered.' * centered) / (rows - 1);
        "#;

        let bytecode = compile_semantic(source);
        let cov_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("cov store index");

        let vars = vec![Value::Num(0.0); bytecode.var_count];

        // Compute expected covariance on host
        let data: [[f64; 3]; 5] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [-1.0, 0.5, 2.0],
            [3.0, -2.0, 4.0],
        ];
        let mut means = vec![0.0f64; cols];
        for row in &data {
            for (c, value) in row.iter().enumerate() {
                means[c] += *value;
            }
        }
        for mean in &mut means {
            *mean /= rows as f64;
        }
        let mut cov_expected = vec![0.0f64; cols * cols];
        for row in &data {
            let mut centered = vec![0.0f64; cols];
            for (c, value) in row.iter().enumerate() {
                centered[c] = value - means[c];
            }
            for i in 0..cols {
                for j in 0..cols {
                    cov_expected[i * cols + j] += centered[i] * centered[j];
                }
            }
        }
        let denom = (rows as f64) - 1.0;
        for value in &mut cov_expected {
            *value /= denom;
        }

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let cov_gpu = vars_gpu.get(cov_index).expect("cov gpu");
        let gathered_gpu = gather_if_needed(cov_gpu).expect("gather gpu");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor cov (gpu), got {other:?}"),
        };

        assert_eq!(gpu_tensor.shape(), vec![cols, cols], "shape mismatch");
        let tol = 1e-6;
        for (lhs, rhs) in cov_expected.iter().zip(gpu_tensor.materialize_f64().iter()) {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "covariance mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
            );
        }
    });
}

#[test]
fn mean_all_gpu_matches_cpu() {
    gc_test_context(|| {
        let source = r#"
            A = gpuArray(single([1.0; 2.0; 3.0; 4.0]));
            mu = mean(A, 'all');
        "#;

        let bytecode = compile_semantic(source);

        let mu_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("mu store index");

        let vars = vec![Value::Num(0.0); bytecode.var_count];

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let mu_gpu = vars_gpu.get(mu_index).expect("mu result");
        let gathered = gather_if_needed(mu_gpu).expect("gather mu");
        let scalar = match gathered {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64().len(), 1, "expected scalar tensor");
                t.materialize_f64()[0]
            }
            Value::Num(n) => n,
            other => panic!("expected numeric result, got {other:?}"),
        };
        let expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        assert!(
            (scalar - expected).abs() <= 1e-6,
            "mean(all) mismatch: lhs={scalar}, rhs={expected}"
        );
    });
}

#[test]
fn power_step_normalization_matches_cpu() {
    gc_test_context(|| {
        use runmat_accelerate::FusionKind;
        let rows = 3usize;
        let cols = 2usize;
        let source = r#"
        G = [
            1.0, -0.5, 2.0;
            0.0, 1.5, -1.0;
            0.75, 0.25, 0.5
        ];
        Q = [
            2.0, -1.0;
            0.5, 3.0;
            -2.0, 1.0
        ];
        Q = mtimes(G, Q);
        norms = sqrt(sum(Q.^2, 1) + 1e-6);
        Q = Q ./ norms;
        "#;

        let bytecode = compile_semantic(source);

        {
            let graph = graph_for_fusion_test(&bytecode).expect("power-step acceleration graph");
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::PowerStepNormalize)),
                "expected power-step group, got {:?}",
                groups
            );
        }

        let q_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store index for Q");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let q_cpu = vars_cpu.get(q_index).expect("cpu Q");
        let gathered_cpu = gather_if_needed(q_cpu).expect("gather cpu");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor Q (cpu), got {other:?}"),
        };
        assert_eq!(cpu_tensor.shape(), vec![rows, cols], "cpu shape mismatch");

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars).expect("gpu interpret");
        let q_gpu = vars_gpu.get(q_index).expect("gpu Q");
        let gathered_gpu = gather_if_needed(q_gpu).expect("gather gpu");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor Q (gpu), got {other:?}"),
        };

        assert_eq!(cpu_tensor.shape(), gpu_tensor.shape(), "shape mismatch");
        let tol = 1e-6;
        for (lhs, rhs) in cpu_tensor
            .materialize_f64()
            .iter()
            .zip(gpu_tensor.materialize_f64().iter())
        {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "power-step mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
            );
        }
    });
}

#[test]
fn image_normalize_feasibility_admits_native_single_without_widening_other_fusion_routes() {
    let provider = TestProvider::new();
    let representation = runmat_accelerate_api::ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
        shape: vec![2, 4, 5],
        residency: runmat_accelerate_api::ProviderResidency::Host,
    };
    let mut query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(
            runmat_accelerate::placement::fusion_operation_token(&FusionKind::ImageNormalize),
        ),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![representation.clone()],
        outputs: vec![representation],
        workload: Default::default(),
    };
    assert!(provider.query_feasibility(&query).is_supported());

    for kind in [FusionKind::CenteredGram, FusionKind::ElementwiseChain] {
        query.operation = ProviderOperationIdentity::new(
            runmat_accelerate::placement::fusion_operation_token(&kind),
        );
        assert!(
            !provider.query_feasibility(&query).is_supported(),
            "native single inputs must not broaden {kind:?}"
        );
    }
}

#[test]
fn image_normalize_matches_cpu() {
    gc_test_context(|| {
        let _accel_guard = accel_test_lock();
        fusion_diagnostics::enable();
        use runmat_accelerate::FusionKind;
        let source = r#"
        B = 4; H = 8; W = 12;
        gain = single(1.0123);
        bias = single(-0.02);
        gamma = single(1.8);
        eps0 = single(1e-6);
        rng(0);
        imgs = single(rand(B, H, W));
        mu = mean(mean(imgs, 2), 3);
        sigma = sqrt(mean(mean((imgs - mu).^2, 2), 3) + eps0);
        out = ((imgs - mu) ./ sigma) * gain + bias;
        out = max(out, single(0));
        out = out .^ gamma;
        "#;

        let bytecode = compile_semantic(source);

        let graph = graph_for_fusion_test(&bytecode).expect("image normalize accel graph");
        let groups = graph.detect_fusion_groups();
        assert!(
            groups
                .iter()
                .any(|group| matches!(group.kind, FusionKind::ImageNormalize)),
            "nested image-normalization group not detected: {graph:#?}"
        );
        assert!(
            !bytecode
                .prospective_image_normalize_publications()
                .is_empty(),
            "nested image source must seal an executable VM publication region"
        );

        let out_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store index for out");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let out_cpu = vars_cpu.get(out_index).expect("cpu out");
        let gathered_cpu = gather_if_needed(out_cpu).expect("gather cpu out");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor out (cpu), got {other:?}"),
        };

        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars.clone()).expect("gpu interpret");
        let out_gpu = vars_gpu.get(out_index).expect("gpu out");
        let gathered_gpu = gather_if_needed(out_gpu).expect("gather gpu out");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor out (gpu), got {other:?}"),
        };

        assert_eq!(cpu_tensor.shape(), gpu_tensor.shape(), "shape mismatch");
        let tol = 5e-4;
        for (lhs, rhs) in cpu_tensor
            .materialize_f64()
            .iter()
            .zip(gpu_tensor.materialize_f64().iter())
        {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "image normalize mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
            );
        }
    });
}

#[test]
fn image_normalize_vecdim_unclamped_detects_fusion() {
    gc_test_context(|| {
        use runmat_accelerate::{FusionKind, FusionPattern};
        let source = r#"
        rng(0); B=2; H=4; W=5;
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

        imgs = rand(B, H, W, 'single');
        mu = mean(imgs, [2 3]);
        sigma = sqrt(mean((imgs - mu).^2, [2 3]) + eps0);
        out = ((imgs - mu) ./ sigma) * gain + bias;
        out = out .^ gamma;
        mse = mean((out - imgs).^2, 'all');

        fprintf('RESULT_ok MSE=%.6e\n', double(mse));
        "#;

        let bytecode = compile_semantic(source);

        let graph = graph_for_fusion_test(&bytecode).expect("vecdim image normalize accel graph");
        let groups = graph.detect_fusion_groups();
        let group = groups
            .iter()
            .find(|group| matches!(group.kind, FusionKind::ImageNormalize))
            .expect("image normalize group not detected for vecdim form");
        match group.pattern.as_ref() {
            Some(FusionPattern::ImageNormalize(pattern)) => {
                assert!(
                    !pattern.clamp_zero,
                    "exact unclamped script should not set clamp_zero"
                );
                assert!(!pattern.requires_single_input);
            }
            other => panic!("missing image normalize pattern, got {other:?}"),
        }
    });
}

#[test]
fn image_normalize_vecdim_double_reduction_does_not_claim_native_fusion() {
    gc_test_context(|| {
        let source = r#"
        imgs = rand(2, 4, 5, 'single');
        mu = mean(imgs, [2 3], 'double');
        sigma = sqrt(mean((imgs - mu).^2, [2 3], 'double') + single(1e-6));
        out = ((imgs - mu) ./ sigma) .^ single(1.8);
        "#;
        let bytecode = compile_semantic(source);
        assert!(
            bytecode
                .runtime_fusion_groups()
                .iter()
                .all(|group| !matches!(group.kind, FusionKind::ImageNormalize)),
            "double-output mean cannot use the native-precision image kernel"
        );
        // A program without an eligible group has no reconciled graph. Check
        // detection too when another eligible group keeps the graph alive.
        if let Some(graph) = graph_for_fusion_test(&bytecode) {
            assert!(graph
                .detect_fusion_groups()
                .iter()
                .all(|group| !matches!(group.kind, FusionKind::ImageNormalize)));
        }
    });
}

#[test]
fn image_normalize_vecdim_matches_cpu() {
    gc_test_context(|| {
        let _accel_guard = accel_test_lock();
        use runmat_accelerate::{FusionKind, FusionPattern};
        let source = r#"
        rng(0); B=2; H=4; W=5;
        gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

        imgs = rand(B, H, W, 'single');
        mu = single(mean(imgs, [2 3], 'native'));
        sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));
        out = single(((imgs - mu) ./ sigma) * gain + bias);
        out = max(out, single(0));
        out = single(out .^ gamma);
        err = out - imgs;
        mse = mean(err .* err, 'all');

        fprintf('RESULT_ok MSE=%.6e\n', double(mse));
        "#;

        let bytecode = compile_semantic(source);

        let graph = graph_for_fusion_test(&bytecode).expect("vecdim image normalize accel graph");
        let groups = graph.detect_fusion_groups();
        let group = groups
            .iter()
            .find(|group| matches!(group.kind, FusionKind::ImageNormalize))
            .unwrap_or_else(|| {
                panic!("image normalize group not detected for vecdim clamped form: {graph:#?}")
            });
        match group.pattern.as_ref() {
            Some(FusionPattern::ImageNormalize(pattern)) => {
                assert!(
                    pattern.clamp_zero,
                    "clamped vecdim script should set clamp_zero"
                );
                assert!(pattern.requires_single_input);
            }
            other => panic!("missing image normalize pattern, got {other:?}"),
        }

        let out_index = bytecode
            .var_names
            .iter()
            .find_map(|(idx, name)| (name == "out").then_some(*idx))
            .expect("out variable index");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let out_cpu = vars_cpu.get(out_index).expect("cpu out");
        let gathered_cpu = gather_if_needed(out_cpu).expect("gather cpu out");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor out (cpu), got {other:?}"),
        };
        assert_eq!(cpu_tensor.numeric_dtype(), runmat_value::NumericDType::F32);

        runmat_accelerate::placement::reset();
        ensure_provider_registered();
        let vars_gpu = interpret_function(&bytecode, vars.clone()).expect("gpu interpret");
        let out_gpu = vars_gpu.get(out_index).expect("gpu out");
        let Value::GpuTensor(output_handle) = out_gpu else {
            panic!(
                "vecdim image normalize should retain a resident native single output; got {out_gpu:?}; placement={}",
                serde_json::to_string_pretty(&runmat_accelerate::placement_report()).unwrap()
            );
        };
        assert_eq!(
            output_handle.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        let gathered_gpu = gather_if_needed(out_gpu).expect("gather gpu out");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor out (gpu), got {other:?}"),
        };

        assert_eq!(gpu_tensor.numeric_dtype(), runmat_value::NumericDType::F32);

        assert_eq!(cpu_tensor.shape(), gpu_tensor.shape(), "shape mismatch");
        let tol = 5e-4;
        for (lhs, rhs) in cpu_tensor
            .materialize_f64()
            .iter()
            .zip(gpu_tensor.materialize_f64().iter())
        {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "image normalize NaN mismatch: lhs={lhs}, rhs={rhs}"
                );
            } else {
                let diff = (lhs - rhs).abs();
                assert!(
                    diff <= tol,
                    "image normalize vecdim mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
                );
            }
        }
    });
}

#[test]
fn explained_variance_matches_cpu() {
    gc_test_context(|| {
        use runmat_accelerate::FusionKind;
        let source = r#"
        rng(0);
        rows = 4;
        cols = 2;
        G = rand(rows, rows);
        Q = rand(rows, cols);
        tmp = mtimes(Q.', G);
        prod = mtimes(tmp, Q);
        eval = diag(prod);
        "#;

        let bytecode = compile_semantic(source);

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (pc, instr) in bytecode.instructions.iter().enumerate() {
                eprintln!("instr {pc}: {instr:?}");
            }
        }

        let (q_var_idx, g_var_idx) = {
            let graph =
                graph_for_fusion_test(&bytecode).expect("explained-variance acceleration graph");
            let groups = graph.detect_fusion_groups();
            assert!(
                groups
                    .iter()
                    .any(|g| matches!(g.kind, FusionKind::ExplainedVariance)),
                "expected explained variance group, got {:?}",
                groups.iter().map(|g| g.kind.clone()).collect::<Vec<_>>()
            );
            let plan = runmat_accelerate::FusionPlan::from_graph(&graph, &groups);
            let explained = plan
                .groups
                .iter()
                .find(|g| matches!(g.group.kind, FusionKind::ExplainedVariance))
                .expect("explained variance plan");
            let (q_vid, g_vid) = match explained.pattern.as_ref() {
                Some(runmat_accelerate::fusion::FusionPattern::ExplainedVariance { q, g }) => {
                    (*q, *g)
                }
                other => panic!("unexpected pattern {other:?}"),
            };
            let q_binding = graph
                .var_binding(q_vid)
                .expect("q binding available for explained variance");
            let g_binding = graph
                .var_binding(g_vid)
                .expect("g binding available for explained variance");
            (q_binding.index, g_binding.index)
        };

        let eval_index = bytecode
            .instructions
            .iter()
            .filter_map(|instr| match instr {
                Instr::StoreVar(idx) => Some(*idx),
                _ => None,
            })
            .next_back()
            .expect("store index for eval");

        let vars = vec![Value::Num(0.0); bytecode.var_count];
        let vars_cpu = interpret_function(&bytecode, vars.clone()).expect("cpu interpret");
        let eval_cpu = vars_cpu.get(eval_index).expect("cpu eval");
        let gathered_cpu = gather_if_needed(eval_cpu).expect("gather cpu");
        let cpu_tensor = match gathered_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eval (cpu), got {other:#?}"),
        };

        let gather_tensor = |value: &Value| -> Tensor {
            match gather_if_needed(value).expect("gather tensor") {
                Value::Tensor(t) => t,
                other => panic!("expected tensor, got {other:#?}"),
            }
        };

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("q_var_idx {} g_var_idx {}", q_var_idx, g_var_idx);
            println!(
                "q tensor shape {:?}",
                gather_tensor(
                    vars_cpu
                        .get(q_var_idx)
                        .expect("q var present after cpu run")
                )
                .shape()
            );
            println!(
                "g tensor shape {:?}",
                gather_tensor(
                    vars_cpu
                        .get(g_var_idx)
                        .expect("g var present after cpu run")
                )
                .shape()
            );
            println!(
                "q tensor data {:?}",
                gather_tensor(
                    vars_cpu
                        .get(q_var_idx)
                        .expect("q var present after cpu run")
                )
                .materialize_f64()
            );
            println!(
                "g tensor data {:?}",
                gather_tensor(
                    vars_cpu
                        .get(g_var_idx)
                        .expect("g var present after cpu run")
                )
                .materialize_f64()
            );
        }

        let q_tensor = gather_tensor(
            vars_cpu
                .get(q_var_idx)
                .expect("q var present after cpu run"),
        );
        let g_tensor = gather_tensor(
            vars_cpu
                .get(g_var_idx)
                .expect("g var present after cpu run"),
        );
        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("q tensor shape {:?}", q_tensor.shape());
            println!("g tensor shape {:?}", g_tensor.shape());
            println!("q tensor data {:?}", q_tensor.materialize_f64());
            println!("g tensor data {:?}", g_tensor.materialize_f64());
        }

        let rows = q_tensor.shape().first().copied().unwrap_or(0);
        let cols = q_tensor.shape().get(1).copied().unwrap_or(1);
        assert!(
            rows > 0 && cols > 0,
            "explained variance requires non-empty Q"
        );

        // Compute tmp = Q.' * G with correct column-major transpose semantics.
        let mut tmp = vec![0.0; cols * rows];
        for j in 0..rows {
            for i in 0..cols {
                let mut acc = 0.0;
                for k in 0..rows {
                    let q_ki = q_tensor.materialize_f64()[k + i * rows];
                    let g_kj = g_tensor.materialize_f64()[k + j * rows];
                    acc += q_ki * g_kj;
                }
                tmp[i + cols * j] = acc;
            }
        }

        let mut expected_diag = Vec::with_capacity(cols);
        for i in 0..cols {
            let mut acc = 0.0;
            for j in 0..rows {
                let tmp_ij = tmp[i + cols * j];
                let q_ji = q_tensor.materialize_f64()[j + rows * i];
                acc += tmp_ij * q_ji;
            }
            expected_diag.push(acc);
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("tmp runtime shape {:?}", vec![cols, rows]);
            println!("tmp expected data {:?}", tmp);
            println!("expected diag {:?}", expected_diag);
            println!("cpu diag {:?}", cpu_tensor.materialize_f64());
            let tmp_idx = 4;
            if let Some(tmp_value) = vars_cpu.get(tmp_idx) {
                if let Ok(Value::Tensor(tmp_tensor)) = gather_if_needed(tmp_value) {
                    println!(
                        "tmp var idx {tmp_idx} shape {:?} data {:?}",
                        tmp_tensor.shape(),
                        tmp_tensor.materialize_f64()
                    );
                }
            }
            let prod_idx = 5;
            if let Some(prod_value) = vars_cpu.get(prod_idx) {
                if let Ok(Value::Tensor(prod_tensor)) = gather_if_needed(prod_value) {
                    println!(
                        "prod var idx {prod_idx} shape {:?} data {:?}",
                        prod_tensor.shape(),
                        prod_tensor.materialize_f64()
                    );
                }
            }
        }
        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (lhs, rhs) in expected_diag
                .iter()
                .zip(cpu_tensor.materialize_f64().iter())
            {
                println!("cpu diag diff {}", (lhs - rhs).abs());
            }
        } else {
            for (lhs, rhs) in expected_diag
                .iter()
                .zip(cpu_tensor.materialize_f64().iter())
            {
                assert!((lhs - rhs).abs() <= 1e-9, "cpu diag mismatch");
            }
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (idx, value) in vars_cpu.iter().enumerate() {
                if let Ok(Value::Tensor(t)) =
                    futures::executor::block_on(gather_if_needed_async(value))
                {
                    if t.shape() == vec![2, 2] {
                        println!(
                            "tensor idx {idx} shape {:?} data {:?}",
                            t.shape(),
                            t.materialize_f64()
                        );
                    }
                }
            }
        }

        ensure_provider_registered();
        let vars_gpu = vars_cpu
            .with_unbounded_values(|vars| interpret_function(&bytecode, vars))
            .and_then(std::convert::identity)
            .expect("gpu interpret");
        let eval_gpu = vars_gpu.get(eval_index).expect("gpu eval");
        assert!(matches!(eval_gpu, Value::GpuTensor(_) | Value::Tensor(_)));
        let gathered_gpu = gather_if_needed(eval_gpu).expect("gather gpu");
        let gpu_tensor = match gathered_gpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eval (gpu), got {other:#?}"),
        };

        assert_eq!(cpu_tensor.shape(), gpu_tensor.shape(), "shape mismatch");
        let tol = 1e-6;
        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            for (lhs, rhs) in expected_diag
                .iter()
                .zip(gpu_tensor.materialize_f64().iter())
            {
                let diff = (lhs - rhs).abs();
                println!("gpu diag diff lhs={lhs}, rhs={rhs}, diff={diff}");
            }
        } else {
            for (lhs, rhs) in expected_diag
                .iter()
                .zip(gpu_tensor.materialize_f64().iter())
            {
                let diff = (lhs - rhs).abs();
                assert!(
                    diff <= tol,
                    "explained variance mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
                );
            }
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            println!("Q tensor data {:?}", q_tensor.materialize_f64());
            println!("G tensor data {:?}", g_tensor.materialize_f64());
        }

        if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
            let q_tensor_value = Value::Tensor(q_tensor.clone());
            let g_tensor_value = Value::Tensor(g_tensor.clone());
            let q_t_value =
                runmat_runtime::call_builtin("transpose", std::slice::from_ref(&q_tensor_value))
                    .expect("transpose");
            if let Value::Tensor(q_t_tensor) = q_t_value.value() {
                println!("q_t tensor data {:?}", q_t_tensor.materialize_f64());
            }
            let tmp_via_runtime = futures::executor::block_on(runmat_runtime::value_matmul(
                &q_t_value,
                &g_tensor_value,
            ))
            .expect("q' * g via runtime");
            if let Value::Tensor(ref tmp_tensor) = tmp_via_runtime {
                println!("tmp via runtime {:?}", tmp_tensor.materialize_f64());
            }
        }
    });
}
