use super::*;
use crate::fusion::image_reduction::{
    ImageMeanProviderCall, ImageMeanProviderHook, ImageMeanProviderPlan, ImageSourcePower,
};
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    unsupported_provider_operation, AccelDownloadFuture, AccelProviderFuture, HostTensorView,
    NumericalDomainEvidence, PowerResultDomain, ProviderCapabilityOperation,
    ProviderCapabilitySnapshot, ProviderExecutedOrderedRealMean, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderOrderedRealMean, ProviderOrderedRealMeanControl, ProviderOrderedRealMeanLimits,
    ProviderOrderedRealMeanRecipe, ProviderOrderedRealMeanResources, ProviderOrderedRealMeanSource,
    ProviderOrderedRealMeanStep, ProviderRealMeanNaNMode, ProviderRejectionCode,
    ProviderRepresentation, ProviderResourceEstimate, ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct StagedProvider {
    precision: ProviderPrecision,
    controls: Mutex<Vec<runmat_accelerate_api::PowerScalarControl>>,
    uploaded_shapes: Mutex<Vec<Vec<usize>>>,
    allocations: TestAllocationOwner,
    foreign_allocations: TestAllocationOwner,
    probes: Mutex<Vec<RetirementProbe>>,
    trace: Mutex<Vec<StageHook>>,
    calls: AtomicUsize,
    cache_reservations: AtomicUsize,
    fail_at: Option<usize>,
    pending_at: Option<usize>,
    reserved_mu: Mutex<Option<(GpuTensorHandle, GpuTensorHandle)>>,
    fault: Mutex<Option<(usize, StageFault)>>,
    invalidate_on_call: Mutex<Option<(usize, RetirementProbe)>>,
    mutate_cast_residency: Mutex<bool>,
    mutate_protected_residency: Mutex<Option<GpuTensorHandle>>,
    ordered_completion_fault: Mutex<Option<OrderedCompletionFault>>,
    ordered_output: Mutex<Option<GpuTensorHandle>>,
    alias_same_class_cast: Mutex<bool>,
    reject_cast_query: Mutex<bool>,
    reject_cast_query_after_cast: Mutex<bool>,
    numeric_output_bytes_override: Mutex<Option<u64>>,
    reject_numeric_output_projection: Mutex<bool>,
    real_fault: Mutex<Option<(bool, real_elementwise_provider::RealFault)>>,
    real_controls: Mutex<Vec<runmat_accelerate_api::ProviderRealElementwiseOperation>>,
    binary_fault: Mutex<Option<real_binary_provider::BinaryFault>>,
}

#[derive(Clone)]
enum StageFault {
    WrongShape,
    WrongClass,
    WrongOwner,
    Alias(GpuTensorHandle),
}

#[derive(Clone, Copy)]
enum OrderedCompletionFault {
    RequestedPeak,
    ObservedPeak,
    OutputBytes,
    Submissions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StageHook {
    Nd(Vec<usize>),
    Dim(usize),
    Upload,
    Sub,
    Pow,
    Div,
    Add,
    Mul,
    Max,
    Single,
    Double,
    Sqrt,
}

fn provider(fail_at: Option<usize>) -> &'static StagedProvider {
    provider_with(fail_at, None)
}

fn provider_with(fail_at: Option<usize>, pending_at: Option<usize>) -> &'static StagedProvider {
    provider_with_precision(fail_at, pending_at, ProviderPrecision::F32)
}

fn provider_with_precision(
    fail_at: Option<usize>,
    pending_at: Option<usize>,
    precision: ProviderPrecision,
) -> &'static StagedProvider {
    crate::ensure_residency_hooks();
    Box::leak(Box::new(StagedProvider {
        precision,
        controls: Mutex::new(Vec::new()),
        uploaded_shapes: Mutex::new(Vec::new()),
        allocations: TestAllocationOwner::new(),
        foreign_allocations: TestAllocationOwner::new(),
        probes: Mutex::new(Vec::new()),
        trace: Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
        cache_reservations: AtomicUsize::new(0),
        fail_at,
        pending_at,
        reserved_mu: Mutex::new(None),
        fault: Mutex::new(None),
        invalidate_on_call: Mutex::new(None),
        mutate_cast_residency: Mutex::new(false),
        mutate_protected_residency: Mutex::new(None),
        ordered_completion_fault: Mutex::new(None),
        ordered_output: Mutex::new(None),
        alias_same_class_cast: Mutex::new(false),
        reject_cast_query: Mutex::new(false),
        reject_cast_query_after_cast: Mutex::new(false),
        numeric_output_bytes_override: Mutex::new(None),
        reject_numeric_output_projection: Mutex::new(false),
        real_fault: Mutex::new(None),
        real_controls: Mutex::new(Vec::new()),
        binary_fault: Mutex::new(None),
    }))
}

impl StagedProvider {
    fn produce(&self, shape: Vec<usize>, hook: StageHook) -> anyhow::Result<GpuTensorHandle> {
        self.produce_class(shape, hook, NumericElementType::F32)
    }

    fn produce_class(
        &self,
        shape: Vec<usize>,
        hook: StageHook,
        element: NumericElementType,
    ) -> anyhow::Result<GpuTensorHandle> {
        let call = self.begin_stage(hook);
        self.produce_class_at(shape, element, call)
    }

    fn begin_stage(&self, hook: StageHook) -> usize {
        self.trace.lock().unwrap().push(hook);
        self.calls.fetch_add(1, Ordering::SeqCst)
    }

    fn produce_optional(
        &self,
        shape: Vec<usize>,
        hook: StageHook,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        let call = self.begin_stage(hook);
        // This is the optional hook's deliberate availability injection, before
        // allocation, borrowed-resource mutation, or any production attempt.
        if self.fail_at == Some(call) {
            return Ok(None);
        }
        self.produce_class_at(shape, NumericElementType::F32, call)
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
    }

    fn produce_class_at(
        &self,
        shape: Vec<usize>,
        element: NumericElementType,
        call: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        if let Some((at, probe)) = self.invalidate_on_call.lock().unwrap().clone() {
            if call == at {
                probe.mark_unavailable();
            }
        }
        if self.fail_at == Some(call) {
            return Err(unsupported_provider_operation(
                "selected stage hook unavailable",
            ));
        }
        if let Some((at, fault)) = self.fault.lock().unwrap().clone() {
            if call == at {
                let faulty = match fault {
                    StageFault::WrongShape => {
                        let mut wrong = shape;
                        wrong[1] += 1;
                        self.allocations.allocate_zeroed(
                            wrong,
                            NumericElementType::F32,
                            GpuTensorStorage::Real,
                        )?
                    }
                    StageFault::WrongClass => self.allocations.allocate_zeroed(
                        shape,
                        if element == NumericElementType::F32 {
                            NumericElementType::F64
                        } else {
                            NumericElementType::F32
                        },
                        GpuTensorStorage::Real,
                    )?,
                    StageFault::WrongOwner => self.foreign_allocations.allocate_zeroed(
                        shape,
                        NumericElementType::F32,
                        GpuTensorStorage::Real,
                    )?,
                    StageFault::Alias(handle) => return Ok(handle),
                };
                self.probes.lock().unwrap().push(faulty.probe);
                return Ok(faulty.handle);
            }
        }
        let output = self
            .allocations
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)?;
        self.probes.lock().unwrap().push(output.probe);
        Ok(output.handle)
    }

    fn mutate_cast_metadata(&self, input: &GpuTensorHandle) {
        if *self.mutate_cast_residency.lock().unwrap() {
            runmat_accelerate_api::clear_residency(input)
                .expect("test cast clears borrowed residency metadata");
            assert!(!crate::fusion_residency::is_resident(input));
        }
    }

    fn probes(&self) -> Vec<RetirementProbe> {
        self.probes.lock().unwrap().clone()
    }

    fn trace(&self) -> Vec<StageHook> {
        self.trace.lock().unwrap().clone()
    }
}

impl AccelProvider for StagedProvider {
    fn numeric_upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence> {
        self.staged_numeric_upload_evidence(query)
    }

    fn floating_cast_physical_evidence(
        &self,
        input: &ProviderRepresentation,
        target: ProviderPrecision,
    ) -> Option<runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence> {
        self.staged_floating_cast_evidence(input, target)
    }

    fn propose_power_pair_geometry(
        &self,
        inputs: &[ProviderRepresentation; 2],
        descriptions: &[runmat_accelerate_api::PowerInputDescription; 2],
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerPairPhysicalProjection>,
    > {
        self.staged_power_pair_geometry(inputs, descriptions, frozen)
    }

    fn propose_power_probe(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<
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
        Box::pin(async move {
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            self.staged_power_probe_execute(recipe)
        })
    }

    fn propose_power_evaluation(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: PowerResultDomain,
    ) -> ProviderExecutionResult<
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
    ) -> ProviderExecutionResult<Option<runmat_accelerate_api::ProviderRealBinaryPhysicalProposal>>
    {
        self.binary_proposal(request)
    }

    fn execute_real_binary_recipe<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        recipe: &'a runmat_accelerate_api::ProviderRealBinaryRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealBinary,
    > {
        self.binary_execution(lhs, rhs, recipe)
    }
    fn propose_real_elementwise(
        &self,
        request: &runmat_accelerate_api::ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<
        Option<runmat_accelerate_api::ProviderRealElementwisePhysicalProposal>,
    > {
        self.real_proposal(request)
    }

    fn execute_real_elementwise_recipe<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        recipe: &'a runmat_accelerate_api::ProviderRealElementwiseRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealElementwise,
    > {
        self.real_execution(input, recipe)
    }
    fn plan_dense_real_output_buffer_bytes(
        &self,
        output: &ProviderRepresentation,
    ) -> ProviderExecutionResult<Option<u64>> {
        if *self.reject_numeric_output_projection.lock().unwrap()
            || output.storage != runmat_accelerate_api::ProviderStorage::DenseReal
            || output.layout != runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous
            || output.residency != runmat_accelerate_api::ProviderResidency::Device
        {
            return Ok(None);
        }
        Ok((*self.numeric_output_bytes_override.lock().unwrap())
            .or_else(|| output.checked_byte_len()))
    }

    fn plan_power_output_buffer_bytes(
        &self,
        planned: &runmat_accelerate_api::power_service::PowerSymbolicPlan,
        domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<u64>> {
        Ok(planned
            .physical_work(domain)
            .and_then(|work| work.output.checked_byte_len()))
    }

    fn plan_ordered_real_mean(
        &self,
        source: &ProviderRepresentation,
        source_buffer_bytes: u64,
        axes_zero_based: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<ProviderOrderedRealMeanRecipe>> {
        if self.precision != ProviderPrecision::F64
            || nan_mode != ProviderRealMeanNaNMode::Include
            || axes_zero_based.is_empty()
            || axes_zero_based
                .iter()
                .any(|&axis| axis >= source.shape.len())
            || source
                .checked_byte_len()
                .is_none_or(|bytes| bytes != source_buffer_bytes)
        {
            return Ok(None);
        }
        let mut shape = source
            .shape
            .iter()
            .map(|&n| usize::try_from(n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_precision = match source.element_type {
            runmat_accelerate_api::ProviderElementType::F32 => ProviderPrecision::F32,
            runmat_accelerate_api::ProviderElementType::F64 => ProviderPrecision::F64,
            _ => return Ok(None),
        };
        let mut input_bytes = source_buffer_bytes;
        // Test-only provider receipt: distinct from WGPU hardware/resource
        // qualification, but nonzero and internally reconciled at execution.
        let mut incremental_bytes = 0_u64;
        let mut steps = Vec::new();
        for (index, &axis) in axes_zero_based.iter().enumerate() {
            let previous = shape.clone();
            shape[axis] = 1;
            let step_precision = if index + 1 == axes_zero_based.len() {
                output_precision
            } else {
                ProviderPrecision::F64
            };
            let output_bytes = shape
                .iter()
                .try_fold(1_u64, |n, &extent| {
                    n.checked_mul(u64::try_from(extent).ok()?)
                })
                .and_then(|n| {
                    n.checked_mul(if step_precision == ProviderPrecision::F32 {
                        4
                    } else {
                        8
                    })
                })
                .ok_or_else(|| anyhow!("test ordered mean output bytes overflow"))?;
            incremental_bytes = incremental_bytes
                .checked_add(output_bytes)
                .and_then(|bytes| bytes.checked_add(16))
                .ok_or_else(|| anyhow!("test ordered mean resources overflow"))?;
            steps.push(
                ProviderOrderedRealMeanStep::new(
                    axis,
                    previous,
                    shape.clone(),
                    input_precision,
                    step_precision,
                )
                .with_physical_resources(input_bytes, output_bytes, 16, 1),
            );
            input_precision = step_precision;
            input_bytes = output_bytes;
        }
        Ok(Some(ProviderOrderedRealMeanRecipe::new(
            ProviderOrderedRealMeanSource {
                representation: source.clone(),
                buffer_bytes: source_buffer_bytes,
            },
            ProviderOrderedRealMeanControl {
                axes: axes_zero_based.to_vec(),
                output_precision,
                nan_mode,
            },
            ProviderOrderedRealMeanResources {
                limits: ProviderOrderedRealMeanLimits {
                    max_buffer_size: u64::MAX,
                    max_storage_buffer_binding_size: u32::MAX,
                    max_uniform_buffer_binding_size: u32::MAX,
                    max_compute_invocations_per_workgroup: 256,
                    max_compute_workgroup_size_x: 256,
                    max_compute_workgroups_per_dimension: u32::MAX,
                },
                steps,
                incremental_peak_bytes: incremental_bytes,
                queue_submissions: u32::try_from(axes_zero_based.len()).expect("test axes fit u32"),
            },
        )))
    }

    fn prepare_ordered_real_mean_against_recipe(
        &self,
        source: &GpuTensorHandle,
        recipe: &ProviderOrderedRealMeanRecipe,
    ) -> anyhow::Result<Option<ProviderOrderedRealMean>> {
        let rep = super::ordered_mean::representation_for_test(source)?;
        let current = self.plan_ordered_real_mean(
            &rep,
            source.physical().requested_bytes(),
            recipe.axes(),
            recipe.output_precision(),
            recipe.nan_mode(),
        )?;
        Ok(current
            .filter(|current| current == recipe)
            .map(|current| ProviderOrderedRealMean::new(source.clone(), current)))
    }

    fn execute_ordered_real_mean(
        &self,
        selected: ProviderOrderedRealMean,
        _admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedOrderedRealMean> {
        let step = selected
            .steps()
            .last()
            .ok_or_else(|| ProviderExecutionError::from(anyhow!("no ordered mean step")))?;
        let element = match step.output_precision() {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        let output = self
            .produce_class(
                step.output_shape().to_vec(),
                StageHook::Nd(selected.recipe().axes().to_vec()),
                element,
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
        if *self.mutate_cast_residency.lock().unwrap() {
            // The pending-cast witness must start with real resident metadata;
            // the guard restores its prior state, rather than manufacturing it.
            runmat_accelerate_api::mark_residency(&output);
            assert!(crate::fusion_residency::is_resident(&output));
        }
        *self.ordered_output.lock().unwrap() = Some(output.clone());
        let output_bytes = output.physical().requested_bytes();
        let mut completed = ProviderExecutedOrderedRealMean {
            output,
            requested_incremental_peak_bytes: selected.requested_incremental_peak_bytes(),
            observed_incremental_peak_bytes: selected.requested_incremental_peak_bytes(),
            observed_output_bytes: output_bytes,
            selected_submissions: u32::try_from(selected.steps().len()).unwrap(),
        };
        match *self.ordered_completion_fault.lock().unwrap() {
            Some(OrderedCompletionFault::RequestedPeak) => {
                completed.requested_incremental_peak_bytes += 1;
            }
            Some(OrderedCompletionFault::ObservedPeak) => {
                completed.observed_incremental_peak_bytes += 1;
            }
            Some(OrderedCompletionFault::OutputBytes) => {
                completed.observed_output_bytes += 1;
            }
            Some(OrderedCompletionFault::Submissions) => {
                completed.selected_submissions += 1;
            }
            None => {}
        }
        Ok(completed)
    }
    fn reserve_cached_mean_nd(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
    ) -> anyhow::Result<Option<ProviderPinnedNdMean>> {
        self.cache_reservations.fetch_add(1, Ordering::SeqCst);
        assert_eq!(dims, [1, 2]);
        Ok(self.reserved_mu.lock().unwrap().take().map(|(mean, ex2)| {
            ProviderPinnedNdMean::new(source.clone(), dims.to_vec(), mean, ex2, 16)
        }))
    }
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        snapshot.operations.push(ProviderCapabilityOperation {
            identity: ProviderOperationIdentity::new(
                runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
            ),
            family: ProviderOperationFamily::Elementwise,
        });
        snapshot
    }

    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        let cast_target = if query.operation
            == ProviderOperationIdentity::new(runmat_builtins::SINGLE_CATALOG_ENTRY.identity.name)
        {
            Some(ProviderPrecision::F32)
        } else if query.operation
            == ProviderOperationIdentity::new(runmat_builtins::DOUBLE_CATALOG_ENTRY.identity.name)
        {
            Some(ProviderPrecision::F64)
        } else {
            None
        };
        if let Some(target) = cast_target {
            if *self.reject_cast_query.lock().unwrap() {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedOperation,
                    "staged.test.cast.route_changed",
                );
            }
            if query.family == ProviderOperationFamily::Elementwise
                && query.workload.preparation_route
                    == Some(
                        runmat_accelerate_api::ProviderPreparationRoute::FloatingCast { target },
                    )
                && query.inputs.len() == 1
                && query.outputs.len() == 1
                && query.inputs[0].shape == query.outputs[0].shape
                && query.inputs[0].storage == runmat_accelerate_api::ProviderStorage::DenseReal
            {
                let Some(source) = query.inputs[0].checked_byte_len() else {
                    return ProviderFeasibility::rejected(
                        ProviderRejectionCode::UnknownResourceEstimate,
                        "staged.test.cast.source_overflow",
                    );
                };
                let Some(output) = query.outputs[0].checked_byte_len() else {
                    return ProviderFeasibility::rejected(
                        ProviderRejectionCode::UnknownResourceEstimate,
                        "staged.test.cast.output_overflow",
                    );
                };
                let same = query.inputs[0].element_type == query.outputs[0].element_type;
                return ProviderFeasibility::supported(ProviderResourceEstimate {
                    transient_bytes: if same {
                        Some(source)
                    } else {
                        output
                            .checked_add(16)
                            .and_then(|extra| source.checked_add(extra))
                    },
                    output_bytes: Some(if same { source } else { output }),
                    dispatches: Some(u32::from(!same)),
                });
            }
        }
        if query.operation
            == ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name)
            && query.family == ProviderOperationFamily::Elementwise
            && query.workload.real_power_mode.is_some()
            && matches!(
                query.workload.numerical_domain,
                NumericalDomainEvidence::RealPower(_)
            )
        {
            ProviderFeasibility::supported(ProviderResourceEstimate::default())
        } else {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedElementType,
                "staged.test.unsupported_power",
            )
        }
    }

    fn device_info(&self) -> String {
        "staged-image-owner".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("staged power controls require native F32 upload")
    }

    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        assert_eq!(
            host.shape
                .iter()
                .try_fold(1_usize, |n, &extent| n.checked_mul(extent)),
            Some(1),
            "only a rooted scalar power control uploads"
        );
        let class = match host.data {
            runmat_accelerate_api::HostNumericDataView::F32(values) => {
                self.controls.lock().unwrap().push(
                    runmat_accelerate_api::PowerScalarControl::SingleBits(values[0].to_bits()),
                );
                NumericElementType::F32
            }
            runmat_accelerate_api::HostNumericDataView::F64(values) => {
                self.controls.lock().unwrap().push(
                    runmat_accelerate_api::PowerScalarControl::DoubleBits(values[0].to_bits()),
                );
                NumericElementType::F64
            }
            _ => return Err(anyhow!("only real scalar POWER controls are supported").into()),
        };
        self.uploaded_shapes
            .lock()
            .unwrap()
            .push(host.shape.to_vec());
        Ok(self.produce_class(host.shape.to_vec(), StageHook::Upload, class)?)
    }

    fn fill_like(
        &self,
        input: &GpuTensorHandle,
        control: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = (|| {
            self.controls
                .lock()
                .unwrap()
                .push(match input.physical().element_type() {
                    NumericElementType::F32 => {
                        runmat_accelerate_api::PowerScalarControl::SingleBits(
                            (control as f32).to_bits(),
                        )
                    }
                    NumericElementType::F64 => {
                        runmat_accelerate_api::PowerScalarControl::DoubleBits(control.to_bits())
                    }
                    _ => anyhow::bail!("POWER fill requires floating input"),
                });
            self.produce_class(
                input.shape().to_vec(),
                StageHook::Upload,
                input.physical().element_type(),
            )
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }

    fn classify_real_power<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        Box::pin(async move {
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            Ok(PowerResultDomain::Real)
        })
    }

    fn real_power_in_domain<'a>(
        &'a self,
        base: &'a GpuTensorHandle,
        exponent: &'a GpuTensorHandle,
        _: PowerResultDomain,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let shape =
                runmat_types::geometry::broadcast_concrete_shape(base.shape(), exponent.shape())?;
            self.produce_class(shape, StageHook::Pow, base.physical().element_type())
        })
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("resident session must never download")
    }

    fn reduce_mean_nd<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dims: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            let mut shape = input.shape().to_vec();
            for &dimension in dims {
                shape[dimension] = 1;
            }
            self.produce_optional(shape, StageHook::Nd(dims.to_vec()))
        })
    }

    fn reduce_mean_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            let mut shape = input.shape().to_vec();
            shape[dim] = 1;
            self.produce_optional(shape, StageHook::Dim(dim))
        })
    }

    fn elem_sub<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            self.produce(left.shape().to_vec(), StageHook::Sub)
                .map_err(Into::into)
        })
    }

    fn elem_pow<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.produce(left.shape().to_vec(), StageHook::Pow)
                .map_err(Into::into)
        })
    }

    fn elem_div<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.produce(left.shape().to_vec(), StageHook::Div)
                .map_err(Into::into)
        })
    }

    fn scalar_add(&self, input: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.produce(input.shape().to_vec(), StageHook::Add)
    }

    fn scalar_mul(&self, input: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.produce(input.shape().to_vec(), StageHook::Mul)
    }

    fn scalar_max(&self, input: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.produce(input.shape().to_vec(), StageHook::Max)
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.mutate_cast_metadata(input);
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            if *self.alias_same_class_cast.lock().unwrap()
                && input.physical().element_type() == NumericElementType::F32
            {
                self.trace.lock().unwrap().push(StageHook::Single);
                self.calls.fetch_add(1, Ordering::SeqCst);
                if *self.reject_cast_query_after_cast.lock().unwrap() {
                    *self.reject_cast_query.lock().unwrap() = true;
                }
                return Ok(input.clone());
            }
            self.produce(input.shape().to_vec(), StageHook::Single)
                .map_err(Into::into)
        })
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.mutate_cast_metadata(input);
            if let Some(protected) = self.mutate_protected_residency.lock().unwrap().as_ref() {
                runmat_accelerate_api::clear_residency(protected)
                    .expect("test cast clears protected residency metadata");
            }
            if self.pending_at == Some(self.calls.load(Ordering::SeqCst)) {
                std::future::pending::<()>().await;
            }
            self.produce_class(
                input.shape().to_vec(),
                StageHook::Double,
                NumericElementType::F64,
            )
            .map_err(Into::into)
        })
    }

    fn unary_sqrt<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.produce(input.shape().to_vec(), StageHook::Sqrt) })
    }
}

fn descriptor() -> ImageNormalizeDescriptor {
    ImageNormalizeDescriptor {
        batch: 2,
        height: 3,
        width: 4,
        epsilon: 0.25,
        gain: None,
        bias: None,
        gamma: None,
        clamp_zero: false,
    }
}

pub(super) fn source_recipe() -> ImageMeanAndPowerSource {
    // Lifecycle-only fixture. Real source identity is checked separately by
    // ImageMeanAndPowerSource::from_graph; this is not an arithmetic proof.
    let mean = ImageMeanProviderPlan {
        calls: vec![
            ImageMeanProviderCall {
                hook: ImageMeanProviderHook::ReduceDim(1),
                output_class: crate::fusion::image_reduction::ImageMeanOutput::Native,
                shape: [2, 1, 4],
            },
            ImageMeanProviderCall {
                hook: ImageMeanProviderHook::ReduceDim(2),
                output_class: crate::fusion::image_reduction::ImageMeanOutput::Native,
                shape: [2, 1, 1],
            },
        ],
    };
    ImageMeanAndPowerSource {
        mu: mean.clone(),
        variance: mean,
        square: ImageSourcePower {
            node: 0,
            base: 0,
            exponent_value: 0,
            represented_exponent: 2.0,
        },
        final_power: ImageSourcePower {
            node: 0,
            base: 0,
            exponent_value: 0,
            represented_exponent: 1.0,
        },
    }
}

fn begin(
    provider: &'static StagedProvider,
) -> (ImageNormalizeStageSession, GpuTensorHandle, RetirementProbe) {
    let input = provider
        .allocations
        .allocate_zeroed(
            vec![2, 3, 4],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .expect("owned input");
    let session = ImageNormalizeStageSession::begin(
        provider,
        &input.handle,
        descriptor(),
        source_recipe(),
        ImageArithmeticProof::fixture(),
    )
    .expect("valid owner")
    .expect("supported structural subset");
    (session, input.handle, input.probe)
}

#[path = "tests/mean_sigma.rs"]
mod mean_sigma;
#[path = "tests/moment_publications.rs"]
mod moment_publications;
#[path = "tests/nd_resources.rs"]
mod nd_resources;
#[path = "tests/ordered_mean.rs"]
mod ordered_mean;
#[path = "tests/output_checkpoint.rs"]
mod output_checkpoint;
#[path = "tests/power_service.rs"]
mod power_service;
#[path = "tests/power_pair_provider.rs"]
mod power_pair_provider;
#[path = "tests/real_binary_provider.rs"]
mod real_binary_provider;
#[path = "tests/real_elementwise_provider.rs"]
mod real_elementwise_provider;
#[path = "tests/selected_source.rs"]
mod selected_source;
#[path = "tests/sigma_cast.rs"]
mod sigma_cast;
