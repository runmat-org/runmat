//! Strongly typed provider fixture for caller route and lifecycle tests.
use anyhow::anyhow;
use runmat_accelerate_api::reduction::{
    ReductionRecipeConfig, ReductionStrategy, ReductionSubmission,
};
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, HostTensorView, NumericElementType, ProviderCapabilitySnapshot,
    ProviderCostEstimate, ProviderCostQuery, ProviderExecutedRealElementwise,
    ProviderExecutionFuture, ProviderExecutionResult, ProviderFeasibility,
    ProviderFeasibilityQuery, ProviderPrecision, ProviderRealElementwiseObserved,
    ProviderRealElementwiseOperation, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwisePhysicalProposal, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest, ProviderResourceEstimate, ProviderUploadPhysicalEvidence,
};
use runmat_execution::{
    EstimateConfidence, EstimateSource, ExecutionCostComponents, ExecutionCostEstimate,
    ProviderResourceSnapshot, ResourceByteBound,
};
use runmat_types::FusionReductionOperation;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SubmissionMode {
    Success,
    Unsupported,
    MalformedOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CastExecutionMode {
    Ready,
    PendingAfterAllocation,
}

pub(super) struct TypedCallerProvider {
    allocations: TestAllocationOwner,
    binding_cap: u64,
    precision: ProviderPrecision,
    pub(super) configs: AtomicUsize,
    pub(super) candidates: AtomicUsize,
    pub(super) feasibility: AtomicUsize,
    pub(super) evidence: AtomicUsize,
    pub(super) uploads: AtomicUsize,
    pub(super) submissions: AtomicUsize,
    pub(super) cast_proposals: AtomicUsize,
    pub(super) cast_executions: AtomicUsize,
    pub(super) submission_mode: Mutex<SubmissionMode>,
    pub(super) cast_execution_mode: Mutex<CastExecutionMode>,
    pub(super) estimated_execution_ns: AtomicUsize,
    pub(super) strategies: Mutex<Vec<ReductionStrategy>>,
    pub(super) submitted_strategy: Mutex<Option<ReductionStrategy>>,
    pub(super) switch_after_upload: Mutex<Option<&'static dyn AccelProvider>>,
    pub(super) probes: Mutex<Vec<RetirementProbe>>,
    pub(super) uploaded_elements: Mutex<Vec<NumericElementType>>,
    input_values: Mutex<Vec<f64>>,
    pub(super) submitted_input_values: Mutex<Option<Vec<f64>>>,
    output_values: Mutex<Option<Vec<f64>>>,
}

impl TypedCallerProvider {
    pub(super) fn registered(binding_cap: u64) -> &'static Self {
        Self::registered_with_precision(binding_cap, ProviderPrecision::F64)
    }

    pub(super) fn registered_with_precision(
        binding_cap: u64,
        precision: ProviderPrecision,
    ) -> &'static Self {
        let provider = Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            binding_cap,
            precision,
            configs: AtomicUsize::new(0),
            candidates: AtomicUsize::new(0),
            feasibility: AtomicUsize::new(0),
            evidence: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            submissions: AtomicUsize::new(0),
            cast_proposals: AtomicUsize::new(0),
            cast_executions: AtomicUsize::new(0),
            submission_mode: Mutex::new(SubmissionMode::Success),
            cast_execution_mode: Mutex::new(CastExecutionMode::Ready),
            estimated_execution_ns: AtomicUsize::new(0),
            strategies: Mutex::new(vec![ReductionStrategy::SinglePass]),
            submitted_strategy: Mutex::new(None),
            switch_after_upload: Mutex::new(None),
            probes: Mutex::new(Vec::new()),
            uploaded_elements: Mutex::new(Vec::new()),
            input_values: Mutex::new(Vec::new()),
            submitted_input_values: Mutex::new(None),
            output_values: Mutex::new(None),
        }));
        // SAFETY: the leaked provider retains the unique allocation owner for
        // the process/worker lifetime.
        unsafe { runmat_accelerate_api::register_device_provider(provider) };
        provider
    }

    fn allocate(
        &self,
        shape: Vec<usize>,
        element: NumericElementType,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let allocation = self
            .allocations
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }

    pub(super) fn resident_input(
        &self,
        values: Vec<f64>,
        element: NumericElementType,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        *self.input_values.lock().unwrap() = values;
        self.allocate(vec![2, 1], element)
    }
}

impl AccelProvider for TypedCallerProvider {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        self.precision
    }

    fn device_info(&self) -> String {
        "typed-reduction-caller-fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed caller must use class-preserving numeric upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("typed caller fixture does not download")
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            handle.allocation_lease().validate_live()?;
            let values = self
                .output_values
                .lock()
                .unwrap()
                .clone()
                .expect("typed submission published fixture values");
            let data = match handle.physical().element_type() {
                NumericElementType::F32 => HostNumericDataOwned::F32(
                    values.into_iter().map(|value| value as f32).collect(),
                ),
                NumericElementType::F64 => HostNumericDataOwned::F64(values),
                other => panic!("typed caller fixture cannot download {other:?}"),
            };
            Ok(HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let (values, element) = match host.data {
            HostNumericDataView::F32(values) => (
                values.iter().map(|&value| f64::from(value)).collect(),
                NumericElementType::F32,
            ),
            HostNumericDataView::F64(values) => (values.to_vec(), NumericElementType::F64),
            _ => panic!("typed caller fixture expects floating real upload"),
        };
        *self.input_values.lock().unwrap() = values;
        self.uploaded_elements.lock().unwrap().push(element);
        let output = self.allocate(host.shape.to_vec(), element)?;
        if let Some(provider) = *self.switch_after_upload.lock().unwrap() {
            runmat_accelerate_api::set_thread_provider(Some(provider));
        }
        Ok(output)
    }

    fn upload_physical_evidence(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderUploadPhysicalEvidence> {
        self.evidence.fetch_add(1, Ordering::SeqCst);
        Some(ProviderUploadPhysicalEvidence {
            output_requested_bytes: query.outputs.first()?.checked_byte_len()?,
            host_workspace_bytes: ResourceByteBound::Known(0),
            provider_scratch_bytes: ResourceByteBound::Known(0),
            completion_queue_slots: 1,
        })
    }

    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        self.feasibility.fetch_add(1, Ordering::SeqCst);
        let typed = query
            .workload
            .typed_reduction
            .as_ref()
            .expect("typed caller must carry the selected workload");
        ProviderFeasibility::supported(ProviderResourceEstimate {
            output_bytes: Some(typed.work.output_requested_bytes),
            transient_bytes: typed
                .work
                .output_requested_bytes
                .checked_add(typed.work.partial_requested_bytes),
            dispatches: Some(typed.work.dispatches),
        })
    }

    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        let mut snapshot = ProviderCapabilitySnapshot::conservative(self);
        snapshot.concurrency.transactional_results = true;
        snapshot
    }

    fn placement_resources(&self) -> ProviderResourceSnapshot {
        ProviderResourceSnapshot {
            device_id: self.device_id(),
            capacity_bytes: Some(1 << 20),
            scratch_available_bytes: Some(1 << 20),
            queue_depth: Some(0),
            queue_limit: Some(8),
            ..Default::default()
        }
    }

    fn estimate_cost(&self, _: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
        let execution_ns =
            u64::try_from(self.estimated_execution_ns.load(Ordering::SeqCst)).ok()?;
        (execution_ns != 0).then_some(ProviderCostEstimate {
            cost: ExecutionCostEstimate {
                components: ExecutionCostComponents {
                    execution_ns,
                    ..ExecutionCostComponents::default()
                },
                scratch_bytes: Some(0),
                confidence: EstimateConfidence::Exact,
                source: EstimateSource::Synthetic,
            },
        })
    }

    fn reduction_recipe_config(
        &self,
        _: u32,
        _: u32,
    ) -> ProviderExecutionResult<ReductionRecipeConfig> {
        self.configs.fetch_add(1, Ordering::SeqCst);
        Ok(ReductionRecipeConfig {
            workgroup_size: 8,
            chunk_elements: 2,
            max_workgroup_size: 256,
            max_workgroup_storage_bytes: 32_768,
            max_dispatch_dimension: 65_535,
            max_storage_binding_bytes: self.binding_cap,
            max_rank: 8,
        })
    }

    fn reduction_recipe_candidates(
        &self,
        _: &runmat_accelerate_api::reduction::ProviderReductionRecipe,
        _: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<[Option<ReductionStrategy>; 2]> {
        self.candidates.fetch_add(1, Ordering::SeqCst);
        let strategies = self.strategies.lock().unwrap();
        if strategies.len() > 2 {
            return Err(anyhow!("invalid test candidate count").into());
        }
        Ok([strategies.first().copied(), strategies.get(1).copied()])
    }

    fn submit_reduction_recipe(
        &self,
        submission: &ReductionSubmission<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.submissions.fetch_add(1, Ordering::SeqCst);
        *self.submitted_strategy.lock().unwrap() = Some(submission.selection().strategy());
        assert_eq!(submission.device_id(), self.device_id());
        assert_eq!(submission.inputs().len(), 1);
        submission.inputs()[0]
            .allocation_lease()
            .validate_live()
            .expect("prepared input must remain live through submission");
        if *self.submission_mode.lock().unwrap() == SubmissionMode::Unsupported {
            return Err(runmat_accelerate_api::ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation(
                    "fixture selected submission unsupported",
                ),
            ));
        }
        let descriptor = submission.selection().recipe().descriptor();
        let mut values = self.input_values.lock().unwrap().clone();
        // Independent host oracle for the compact certified producer fixture,
        // not the production integer-word implementation or a capability claim.
        if let Some(program) = submission.selection().recipe().checked_scalar() {
            for node in &program.wire().nodes {
                if let runmat_accelerate_api::scalar_program::ScalarNode::Apply {
                    operation:
                        runmat_types::ScalarOperation::Operator(
                            runmat_types::OperatorKind::ElementwisePower,
                        ),
                    operands,
                    ..
                } = node
                {
                    let exponent = program.wire().nodes.iter().find_map(|node| match node {
                        runmat_accelerate_api::scalar_program::ScalarNode::Literal { id, value: runmat_accelerate_api::scalar_program::ScalarLiteral::F64Bits(bits) } if Some(id) == operands.get(1) => Some(f64::from_bits(*bits)),
                        _ => None,
                    }).expect("checked original control");
                    for value in &mut values {
                        *value = f64::from(value.powf(exponent) as f32);
                    }
                }
            }
        }
        *self.submitted_input_values.lock().unwrap() = Some(values.clone());
        let result = match descriptor.contract.operation {
            FusionReductionOperation::Sum => values.iter().copied().sum(),
            FusionReductionOperation::Minimum => {
                assert!(
                    values.iter().all(|value| value.is_finite()),
                    "fixture minimum covers finite values only"
                );
                values
                    .iter()
                    .copied()
                    .reduce(f64::min)
                    .expect("nonempty minimum fixture")
            }
            other => panic!("typed caller fixture has no {other:?} numerical oracle"),
        };
        *self.output_values.lock().unwrap() = Some(vec![result]);
        let element = match descriptor.output_precision {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        let shape = if *self.submission_mode.lock().unwrap() == SubmissionMode::MalformedOutput {
            vec![2, 1]
        } else {
            descriptor.output_shape.clone()
        };
        self.allocate(shape, element)
    }

    fn propose_real_elementwise(
        &self,
        request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        let ProviderRealElementwiseOperation::FloatingCast { target } = request.operation else {
            return Ok(None);
        };
        self.cast_proposals.fetch_add(1, Ordering::SeqCst);
        let output_buffer_bytes = request
            .source
            .checked_element_count()
            .and_then(|count| {
                count.checked_mul(match target {
                    ProviderPrecision::F32 => 4,
                    ProviderPrecision::F64 => 8,
                })
            })
            .ok_or_else(|| anyhow!("fixture cast output bytes overflow"))?;
        Ok(Some(ProviderRealElementwisePhysicalProposal {
            output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
            output_buffer_bytes,
            workgroup_size: 8,
            chunk_capacity: 2,
            uniform_allocations: 1,
            uniform_buffer_bytes: 16,
            dispatches: 1,
            queue_submissions: 1,
            pipeline_key: Some(7),
        }))
    }

    fn execute_real_elementwise_recipe<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        recipe: &'a ProviderRealElementwiseRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        self.cast_executions.fetch_add(1, Ordering::SeqCst);
        let element = match recipe.output().element_type {
            runmat_accelerate_api::ProviderElementType::F32 => NumericElementType::F32,
            runmat_accelerate_api::ProviderElementType::F64 => NumericElementType::F64,
            _ => unreachable!("fixture cast admits floating real output only"),
        };
        let output = match self.allocate(input.shape().to_vec(), element) {
            Ok(output) => output,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        if *self.cast_execution_mode.lock().unwrap() == CastExecutionMode::PendingAfterAllocation {
            return Box::pin(async move {
                let _retained_until_future_drop = &output;
                std::future::pending::<()>().await;
                unreachable!("pending cast fixture cannot complete")
            });
        }
        Box::pin(async move {
            // The ready fixture cast must model its selected F64→F32 words,
            // not merely allocate an F32-labelled handle. Pending/abandoned
            // casts never publish converted fixture values.
            if input.physical().element_type() == NumericElementType::F64
                && element == NumericElementType::F32
            {
                let mut values = self.input_values.lock().unwrap();
                for value in values.iter_mut() {
                    *value = f64::from(*value as f32);
                }
            }
            let physical = recipe.physical();
            Ok(ProviderExecutedRealElementwise {
                observed: ProviderRealElementwiseObserved {
                    output_buffer_bytes: Some(physical.output_buffer_bytes),
                    uniform_allocations: physical.uniform_allocations,
                    uniform_buffer_bytes: physical.uniform_buffer_bytes,
                    dispatches: physical.dispatches,
                    queue_submissions: physical.queue_submissions,
                },
                output,
            })
        })
    }
}
