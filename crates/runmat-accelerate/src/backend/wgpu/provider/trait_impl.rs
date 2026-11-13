#[path = "cumulative_scan.rs"]
mod cumulative_scan;
use super::allocation_state::AllocationState;
use super::*;
#[path = "trait_impl/typed_elementwise.rs"]
mod typed_elementwise;
use runmat_accelerate_api::{
    AccelIntegerDownloadFuture, AccelNumericDownloadFuture, HostIntegerTensorView,
    HostNumericTensorView, IntegerElementType,
};

#[cfg(test)]
#[path = "trait_impl/numeric_upload_error_tests.rs"]
mod numeric_upload_error_tests;
#[cfg(test)]
#[path = "trait_impl/validated_fusion_tests.rs"]
mod validated_fusion_tests;

#[path = "trait_impl/fusion_observation.rs"]
mod fusion_observation;
use fusion_observation::ElementwiseKernelObservation;

fn numeric_upload_failure(
    error: anyhow::Error,
    buffer_created: bool,
) -> runmat_accelerate_api::ProviderExecutionError {
    let error = runmat_accelerate_api::ProviderExecutionError::from(error);
    if buffer_created {
        error.after_effect()
    } else {
        error
    }
}

fn upload_numeric_with_admission(
    provider: &WgpuProvider,
    host: &HostNumericTensorView,
    admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    scratch: Option<runmat_accelerate_api::HostNumericUploadScratch<'_>>,
) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
    let buffer_created = std::cell::Cell::new(false);
    let output = provider
        .upload_numeric_exec_observed_admitted_with_scratch(
            host,
            || {
                buffer_created.set(true);
                #[cfg(test)]
                {
                    use std::sync::atomic::Ordering;
                    provider
                        .numeric_upload_post_buffer_count
                        .fetch_add(1, Ordering::SeqCst);
                    if let Some(error) = provider
                        .numeric_upload_post_buffer_fault
                        .lock()
                        .map_err(|_| anyhow::anyhow!("numeric upload test fault lock poisoned"))?
                        .take()
                    {
                        return Err(error);
                    }
                }
                Ok(())
            },
            admission,
            scratch,
        )
        .map_err(|error| numeric_upload_failure(error, buffer_created.get()))?;
    if let Err(error) = complete_observed_upload(provider) {
        let primary = numeric_upload_failure(error, true);
        let cleanup = output.release().err().into_iter().collect();
        return Err(runmat_accelerate_api::ProviderExecutionError::from_source(
            runmat_accelerate_api::ProviderExecutionCompletionError::new(Some(primary), cleanup),
        )
        .after_effect());
    }
    Ok(output)
}

/// A host upload may use mapped-at-creation rather than a provider dispatch.
/// Submit a provider-owned fence *after* publication so the whole-call receipt
/// observes queue completion, not merely the return of `create_buffer_init`.
fn complete_observed_upload(provider: &WgpuProvider) -> Result<()> {
    let operation = runmat_accelerate_api::ProviderOperationIdentity::new("transfer.upload");
    let Some(recorder) =
        runmat_accelerate_api::observation::current_provider_observation().filter(|recorder| {
            recorder.expects_operation(&operation)
                || recorder.observes_conditional_provider_upload(provider.device_id())
                || recorder.observes_selected_image_upload(provider.device_id())
                || recorder.observes_stochastic_evolution_source_transfer(provider.device_id())
                || recorder.observes_real_unary_upload(provider.device_id())
                || recorder.observes_power_scalar_upload(provider.device_id())
        })
    else {
        return Ok(());
    };
    let source_transfer_retirement = if recorder
        .observes_stochastic_evolution_source_transfer(provider.device_id())
    {
        match recorder.take_stochastic_evolution_source_transfer_retirement(provider.device_id()) {
            Some(notifier) => Some(notifier),
            None => {
                recorder.fail(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ExecutionUnproven,
                );
                anyhow::bail!(
                    "selected stochastic source upload lacks an armed retirement witness"
                );
            }
        }
    } else {
        None
    };
    let checkpoint = recorder.submission_checkpoint();
    provider.with_emission(move |emission, pending| -> Result<()> {
        if let Some(notifier) = source_transfer_retirement {
            pending.retire_source_transfer_with_next_submit(notifier)?;
        }
        let encoder = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-observed-upload-completion"),
            },
        );
        provider.submit(emission, pending, encoder)?;
        Ok(())
    })??;
    if recorder.expects_operation(&operation) {
        recorder.confirm_submitted_provider_execution_after(
            provider.device_id(),
            &operation,
            checkpoint,
        );
    }
    Ok(())
}
use runmat_accelerate_api::{
    ProviderAdamUpdateRequest, ProviderAdamUpdateResult, ProviderBitModulationRequest,
    ProviderBlackScholesPriceRequest, ProviderBlackScholesPriceResult,
    ProviderCovarianceToCorrelationResult, ProviderCrossentropyRequest, ProviderCrossentropyResult,
    ProviderEnvelopeRequest, ProviderEnvelopeResult, ProviderHilbertRequest,
    ProviderInterp1Request, ProviderModulationRequest, ProviderMovingWindowRequest,
    ProviderNdgridRequest, ProviderNdgridResult, ProviderTrapezoidSpacing,
};

/// The reduction hook, rather than its caller, certifies the selected sum
/// kernel. Host uploads and publication casts may also submit commands, so a
/// submission observed only at the public binding cannot prove reduction work.
fn sum_observation_identity(
    invocation: runmat_types::SumInvocation,
) -> runmat_accelerate_api::ProviderOperationIdentity {
    runmat_accelerate_api::ProviderOperationIdentity::new(
        runmat_builtins::sum_invocation_entry(invocation)
            .identity
            .name,
    )
}

fn sum_submission_checkpoint() -> Option<(
    runmat_accelerate_api::observation::ProviderOperationRecorder,
    u64,
    runmat_accelerate_api::ProviderOperationIdentity,
)> {
    let recorder = runmat_accelerate_api::observation::current_provider_observation()?;
    [
        runmat_types::SumInvocation::Sum,
        runmat_types::SumInvocation::NanSum,
    ]
    .into_iter()
    .find_map(|invocation| {
        let identity = sum_observation_identity(invocation);
        recorder.expects_operation(&identity).then(|| {
            let checkpoint = recorder.submission_checkpoint();
            (recorder.clone(), checkpoint, identity)
        })
    })
}

fn confirm_sum_kernel_submission(
    provider: &WgpuProvider,
    checkpoint: Option<(
        runmat_accelerate_api::observation::ProviderOperationRecorder,
        u64,
        runmat_accelerate_api::ProviderOperationIdentity,
    )>,
) {
    if let Some((recorder, before, identity)) = checkpoint {
        recorder.confirm_submitted_provider_execution_after(
            provider.device_id_exec(),
            &identity,
            before,
        );
    }
}

fn extrema_submission_checkpoint(
    kind: runmat_accelerate_api::ProviderExtremaKind,
) -> Option<(
    runmat_accelerate_api::observation::ProviderOperationRecorder,
    u64,
    runmat_accelerate_api::ProviderOperationIdentity,
)> {
    let recorder = runmat_accelerate_api::observation::current_provider_observation()?;
    let identity = runmat_accelerate_api::ProviderOperationIdentity::new(match kind {
        runmat_accelerate_api::ProviderExtremaKind::Minimum => {
            runmat_builtins::MIN_CATALOG_ENTRY.identity.name
        }
        runmat_accelerate_api::ProviderExtremaKind::Maximum => {
            runmat_builtins::MAX_CATALOG_ENTRY.identity.name
        }
    });
    recorder.expects_operation(&identity).then(|| {
        let checkpoint = recorder.submission_checkpoint();
        (recorder, checkpoint, identity)
    })
}

fn confirm_extrema_kernel_submission(
    provider: &WgpuProvider,
    checkpoint: Option<(
        runmat_accelerate_api::observation::ProviderOperationRecorder,
        u64,
        runmat_accelerate_api::ProviderOperationIdentity,
    )>,
) {
    if let Some((recorder, before, identity)) = checkpoint {
        recorder.confirm_submitted_provider_execution_after(
            provider.device_id_exec(),
            &identity,
            before,
        );
    }
}

impl AccelProvider for WgpuProvider {
    fn region_queue_tail_resources(
        &self,
    ) -> Option<runmat_accelerate_api::ProviderRegionTailResources> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.observation_progress
                .as_ref()
                .is_some_and(|progress| progress.is_available())
                .then_some(runmat_accelerate_api::ProviderRegionTailResources {
                    queue_slots: 1,
                    provider_scratch_bytes: 0,
                    completion_control_host_bytes: self.region_tail_control_host_bytes(),
                })
        }
        #[cfg(target_arch = "wasm32")]
        {
            let bytes = crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES;
            (self.observation_budget.wgpu_query_requested_byte_limit() >= bytes).then_some(
                runmat_accelerate_api::ProviderRegionTailResources {
                    queue_slots: 1,
                    provider_scratch_bytes: bytes,
                    completion_control_host_bytes: self.region_tail_control_host_bytes(),
                },
            )
        }
    }

    fn observe_region_queue_tail<'a>(
        &'a self,
        device_id: u32,
        completion_claim: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ()> {
        Box::pin(self.observe_region_queue_tail_exec(device_id, completion_claim))
    }

    fn wgpu_device_access(&self) -> Option<&runmat_accelerate_api::WgpuDeviceAccess> {
        Some(&self.access)
    }
    fn gpu_device_control(&self) -> Option<&dyn runmat_accelerate_api::ProviderDeviceControl> {
        Some(self)
    }

    fn propose_real_binary(
        &self,
        request: &runmat_accelerate_api::ProviderRealBinaryRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::ProviderRealBinaryPhysicalProposal>,
    > {
        self.propose_real_binary_exec(request)
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
        Box::pin(async move { self.execute_real_binary_recipe_exec(lhs, rhs, recipe, admission) })
    }

    fn registered_output_host_metadata(
        &self,
        rank: usize,
    ) -> Option<runmat_execution::resource::ProviderHostRegistrationMetadata> {
        AllocationState::registration_host_metadata_for_rank(rank)
    }

    fn output_registration_table_upper_bytes(&self, registrations: usize) -> Option<u64> {
        self.allocations.managed_table_upper_bytes(registrations)
    }

    fn output_registration_preparation_host_bytes(&self) -> Option<u64> {
        AllocationState::managed_preparation_host_bytes()
    }

    fn reusable_output_registration_host_bytes(&self) -> Option<u64> {
        runmat_execution::resource::ProviderRegistrationSlotPool::requested_host_bytes()
    }

    fn prepare_reusable_output_host_metadata(
        &self,
        admission: Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        registrations: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    > {
        self.allocations
            .prepare_reusable_managed_metadata(admission, registrations)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)
    }

    fn prepare_output_host_metadata(
        &self,
        admission: Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        registrations: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    > {
        self.allocations
            .prepare_managed_metadata(admission, registrations)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)
    }

    fn propose_real_elementwise(
        &self,
        request: &runmat_accelerate_api::ProviderRealElementwiseRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::ProviderRealElementwisePhysicalProposal>,
    > {
        self.propose_real_elementwise_exec(request)
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
            self.execute_real_elementwise_recipe_exec(input, recipe, admission)
                .await
        })
    }
    fn plan_dense_real_output_buffer_bytes(
        &self,
        output: &runmat_accelerate_api::ProviderRepresentation,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<u64>> {
        self.plan_dense_real_output_buffer_bytes_exec(output)
    }

    fn power_pair_observation_capability(
        &self,
        recipe: &runmat_accelerate_api::power_service::PowerPairCompound,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        self.power_pair_observation_capability_exec(recipe)
    }

    fn power_scalar_observation_capability(
        &self,
        projection: &runmat_accelerate_api::power_service::PowerTypedScalarProjection,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        use runmat_accelerate_api::observation::{
            ProviderObservationCapability as Capability, ProviderObservationOverhead,
        };
        #[cfg(not(target_arch = "wasm32"))]
        if !self
            .observation_progress
            .as_ref()
            .is_some_and(|port| port.is_available())
        {
            return Capability::Unsupported;
        }
        if !projection.is_current_for(self) {
            return Capability::Unsupported;
        }
        let Some(resources) = projection.resource_bounds() else {
            return Capability::Unsupported;
        };
        let submissions = resources.queue_slots;
        // The exact native upload quote already carries its one mapped-upload
        // completion fence. Keep that physical stage in the base demand and
        // add only the other compound fences as observation overhead.
        let fence_bytes = crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES;
        if projection.scalar_upload.provider_scratch_bytes
            != runmat_execution::ResourceByteBound::Known(fence_bytes)
            || projection.scalar_upload.completion_queue_slots != 1
        {
            return Capability::Unsupported;
        }
        let Some(requested_peak_buffer_bytes) = u64::from(submissions)
            .checked_mul(fence_bytes)
            .and_then(|bytes| bytes.checked_sub(fence_bytes))
        else {
            return Capability::Unsupported;
        };
        crate::backend::wgpu::dispatch::observation::gate_prospective_encoder_query(
            Capability::Supported(ProviderObservationOverhead {
                submissions,
                requested_peak_buffer_bytes,
                clear_commands: submissions,
                map_callbacks: submissions,
                wgpu_compute_query: None,
            }),
            self.timestamp_capability,
            &self.observation_budget,
        )
    }

    fn begin_power_scalar_observation(
        &self,
        projection: &runmat_accelerate_api::power_service::PowerTypedScalarProjection,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        match self.power_scalar_observation_capability(projection) {
            runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_) => {
                self.owned_operation_observation()
            }
            runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => {
                runmat_accelerate_api::observation::ProviderObservationSupport::Unsupported
            }
        }
    }

    fn begin_power_pair_observation(
        &self,
        recipe: &runmat_accelerate_api::power_service::PowerPairCompound,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        match self.power_pair_observation_capability_exec(recipe) {
            runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_) => {
                self.owned_operation_observation()
            }
            runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => {
                runmat_accelerate_api::observation::ProviderObservationSupport::Unsupported
            }
        }
    }

    fn propose_power_evaluation(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        domain: runmat_accelerate_api::PowerResultDomain,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerEvaluationPhysicalProposal>,
    > {
        self.propose_power_evaluation_exec(base, exponent, domain)
    }

    fn execute_power_evaluation_recipe<'a>(
        &'a self,
        recipe: &'a runmat_accelerate_api::power_service::PowerEvaluationRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealElementwise,
    > {
        Box::pin(async move { self.execute_power_evaluation_recipe_exec(recipe) })
    }

    fn execute_power_evaluation_recipe_admitted<'a>(
        &'a self,
        recipe: &'a runmat_accelerate_api::power_service::PowerEvaluationRecipe,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedRealElementwise,
    > {
        Box::pin(
            async move { self.execute_power_evaluation_recipe_admitted_exec(recipe, admission) },
        )
    }

    fn propose_power_pair_geometry(
        &self,
        inputs: &[runmat_accelerate_api::ProviderRepresentation; 2],
        descriptions: &[runmat_accelerate_api::PowerInputDescription; 2],
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerPairPhysicalProjection>,
    > {
        self.propose_power_pair_geometry_exec(inputs, descriptions, frozen)
    }

    fn propose_power_probe(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    > {
        self.propose_power_probe_exec(base, exponent, frozen)
    }

    fn execute_power_probe_recipe<'a>(
        &'a self,
        recipe: &'a runmat_accelerate_api::power_service::PowerProbeRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::power_service::PowerProbeObserved,
    > {
        Box::pin(self.execute_power_probe_recipe_exec(recipe))
    }

    fn propose_global_extrema(
        &self,
        request: &runmat_accelerate_api::ProviderGlobalExtremaRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::ProviderGlobalExtremaPhysicalProposal>,
    > {
        self.propose_global_extrema_exec(request)
    }

    fn execute_global_extrema_recipe<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
        recipe: &'a runmat_accelerate_api::ProviderGlobalExtremaRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::ProviderExecutedGlobalExtrema,
    > {
        Box::pin(async move {
            self.execute_global_extrema_recipe_exec(source, recipe)
                .await
        })
    }

    fn plan_power_output_buffer_bytes(
        &self,
        planned: &runmat_accelerate_api::power_service::PowerSymbolicPlan,
        domain: runmat_accelerate_api::PowerResultDomain,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<u64>> {
        self.plan_power_output_buffer_bytes_exec(planned, domain)
    }

    fn plan_ordered_real_mean(
        &self,
        source: &runmat_accelerate_api::ProviderRepresentation,
        source_buffer_bytes: u64,
        axes_zero_based: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: runmat_accelerate_api::ProviderRealMeanNaNMode,
    ) -> Result<Option<runmat_accelerate_api::ProviderOrderedRealMeanRecipe>> {
        self.plan_ordered_real_mean_exec(
            source,
            source_buffer_bytes,
            axes_zero_based,
            output_precision,
            nan_mode,
        )
    }
    fn prepare_ordered_real_mean(
        &self,
        source: &GpuTensorHandle,
        axes_zero_based: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: runmat_accelerate_api::ProviderRealMeanNaNMode,
    ) -> Result<Option<runmat_accelerate_api::ProviderOrderedRealMean>> {
        self.prepare_ordered_real_mean_exec(source, axes_zero_based, output_precision, nan_mode)
    }
    fn prepare_ordered_real_mean_against_recipe(
        &self,
        source: &GpuTensorHandle,
        recipe: &runmat_accelerate_api::ProviderOrderedRealMeanRecipe,
    ) -> Result<Option<runmat_accelerate_api::ProviderOrderedRealMean>> {
        self.prepare_ordered_real_mean_against_recipe_exec(source, recipe)
    }
    fn execute_ordered_real_mean(
        &self,
        selected: runmat_accelerate_api::ProviderOrderedRealMean,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_accelerate_api::ProviderExecutedOrderedRealMean,
    > {
        self.execute_ordered_real_mean_exec(selected, admission)
    }
    fn reserve_cached_mean_nd(
        &self,
        source: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<Option<runmat_accelerate_api::ProviderPinnedNdMean>> {
        self.reserve_cached_mean_nd_exec(source, dims_zero_based)
    }
    fn prepare_cold_mean_nd(
        &self,
        source: &GpuTensorHandle,
        dims_zero_based: &[usize],
    ) -> Result<Option<runmat_accelerate_api::ProviderColdNdMean>> {
        self.prepare_cold_mean_nd_exec(source, dims_zero_based)
    }
    fn execute_cold_mean_nd(
        &self,
        selected: runmat_accelerate_api::ProviderColdNdMean,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<runmat_accelerate_api::ProviderExecutedColdNdMean> {
        self.execute_cold_mean_nd_exec(selected, admission)
    }
    fn resident_network_adjoint_supported(
        &self,
        stages: &[runmat_accelerate_api::ProviderResidentAdjointStage],
        precision: ProviderPrecision,
    ) -> bool {
        let matching_precision = matches!(
            (self.precision, precision),
            (NumericPrecision::F32, ProviderPrecision::F32)
                | (NumericPrecision::F64, ProviderPrecision::F64)
        );
        matching_precision
            && stages.iter().all(|stage| match stage {
                runmat_accelerate_api::ProviderResidentAdjointStage::Dense => true,
                runmat_accelerate_api::ProviderResidentAdjointStage::Conv1D => true,
                runmat_accelerate_api::ProviderResidentAdjointStage::Activation(operation) => {
                    self.activation_adjoint_supported(*operation, precision)
                }
                runmat_accelerate_api::ProviderResidentAdjointStage::SoftmaxRows => true,
                runmat_accelerate_api::ProviderResidentAdjointStage::LayerNorm => true,
                runmat_accelerate_api::ProviderResidentAdjointStage::Pool1D => true,
            })
    }
    fn elementwise_geometry(&self) -> Option<runmat_accelerate_api::ProviderElementwiseGeometry> {
        Some(runmat_accelerate_api::ProviderElementwiseGeometry {
            workgroup_size: self.workgroup_config.scalar,
        })
    }
    fn preparation_snapshot(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
        program: runmat_accelerate_api::ProviderPreparationProgram<'_>,
    ) -> Option<runmat_accelerate_api::ProviderPreparationSnapshot> {
        self.fusion_preparation_snapshot(query, program)
    }
    fn image_stage_observation_capability(
        &self,
        work: runmat_accelerate_api::ProviderImageStagePhysicalWork,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        use runmat_accelerate_api::observation::{
            ProviderObservationCapability, ProviderObservationOverhead,
        };
        #[cfg(not(target_arch = "wasm32"))]
        if !self
            .observation_progress
            .as_ref()
            .is_some_and(|progress| progress.is_available())
        {
            return ProviderObservationCapability::Unsupported;
        }
        if !work.validate() {
            return ProviderObservationCapability::Unsupported;
        }
        let Some(requested_peak_buffer_bytes) = u64::from(work.queue_submissions)
            .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
        else {
            return ProviderObservationCapability::Unsupported;
        };
        crate::backend::wgpu::dispatch::observation::gate_prospective_encoder_query(
            ProviderObservationCapability::Supported(ProviderObservationOverhead {
                submissions: work.queue_submissions,
                requested_peak_buffer_bytes,
                clear_commands: work.queue_submissions,
                map_callbacks: work.queue_submissions,
                wgpu_compute_query: None,
            }),
            self.timestamp_capability,
            &self.observation_budget,
        )
    }

    fn begin_image_stage_observation(
        &self,
        work: runmat_accelerate_api::ProviderImageStagePhysicalWork,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        use runmat_accelerate_api::observation::{
            ProviderObservationCapability, ProviderObservationSupport,
        };
        match self.image_stage_observation_capability(work) {
            ProviderObservationCapability::Supported(_) => self.owned_operation_observation(),
            ProviderObservationCapability::Unsupported => ProviderObservationSupport::Unsupported,
        }
    }
    fn operation_observation_capability(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        let with_device_query = |capability| {
            crate::backend::wgpu::dispatch::observation::gate_prospective_encoder_query(
                capability,
                self.timestamp_capability,
                &self.observation_budget,
            )
        };
        #[cfg(not(target_arch = "wasm32"))]
        if !self
            .observation_progress
            .as_ref()
            .is_some_and(|progress| progress.is_available())
        {
            return runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported;
        }
        if let Some(capability) = self.stochastic_evolution_observation_capability(query) {
            return with_device_query(capability);
        }
        if query.workload.gpu_random.is_some() || query.workload.gpu_integer.is_some() {
            use runmat_accelerate_api::observation::{
                ProviderObservationCapability, ProviderObservationOverhead,
            };
            let Some(runmat_accelerate_api::ProviderFeasibility::Supported { estimate }) =
                self.random_child_feasibility(query)
            else {
                return ProviderObservationCapability::Unsupported;
            };
            let Some(submissions) = estimate.dispatches else {
                return ProviderObservationCapability::Unsupported;
            };
            let Some(requested_peak_buffer_bytes) = u64::from(submissions)
                .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
            else {
                return ProviderObservationCapability::Unsupported;
            };
            return with_device_query(ProviderObservationCapability::Supported(
                ProviderObservationOverhead {
                    submissions,
                    requested_peak_buffer_bytes,
                    clear_commands: submissions,
                    map_callbacks: submissions,
                    wgpu_compute_query: None,
                },
            ));
        }
        if query.workload.typed_reduction.is_some() {
            return with_device_query(crate::placement::typed_reduction_observation_capability(
                query,
            ));
        }
        if query.workload.extrema.is_some() {
            return with_device_query(crate::placement::extrema_observation_capability(
                self, query,
            ));
        }
        if query.operation.0 == "transfer.upload" {
            let output_count = query
                .workload
                .transfer_output_bounds
                .as_ref()
                .map_or(query.outputs.len(), Vec::len);
            let Some(submissions) = u32::try_from(output_count).ok().filter(|count| *count != 0)
            else {
                return runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported;
            };
            let Some(requested_peak_buffer_bytes) = u64::from(submissions)
                .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
            else {
                return runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported;
            };
            return if self.query_feasibility(query).is_supported() {
                runmat_accelerate_api::observation::ProviderObservationCapability::Supported(
                    runmat_accelerate_api::observation::ProviderObservationOverhead {
                        submissions,
                        requested_peak_buffer_bytes,
                        clear_commands: submissions,
                        map_callbacks: submissions,
                        wgpu_compute_query: None,
                    },
                )
            } else {
                runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported
            };
        }
        use runmat_accelerate_api::observation::ProviderObservationCapability;
        let mut capability = crate::placement::provider_observation_capability(self, query);
        if matches!(capability, ProviderObservationCapability::Unsupported) {
            capability = crate::placement::floating_cast_observation_capability(self, query);
        }
        if matches!(capability, ProviderObservationCapability::Unsupported) {
            capability = crate::placement::sum_observation_capability(self, query);
        }
        if matches!(capability, ProviderObservationCapability::Unsupported) {
            capability = crate::placement::convolution_observation_capability(self, query);
        }
        if matches!(capability, ProviderObservationCapability::Unsupported) {
            capability = crate::placement::fspecial_observation_capability(self, query);
        }
        if matches!(capability, ProviderObservationCapability::Unsupported) {
            capability = crate::placement::imfilter_observation_capability(self, query);
        }
        with_device_query(capability)
    }

    fn begin_operation_observation(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        use runmat_accelerate_api::observation::{
            ProviderObservationCapability, ProviderObservationSupport,
        };
        match self.operation_observation_capability(query) {
            ProviderObservationCapability::Unsupported => ProviderObservationSupport::Unsupported,
            ProviderObservationCapability::Supported(_) => self.owned_operation_observation(),
        }
    }
    fn real_unary_observation_capability(
        &self,
        recipe: &runmat_accelerate_api::ProviderRealUnaryCompound,
        continuations: runmat_accelerate_api::ProviderRealUnaryContinuations,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        self.real_unary_observation_capability_exec(recipe, continuations)
    }
    fn begin_real_unary_observation(
        &self,
        recipe: &runmat_accelerate_api::ProviderRealUnaryCompound,
        continuations: runmat_accelerate_api::ProviderRealUnaryContinuations,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        match self.real_unary_observation_capability_exec(recipe, continuations) {
            runmat_accelerate_api::observation::ProviderObservationCapability::Unsupported => {
                runmat_accelerate_api::observation::ProviderObservationSupport::Unsupported
            }
            runmat_accelerate_api::observation::ProviderObservationCapability::Supported(_) => {
                self.owned_operation_observation()
            }
        }
    }
    fn matmul_geometry(&self) -> Option<runmat_accelerate_api::ProviderMatmulGeometry> {
        Some(runmat_accelerate_api::ProviderMatmulGeometry {
            tile_size: self.workgroup_config.matmul_tile,
            diagonal_workgroup_size: self.workgroup_config.scalar,
        })
    }

    fn sum_geometry(&self) -> Option<runmat_accelerate_api::ProviderSumGeometry> {
        // These diagnostic modes add copies or disable the ordinary kernels.
        // Until they are part of the route identity, no ordinary recipe may
        // claim their physical cost or a successful device route.
        if [
            "RUNMAT_PROVIDER_REDUCTION_SNAPSHOT",
            "RUNMAT_FORCE_REDUCE_SNAPSHOT",
            "RUNMAT_DISABLE_REDUCE_GLOBAL",
            "RUNMAT_DISABLE_REDUCE_DIM",
        ]
        .iter()
        .any(|name| std::env::var_os(name).is_some())
        {
            return None;
        }
        Some(runmat_accelerate_api::ProviderSumGeometry {
            max_buffer_bytes: self.adapter_limits.max_buffer_size,
            max_binding_bytes: u64::from(self.adapter_limits.max_storage_buffer_binding_size),
            nd_chunk_elements: crate::backend::wgpu::config::elementwise_chunk_capacity(
                self.workgroup_config.scalar,
            )?,
            cast_chunk_elements: crate::backend::wgpu::config::elementwise_chunk_capacity(
                self.workgroup_config.scalar,
            )?,
            integer_cast_chunk_elements: crate::backend::wgpu::config::elementwise_chunk_capacity(
                self.workgroup_config.scalar,
            )?,
            dimension_2_max_output_elements: u64::from(
                self.adapter_limits.max_compute_workgroups_per_dimension,
            )
            .checked_mul(u64::from(
                crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
            ))?,
            global_chunk_elements: u64::from(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS)
                .checked_mul(2 * u64::from(crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE))?,
            global_group_elements: 2 * u64::from(
                crate::backend::wgpu::config::REDUCE_WORKGROUP_SIZE,
            ),
        })
    }

    fn extrema_geometry(&self) -> Option<runmat_accelerate_api::ProviderExtremaGeometry> {
        Some(runmat_accelerate_api::ProviderExtremaGeometry {
            max_buffer_bytes: self.adapter_limits.max_buffer_size,
            max_binding_bytes: u64::from(self.adapter_limits.max_storage_buffer_binding_size),
            workgroup_size: self.workgroup_config.scalar,
            max_workgroups_per_dimension: self.adapter_limits.max_compute_workgroups_per_dimension,
        })
    }

    fn convolution_geometry(&self) -> Option<runmat_accelerate_api::ProviderConvolutionGeometry> {
        Some(runmat_accelerate_api::ProviderConvolutionGeometry {
            max_buffer_bytes: self.adapter_limits.max_buffer_size,
            max_binding_bytes: u64::from(self.adapter_limits.max_storage_buffer_binding_size),
            max_workgroups_per_dimension: self.adapter_limits.max_compute_workgroups_per_dimension,
            workgroup_size: crate::backend::wgpu::config::WORKGROUP_SIZE,
        })
    }

    fn imfilter_geometry(&self) -> Option<runmat_accelerate_api::ProviderImfilterGeometry> {
        Some(runmat_accelerate_api::ProviderImfilterGeometry {
            max_buffer_bytes: self.adapter_limits.max_buffer_size,
            max_binding_bytes: u64::from(self.adapter_limits.max_storage_buffer_binding_size),
            max_rank: crate::backend::wgpu::params::IMFILTER_MAX_RANK,
            max_workgroups_per_dispatch: crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS,
            workgroup_size: crate::backend::wgpu::config::WORKGROUP_SIZE,
            shader_f64: self.adapter.features().contains(wgpu::Features::SHADER_F64),
        })
    }

    fn imfilter_physical_evidence(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderImfilterPhysicalEvidence> {
        crate::placement::imfilter_physical_evidence(self, query)
    }

    fn numeric_upload_physical_evidence(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence> {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderLayout, ProviderOperationFamily, ProviderResidency,
            ProviderStorage,
        };
        if query.operation.0 != "transfer.upload"
            || query.family != ProviderOperationFamily::Upload
            || query.inputs.len() != 1
            || query.outputs.len() != 1
            || query.workload.requested_outputs != Some(1)
        {
            return None;
        }
        let output = &query.outputs[0];
        if !matches!(
            output.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        ) || output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
        {
            return None;
        }
        let physical = self.upload_physical_evidence(query)?;
        Some(
            runmat_accelerate_api::ProviderNumericUploadPhysicalEvidence {
                output_requested_bytes: physical.output_requested_bytes,
                // create_buffer_init owns the output directly. The only extra
                // requested buffer is the observed completion fence submitted
                // after publication, including empty logical payloads.
                provider_scratch_bytes: physical.provider_scratch_bytes,
                completion_queue_slots: physical.completion_queue_slots,
            },
        )
    }

    fn floating_cast_physical_evidence(
        &self,
        input: &runmat_accelerate_api::ProviderRepresentation,
        target: ProviderPrecision,
    ) -> Option<runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence> {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderLayout, ProviderResidency, ProviderStorage,
        };
        if input.residency != ProviderResidency::Device
            || input.layout != ProviderLayout::ColumnMajorContiguous
            || !matches!(
                (input.storage, input.element_type),
                (
                    ProviderStorage::DenseReal,
                    ProviderElementType::F32 | ProviderElementType::F64
                ) | (
                    ProviderStorage::DenseComplexInterleaved,
                    ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64
                )
            )
            || self.precision() != ProviderPrecision::F64
        {
            return None;
        }
        let target_type = ProviderElementType::from(target);
        let work = crate::placement::floating_cast_build_for_input(self, input, target_type)?;
        if work.source == target || work.estimate.dispatches == Some(0) {
            return None;
        }
        Some(
            runmat_accelerate_api::ProviderFloatingCastPhysicalEvidence {
                output_requested_bytes: work.estimate.output_bytes?,
                uniform_buffer_bytes: work.uniform_bytes,
                dispatches: work.estimate.dispatches?,
                queue_slots: work.estimate.dispatches?,
            },
        )
    }

    fn upload_physical_evidence(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderUploadPhysicalEvidence> {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderFeasibility, ProviderLayout, ProviderOperationFamily,
            ProviderResidency, ProviderStorage, ProviderWorkload,
        };
        if query.operation.0 != "transfer.upload"
            || query.family != ProviderOperationFamily::Upload
            || query.inputs.len() != 1
            || query.outputs.len() != 1
        {
            return None;
        }
        let expected_workload = ProviderWorkload {
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        };
        if query.workload != expected_workload {
            return None;
        }
        let output = &query.outputs[0];
        if !matches!(
            output.element_type,
            ProviderElementType::Logical
                | ProviderElementType::I8
                | ProviderElementType::I16
                | ProviderElementType::I32
                | ProviderElementType::I64
                | ProviderElementType::U8
                | ProviderElementType::U16
                | ProviderElementType::U32
                | ProviderElementType::U64
                | ProviderElementType::F32
                | ProviderElementType::F64
                | ProviderElementType::ComplexF32
                | ProviderElementType::ComplexF64
        ) || output.storage
            != match output.element_type {
                ProviderElementType::ComplexF32 | ProviderElementType::ComplexF64 => {
                    ProviderStorage::DenseComplexInterleaved
                }
                _ => ProviderStorage::DenseReal,
            }
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
        {
            return None;
        }
        let ProviderFeasibility::Supported { estimate } = self.query_feasibility(query) else {
            return None;
        };
        let output_requested_bytes = crate::backend::wgpu_storage::checked_upload_allocation_bytes(
            output,
            self.precision(),
        )?;
        if estimate.output_bytes != Some(output_requested_bytes) {
            return None;
        }
        let host_conversion_width = match output.element_type {
            ProviderElementType::Logical => {
                ProviderElementType::from(self.precision()).byte_width()
            }
            ProviderElementType::I8
            | ProviderElementType::I16
            | ProviderElementType::I32
            | ProviderElementType::U8
            | ProviderElementType::U16
            | ProviderElementType::U32 => 4,
            ProviderElementType::I64 | ProviderElementType::U64 => 8,
            ProviderElementType::F32
            | ProviderElementType::F64
            | ProviderElementType::ComplexF32
            | ProviderElementType::ComplexF64 => 0,
        };
        let host_workspace_bytes = output
            .checked_element_count()?
            .checked_mul(host_conversion_width)?;
        Some(runmat_accelerate_api::ProviderUploadPhysicalEvidence {
            output_requested_bytes,
            host_workspace_bytes: runmat_execution::ResourceByteBound::Known(host_workspace_bytes),
            provider_scratch_bytes: runmat_execution::ResourceByteBound::Known(
                crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
            ),
            completion_queue_slots: 1,
        })
    }

    fn export_context(&self, kind: AccelContextKind) -> Option<AccelContextHandle> {
        self.export_context_exec(kind)
    }

    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<WgpuBufferRef>> {
        self.export_wgpu_buffer_exec(handle)
    }

    fn device_id(&self) -> u32 {
        self.device_id_exec()
    }

    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        self.spawn_handle_concurrency_exec()
    }

    fn capability_snapshot(&self) -> runmat_accelerate_api::ProviderCapabilitySnapshot {
        self.capability_snapshot
            .get()
            .expect("provider capability snapshot initialized at construction")
            .clone()
    }

    fn capability_snapshot_ref(
        &self,
    ) -> Option<&runmat_accelerate_api::ProviderCapabilitySnapshot> {
        self.capability_snapshot.get()
    }

    fn execution_accelerator_device(
        &self,
        inventory_epoch: u64,
    ) -> Result<Option<runmat_execution::resource::AcceleratorDevice>> {
        runmat_accelerate_api::execution_accelerator_device(
            self,
            runmat_accelerate_api::RUNMAT_WGPU_PROVIDER_ID,
            runmat_accelerate_api::RUNMAT_BUILTIN_PROVIDER_VERSION,
            runmat_accelerate_api::ExecutionProviderContract::GenericComputeV1,
            "primary-gpu",
            &format!(
                "wgpu-{:04x}-{:04x}-{:?}",
                self.adapter_info.vendor, self.adapter_info.device, self.adapter_info.backend
            ),
            inventory_epoch,
        )
    }

    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        if let Some(stochastic) = self.stochastic_evolution_feasibility(query) {
            return stochastic;
        }
        if let Some(random) = self.random_child_feasibility(query) {
            return random;
        }
        if let Some(native_integer) = super::integer::integer_division_feasibility(self, query) {
            return native_integer;
        }
        crate::placement::wgpu_feasibility(self, query, self.adapter_limits.max_buffer_size)
    }

    fn query_numeric_download_feasibility(
        &self,
        source: &GpuTensorHandle,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.numeric_download_feasibility_exec(source, query)
    }

    fn estimate_cost(
        &self,
        query: &runmat_accelerate_api::ProviderCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        if query.operation.workload.image_stages.is_some()
            || query.preparation == runmat_execution::CandidatePreparationState::Unknown
            || crate::placement::floating_cast_unresolved_adaptation(&query.operation)
            || query.operation.workload.iir.is_some()
            || query.operation.workload.fspecial.is_some()
            || matches!(
                query.operation.operation.0.as_str(),
                runmat_accelerate_api::PROVIDER_IIR_CAUSAL_PASS_OPERATION
                    | runmat_accelerate_api::PROVIDER_IIR_ZERO_PHASE_OPERATION
                    | runmat_accelerate_api::PROVIDER_IIR_PREPARED_PASS_OPERATION
            )
        {
            return None;
        }
        Some(crate::placement::wgpu_cost_estimate_with_geometry(
            self,
            query,
            self.linear_launch,
        ))
    }

    fn query_sparse_feasibility(
        &self,
        query: &runmat_accelerate_api::SparseFeasibilityQuery,
    ) -> runmat_accelerate_api::SparseFeasibility {
        sparse::feasibility(self, query)
    }

    fn estimate_sparse_cost(
        &self,
        query: &runmat_accelerate_api::SparseCostQuery,
    ) -> Option<runmat_accelerate_api::ProviderCostEstimate> {
        sparse::cost(self, query)
    }

    fn upload_sparse_pattern(
        &self,
        pattern: runmat_accelerate_api::HostCsrPatternView<'_>,
    ) -> Result<runmat_accelerate_api::SparsePatternHandle> {
        let id = self.sparse.next_id();
        self.sparse.upload_pattern(self, id, pattern)
    }

    fn bind_sparse_values(
        &self,
        pattern: &runmat_accelerate_api::SparsePatternHandle,
        values: &HostNumericTensorView<'_>,
    ) -> Result<runmat_accelerate_api::SparseMatrixHandle> {
        let id = self.sparse.next_id();
        self.sparse.bind_values(self, id, pattern, *values)
    }

    fn free_sparse_pattern(
        &self,
        pattern: &runmat_accelerate_api::SparsePatternHandle,
    ) -> Result<()> {
        self.sparse.free_pattern(self, pattern)
    }

    fn free_sparse_matrix(&self, matrix: &runmat_accelerate_api::SparseMatrixHandle) -> Result<()> {
        self.sparse.free_matrix(self, matrix)
    }

    fn sparse_spmv<'a>(
        &'a self,
        request: &'a runmat_accelerate_api::SparseSpmvRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        sparse::execute_spmv(self, request)
    }

    fn prepare_sparse_preconditioner<'a>(
        &'a self,
        request: &'a runmat_accelerate_api::SparsePreconditionerRequest<'a>,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::SparsePreconditionerHandle> {
        sparse::prepare_preconditioner(self, request)
    }

    fn apply_sparse_preconditioner<'a>(
        &'a self,
        request: &'a runmat_accelerate_api::SparsePreconditionerApplyRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        sparse::apply_preconditioner(self, request)
    }

    fn free_sparse_preconditioner(
        &self,
        preconditioner: &runmat_accelerate_api::SparsePreconditionerHandle,
    ) -> Result<()> {
        self.sparse.free_preconditioner(self, preconditioner)
    }

    fn prepare_sparse_multigrid(
        &self,
        request: &runmat_accelerate_api::SparseMultigridRequest<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::SparseMultigridHandle> {
        sparse::prepare_multigrid(self, request)
    }

    fn sparse_multigrid_solve<'a>(
        &'a self,
        request: &'a runmat_accelerate_api::SparseMultigridSolveRequest<'a>,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::SparseMultigridSolveResult> {
        sparse::solve_multigrid(self, request)
    }

    fn free_sparse_multigrid(
        &self,
        hierarchy: &runmat_accelerate_api::SparseMultigridHandle,
    ) -> anyhow::Result<()> {
        sparse::free_multigrid(self, hierarchy)
    }

    fn sparse_krylov_solve<'a>(
        &'a self,
        request: &'a runmat_accelerate_api::SparseKrylovRequest<'a>,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::SparseKrylovResult> {
        sparse::execute_krylov(self, request)
    }

    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        // WebGPU exposes a maximum single-buffer size, not a trustworthy total
        // device-memory budget. Keep total capacity unknown rather than
        // misclassifying several legal buffers as overcommitted.
        let live_bytes = self.allocations.retained_bytes();
        let sparse_bytes = self.sparse.resident_bytes();
        let reclaimable_bytes = self.allocations.pooled_bytes(self.element_size);
        let allocated_bytes = live_bytes
            .saturating_add(sparse_bytes)
            .saturating_add(reclaimable_bytes);
        runmat_execution::ProviderResourceSnapshot {
            device_id: self.runtime_device_id,
            capacity_bytes: None,
            live_bytes: allocated_bytes,
            reclaimable_bytes,
            scratch_available_bytes: None,
            queue_depth: None,
            queue_limit: None,
            lost: self.allocations.is_lost() || self.device_health.loss_reason().is_some(),
            epoch: allocated_bytes
                ^ reclaimable_bytes.rotate_left(32)
                ^ sparse_bytes.rotate_left(17),
        }
    }

    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        self.gather_linear_exec(source, indices, output_shape)
    }

    fn indexing_host_workspace_bound(
        &self,
        source: &GpuTensorHandle,
        rhs: Option<&GpuTensorHandle>,
    ) -> runmat_execution::ResourceByteBound {
        // entry_for_handle constructs exactly one logical-shape Vec per
        // authenticated input; the admitted kernels retain at most these two.
        source
            .shape()
            .len()
            .checked_add(rhs.map_or(0, |rhs| rhs.shape().len()))
            .and_then(|rank| rank.checked_mul(std::mem::size_of::<usize>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .map(runmat_execution::ResourceByteBound::Known)
            .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            ))
    }

    fn gather_linear_admitted(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
        admission: &dyn runmat_execution::resource::ProviderHostMetadataAdmission,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.gather_linear_exec_admitted(source, indices, output_shape, Some(admission))
            .map_err(|cause| {
                runmat_accelerate_api::ProviderExecutionError::from(cause).with_unproven_effects()
            })
    }

    fn scatter_linear_functional_admitted(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
        mapping: runmat_accelerate_api::ProviderScatterValueMapping,
        admission: &dyn runmat_execution::resource::ProviderHostMetadataAdmission,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.scatter_linear_functional_exec_admitted(
            target,
            indices,
            values,
            mapping,
            Some(admission),
        )
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> Result<()> {
        self.scatter_linear_exec(target, indices, values)
    }

    fn scatter_linear_functional<'a>(
        &'a self,
        target: &'a GpuTensorHandle,
        indices: &'a [u32],
        values: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.scatter_linear_functional_exec(target, indices, values) })
    }

    fn zeros(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.zeros_exec(shape)
    }

    fn zeros_with_storage(
        &self,
        shape: &[usize],
        storage: GpuTensorStorage,
    ) -> Result<GpuTensorHandle> {
        self.zeros_with_storage_exec(shape, storage)
    }

    fn zeros_like(
        &self,
        prototype: &GpuTensorHandle,
        shape: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.zeros_like_constructor(prototype, shape)
    }

    fn logical_fill(
        &self,
        shape: &[usize],
        value: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.logical_fill_constructor(shape, value)
    }

    fn zeros_integer_like(
        &self,
        prototype: &GpuTensorHandle,
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(prototype)?;
        let element_type = entry.integer_type().ok_or_else(|| {
            anyhow::anyhow!("zeros_integer_like requires a native integer gpuArray prototype")
        })?;
        anyhow::ensure!(
            entry.storage == GpuTensorStorage::Real,
            "zeros_integer_like does not support complex integer gpuArray buffers"
        );
        let len = shape.iter().try_fold(1usize, |total, &dim| {
            total
                .checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("zeros_integer_like: tensor size overflow"))
        })?;
        integer::identity_integer_buffer(self, element_type, len, shape, false)
    }

    fn precision(&self) -> ProviderPrecision {
        self.provider_precision_exec()
    }

    fn fill(
        &self,
        shape: &[usize],
        value: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.fill_constructor(shape, value)
    }

    fn fill_like(
        &self,
        prototype: &GpuTensorHandle,
        value: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.fill_like_constructor(prototype, value)
    }

    fn eye(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.eye_exec(shape)
    }

    fn eye_like(&self, prototype: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.eye_exec(prototype.shape())
    }

    fn meshgrid(&self, axes: &[MeshgridAxisView<'_>]) -> Result<ProviderMeshgridResult> {
        self.meshgrid_exec(axes)
    }

    fn ndgrid(&self, request: &ProviderNdgridRequest<'_>) -> Result<ProviderNdgridResult> {
        self.ndgrid_exec(request)
    }

    fn black_scholes_price(
        &self,
        request: &ProviderBlackScholesPriceRequest<'_>,
    ) -> Result<ProviderBlackScholesPriceResult> {
        self.black_scholes_price_exec(request)
    }

    fn adam_update(
        &self,
        request: &ProviderAdamUpdateRequest<'_>,
    ) -> Result<ProviderAdamUpdateResult> {
        self.adam_update_exec(request)
    }

    fn crossentropy_terms(
        &self,
        request: &ProviderCrossentropyRequest<'_>,
    ) -> Result<ProviderCrossentropyResult> {
        self.crossentropy_terms_exec(request)
    }

    fn linspace(&self, start: f64, stop: f64, count: usize) -> Result<GpuTensorHandle> {
        self.linspace_exec(start, stop, count)
    }

    fn sequence(
        &self,
        request: &runmat_accelerate_api::ProviderSequenceRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        if request.precision != self.precision() {
            return Ok(None);
        }
        let effect = std::cell::Cell::new(false);
        self.sequence_exec_observed(request, &effect)
            .map(Some)
            .map_err(|error| {
                let error = runmat_accelerate_api::ProviderExecutionError::from(error);
                if effect.get() {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                }
            })
    }

    fn random_uniform(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_uniform_exec(shape)
    }

    fn random_normal(&self, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_normal_exec(shape)
    }

    fn execute_gpu_random(
        &self,
        request: &runmat_accelerate_api::random::GpuRandomRequest<'_>,
        metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.random_child_exec(request, metadata)
    }

    fn propose_gpu_random(
        &self,
        request: &runmat_accelerate_api::random::GpuRandomRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::random::GpuRandomPhysicalProposal>,
    > {
        Ok(self.propose_random_child_exec(request))
    }

    fn propose_gpu_integer(
        &self,
        request: &runmat_accelerate_api::random::GpuIntegerRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::random::GpuRandomPhysicalProposal>,
    > {
        Ok(self.propose_integer_child_exec(request))
    }

    fn execute_gpu_integer(
        &self,
        request: &runmat_accelerate_api::random::GpuIntegerRequest<'_>,
        metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.integer_child_exec(request, metadata)
    }

    fn propose_gpu_integer_batch(
        &self,
        request: runmat_accelerate_api::random::GpuIntegerBatchRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::random::GpuIntegerBatchPhysicalProposal>,
    > {
        Ok(self.propose_integer_child_batch_exec(request))
    }

    fn execute_gpu_integer_batch<'a>(
        &'a self,
        request: runmat_accelerate_api::random::GpuIntegerBatchRequest<'a>,
        metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        readback_control: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::random::GpuIntegerBatchProgress,
    > {
        Box::pin(self.integer_child_batch_exec(request, metadata, readback_control))
    }

    fn random_exponential(&self, mu: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_exponential_exec(mu, shape)
    }

    fn random_normrnd(&self, mu: f64, sigma: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_normrnd_exec(mu, sigma, shape)
    }

    fn random_unifrnd(&self, a: f64, b: f64, shape: &[usize]) -> Result<GpuTensorHandle> {
        self.random_unifrnd_exec(a, b, shape)
    }

    fn stochastic_evolution_planning_host_workspace(
        &self,
        _rank: usize,
    ) -> runmat_execution::ResourceByteBound {
        // `plan_stochastic_evolution_exec` does not allocate temporary host
        // workspace. Its two shape clones are retained by the returned plan,
        // so Runtime accounts them as selected-plan metadata rather than as
        // planning scratch.
        runmat_execution::ResourceByteBound::Known(0)
    }

    fn stochastic_evolution_source_transfer_retirement_host_workspace(
        &self,
    ) -> runmat_execution::ResourceByteBound {
        runmat_accelerate_api::observation::stochastic_evolution_source_transfer_retirement_host_upper_bytes()
            .map(runmat_execution::ResourceByteBound::Known)
            .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            ))
    }

    fn plan_stochastic_evolution(
        &self,
        request: &runmat_accelerate_api::stochastic_evolution::ProviderStochasticEvolutionWorkRequest,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::stochastic_evolution::ProviderStochasticEvolutionSelection>,
    > {
        Ok(self.plan_stochastic_evolution_exec(request))
    }

    fn execute_stochastic_evolution_tile<'a>(
        &'a self,
        request: runmat_accelerate_api::stochastic_evolution::ProviderStochasticEvolutionTileRequest<'a>,
        metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        completion_control: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        runmat_accelerate_api::stochastic_evolution::ProviderStochasticEvolutionProgress,
    > {
        Box::pin(self.stochastic_evolution_tile_exec(request, metadata, completion_control))
    }

    fn fspecial(&self, request: &FspecialRequest) -> Result<GpuTensorHandle> {
        self.fspecial_exec(request)
    }

    fn peaks(&self, n: usize) -> Result<GpuTensorHandle> {
        self.peaks_exec(n)
    }

    fn peaks_xy(&self, x: &GpuTensorHandle, y: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.peaks_xy_exec(x, y)
    }

    fn hann_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Hann, len, periodic)
    }

    fn hamming_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Hamming, len, periodic)
    }

    fn blackman_window(&self, len: usize, periodic: bool) -> Result<GpuTensorHandle> {
        self.window_exec(WindowKind::Blackman, len, periodic)
    }

    fn imfilter<'a>(
        &'a self,
        image: &'a GpuTensorHandle,
        kernel: &'a GpuTensorHandle,
        options: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.imfilter_exec(image, kernel, options).await })
    }

    fn set_rng_state(&self, state: u64) -> Result<()> {
        self.set_rng_state_exec(state)
    }

    fn random_permutation(
        &self,
        n: usize,
        k: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        // The installed F32 permutation pipeline cannot publish the Double
        // output required by randperm. This is a kernel capability, not an
        // inference from preferred precision about typed upload capability.
        if self.precision() != ProviderPrecision::F64 {
            return Err(runmat_accelerate_api::ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation(
                    "Double random permutation kernel is unavailable",
                ),
            ));
        }
        self.randperm_exec(n, k).map_err(|cause| {
            runmat_accelerate_api::ProviderExecutionError::from(cause).with_unproven_effects()
        })
    }

    fn random_permutation_like(
        &self,
        _prototype: &GpuTensorHandle,
        n: usize,
        k: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.random_permutation(n, k)
    }

    fn polyval(
        &self,
        coeffs: &GpuTensorHandle,
        points: &GpuTensorHandle,
        options: &ProviderPolyvalOptions,
    ) -> Result<GpuTensorHandle> {
        self.polyval_exec(coeffs, points, options)
    }

    fn polyfit<'a>(
        &'a self,
        x: &'a GpuTensorHandle,
        y: &'a GpuTensorHandle,
        degree: usize,
        weights: Option<&'a GpuTensorHandle>,
        outputs: runmat_accelerate_api::ProviderPolyfitOutputs,
    ) -> AccelProviderFuture<'a, ProviderPolyfitResult> {
        Box::pin(async move { self.polyfit_exec(x, y, degree, weights, outputs).await })
    }

    fn polyder_single<'a>(
        &'a self,
        polynomial: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.polyder_exec(polynomial).await })
    }

    fn polyder_product<'a>(
        &'a self,
        p: &'a GpuTensorHandle,
        q: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.polyder_product_exec(p, q).await })
    }

    fn polyder_quotient<'a>(
        &'a self,
        u: &'a GpuTensorHandle,
        v: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, ProviderPolyderQuotient> {
        Box::pin(async move { self.polyder_quotient_exec(u, v).await })
    }

    fn polyint(&self, polynomial: &GpuTensorHandle, constant: f64) -> Result<GpuTensorHandle> {
        self.polyint_exec(polynomial, constant)
    }

    fn diag_from_vector(&self, vector: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.diag_from_vector_exec(vector, offset)
    }

    fn diag_from_vector_sized(
        &self,
        vector: &GpuTensorHandle,
        offset: isize,
        rows: usize,
        cols: usize,
    ) -> Result<GpuTensorHandle> {
        self.diag_from_vector_sized_exec(vector, offset, rows, cols)
    }

    fn diag_extract(&self, matrix: &GpuTensorHandle, offset: isize) -> Result<GpuTensorHandle> {
        self.diag_extract_exec(matrix, offset)
    }

    fn tril<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        offset: isize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.tril_exec(matrix, offset).await })
    }

    fn triu<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        offset: isize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.triu_exec(matrix, offset).await })
    }

    fn reduce_mean_nd<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            self.mean_nd_optional_exec(a, dims_zero_based)
                .await
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })
        })
    }

    fn deep_learning_global_average_pool_1d<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        request: DeepLearningPool1DRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.deep_learning_global_average_pool_1d_exec(a, request) })
    }

    fn deep_learning_global_average_pool_1d_adjoint<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        upstream: &'a GpuTensorHandle,
        request: DeepLearningPool1DRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.deep_learning_pool_1d_adjoint_exec(input, upstream, request) })
    }

    fn deep_learning_global_average_pool_1d_adjoint_capability(
        &self,
        anchor: &GpuTensorHandle,
        input_shape: &[usize],
        upstream: Option<&GpuTensorHandle>,
        request: DeepLearningPool1DRequest,
    ) -> Result<DeepLearningPool1DCapability> {
        self.deep_learning_pool_1d_adjoint_capability_exec(anchor, input_shape, upstream, request)
    }

    fn deep_learning_global_average_pool_1d_capability(
        &self,
        input: &GpuTensorHandle,
        route: &DeepLearningPool1DRoute,
    ) -> Result<DeepLearningPool1DCapability> {
        self.deep_learning_global_average_pool_1d_capability_exec(input, route)
    }

    fn deep_learning_global_average_pool_1d_planned_capability(
        &self,
        anchor: &GpuTensorHandle,
        input_shape: &[usize],
        route: &DeepLearningPool1DRoute,
    ) -> Result<DeepLearningPool1DCapability> {
        self.deep_learning_global_average_pool_1d_planned_capability_exec(
            anchor,
            input_shape,
            route,
        )
    }

    fn deep_learning_conv1d<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        weights: &'a GpuTensorHandle,
        bias: &'a GpuTensorHandle,
        request: &'a DeepLearningConv1DRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.deep_learning_conv1d_exec(input, weights, bias, request) })
    }

    fn deep_learning_conv1d_adjoint<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        weights: &'a GpuTensorHandle,
        bias: &'a GpuTensorHandle,
        upstream: &'a GpuTensorHandle,
        request: &'a DeepLearningConv1DRequest,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::DeepLearningConv1DGradients> {
        Box::pin(async move {
            self.deep_learning_conv1d_adjoint_exec(input, weights, bias, upstream, request)
        })
    }

    fn deep_learning_conv1d_capability(
        &self,
        operands: DeepLearningConv1DOperands<'_>,
        request: &DeepLearningConv1DRequest,
    ) -> Result<DeepLearningConv1DCapability> {
        self.deep_learning_conv1d_capability_exec(operands, request)
    }

    fn deep_learning_layer_norm<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        scale: &'a GpuTensorHandle,
        offset: &'a GpuTensorHandle,
        request: &'a DeepLearningLayerNormRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.deep_learning_layer_norm_exec(input, scale, offset, request) })
    }

    fn deep_learning_layer_norm_recorded<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        scale: &'a GpuTensorHandle,
        offset: &'a GpuTensorHandle,
        request: &'a DeepLearningLayerNormRequest,
    ) -> runmat_accelerate_api::AccelProviderFuture<
        'a,
        runmat_accelerate_api::DeepLearningLayerNormRecorded,
    > {
        Box::pin(async move {
            self.deep_learning_layer_norm_recorded_exec(input, scale, offset, request)
        })
    }

    fn deep_learning_layer_norm_adjoint<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        scale: &'a GpuTensorHandle,
        offset: &'a GpuTensorHandle,
        statistics: &'a GpuTensorHandle,
        upstream: &'a GpuTensorHandle,
        request: &'a DeepLearningLayerNormRequest,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::DeepLearningLayerNormGradients> {
        Box::pin(async move {
            self.deep_learning_layer_norm_adjoint_exec(
                input, scale, offset, statistics, upstream, request,
            )
        })
    }

    fn deep_learning_layer_norm_adjoint_capability(
        &self,
        anchor: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        statistics: Option<&GpuTensorHandle>,
        upstream: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        self.deep_learning_layer_norm_adjoint_capability_exec(
            anchor, scale, offset, statistics, upstream, request,
        )
    }

    fn deep_learning_layer_norm_capability(
        &self,
        input: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        self.deep_learning_layer_norm_capability_exec(input, scale, offset, request)
    }

    fn deep_learning_layer_norm_planned_capability(
        &self,
        anchor: &GpuTensorHandle,
        scale: Option<&GpuTensorHandle>,
        offset: Option<&GpuTensorHandle>,
        request: &DeepLearningLayerNormRequest,
    ) -> Result<DeepLearningLayerNormCapability> {
        self.deep_learning_layer_norm_planned_capability_exec(anchor, scale, offset, request)
    }

    fn deep_learning_feed_forward_stage_capability(
        &self,
        anchor: &GpuTensorHandle,
        stage: runmat_accelerate_api::DeepLearningFeedForwardStage,
    ) -> Result<runmat_accelerate_api::DeepLearningFeedForwardCapability> {
        self.deep_learning_feed_forward_stage_capability_exec(anchor, stage)
    }

    fn deep_learning_forward_route_capability(
        &self,
        route: &runmat_accelerate_api::DeepLearningForwardRoute<'_>,
    ) -> Result<Option<runmat_accelerate_api::DeepLearningForwardCapability>> {
        self.deep_learning_forward_route_capability_exec(route)
            .map(Some)
    }

    fn deep_learning_forward_preparation_snapshot(
        &self,
        route: &runmat_accelerate_api::DeepLearningForwardRoute<'_>,
    ) -> Option<runmat_accelerate_api::ProviderPreparationSnapshot> {
        self.deep_learning_forward_preparation_snapshot_exec(route)
    }

    fn provider_view_chain_capability(
        &self,
        anchor: &GpuTensorHandle,
        chain: &runmat_accelerate_api::ProviderViewChain,
    ) -> Result<runmat_accelerate_api::ProviderViewCapability> {
        self.provider_view_chain_capability_exec(anchor, chain)
    }

    fn reduce_moments_nd<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::ProviderMoments2> {
        Box::pin(async move { self.reduce_moments_nd_exec(a, dims_zero_based) })
    }

    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Add,
                "elem_add",
                a,
                b,
            )
            .await
        })
    }

    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Mul,
                "elem_mul",
                a,
                b,
            )
            .await
        })
    }

    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Sub,
                "elem_sub",
                a,
                b,
            )
            .await
        })
    }

    fn elem_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Max,
                "elem_max",
                a,
                b,
            )
            .await
        })
    }

    fn select<'a>(
        &'a self,
        mask: &'a GpuTensorHandle,
        then_value: &'a GpuTensorHandle,
        else_value: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.select_exec(mask, then_value, else_value) })
    }

    fn elem_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Min,
                "elem_min",
                a,
                b,
            )
            .await
        })
    }

    fn elem_div<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Div,
                "elem_div",
                a,
                b,
            )
            .await
        })
    }

    fn elem_rem<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.get_entry_raw(a)?.integer_type().is_some()
                || self.get_entry_raw(b)?.integer_type().is_some()
            {
                self.integer_remainder_exec(false, "elem_rem", a, b).await
            } else {
                let _ = (self.get_entry(a)?, self.get_entry(b)?);
                self.floating_remainder_exec(a, b, runmat_types::FloatingRemainderKind::Remainder)
                    .map_err(|error| {
                        runmat_accelerate_api::ProviderExecutionError::from(error)
                            .with_unproven_effects()
                    })
            }
        })
    }

    fn elem_mod<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.get_entry_raw(a)?.integer_type().is_some()
                || self.get_entry_raw(b)?.integer_type().is_some()
            {
                self.integer_remainder_exec(true, "elem_mod", a, b).await
            } else {
                let _ = (self.get_entry(a)?, self.get_entry(b)?);
                self.floating_remainder_exec(a, b, runmat_types::FloatingRemainderKind::Modulus)
                    .map_err(|error| {
                        runmat_accelerate_api::ProviderExecutionError::from(error)
                            .with_unproven_effects()
                    })
            }
        })
    }

    fn elem_pow<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let downloaded_exponent = matches!(
                self.get_entry_raw(b)?.integer_type(),
                Some(
                    IntegerElementType::I8
                        | IntegerElementType::I16
                        | IntegerElementType::I32
                        | IntegerElementType::I64
                )
            );
            if downloaded_exponent {
                return self.integer_signed_power_exec(a, b).await;
            }
            self.binary_op_public_exec(
                crate::backend::wgpu::types::BinaryOpCode::Pow,
                "elem_pow",
                a,
                b,
            )
            .await
        })
    }

    fn classify_real_power<'a>(
        &'a self,
        base: &'a GpuTensorHandle,
        exponent: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, runmat_accelerate_api::PowerResultDomain> {
        Box::pin(async move { self.classify_real_power_exec(base, exponent).await })
    }

    fn real_power_in_domain<'a>(
        &'a self,
        base: &'a GpuTensorHandle,
        exponent: &'a GpuTensorHandle,
        domain: runmat_accelerate_api::PowerResultDomain,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.real_power_in_domain_exec(base, exponent, domain) })
    }

    fn complex_from_real<'a>(
        &'a self,
        real: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.complex_from_real_exec(real) })
    }

    fn complex_from_real_imag<'a>(
        &'a self,
        real: &'a GpuTensorHandle,
        imag: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.complex_from_real_imag_exec(real, imag) })
    }

    fn modulate_constellation<'a>(
        &'a self,
        request: ProviderModulationRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.modulate_constellation_exec(&request).await })
    }

    fn modulate_bits_constellation<'a>(
        &'a self,
        request: ProviderBitModulationRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.modulate_bits_constellation_exec(&request).await })
    }

    fn elem_ge<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_ge_exec(a, b).await })
    }

    fn elem_le<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_le_exec(a, b).await })
    }

    fn elem_lt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_lt_exec(a, b).await })
    }

    fn elem_gt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_gt_exec(a, b).await })
    }

    fn elem_eq<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_eq_exec(a, b).await })
    }

    fn elem_ne<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elem_ne_exec(a, b).await })
    }

    fn logical_and(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_and_exec(a, b)
    }

    fn logical_or(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_or_exec(a, b)
    }

    fn logical_xor(&self, a: &GpuTensorHandle, b: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_xor_exec(a, b)
    }

    fn logical_not(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_not_exec(a)
    }

    fn logical_islogical(&self, a: &GpuTensorHandle) -> Result<bool> {
        self.logical_islogical_exec(a)
    }

    fn logical_isreal(&self, a: &GpuTensorHandle) -> Result<bool> {
        self.logical_isreal_exec(a)
    }

    fn logical_isfinite(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isfinite_exec(a)
    }

    fn logical_isnan(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isnan_exec(a)
    }

    fn logical_isinf(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.logical_isinf_exec(a)
    }

    fn elem_hypot<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Hypot, a, b)
        })
    }

    fn elem_atan2<'a>(
        &'a self,
        y: &'a GpuTensorHandle,
        x: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.binary_op_exec(crate::backend::wgpu::types::BinaryOpCode::Atan2, y, x)
        })
    }

    fn unary_sin<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_sin_exec(a) })
    }

    fn unary_sinc<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_sinc_exec(a) })
    }

    fn unary_gamma<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Gamma, a) },
        )
    }

    fn unary_gammaln<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Gammaln, a) },
        )
    }

    fn unary_erf<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Erf, a) },
        )
    }

    fn unary_erfcinv<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Erfcinv, a) },
        )
    }

    fn unary_factorial<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Factorial, a)
        })
    }

    fn unary_asinh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asinh, a) },
        )
    }

    fn unary_sinh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_sinh_exec(a) })
    }

    fn unary_cosh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_cosh_exec(a) })
    }

    fn unary_asin<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Asin, a) },
        )
    }

    fn unary_acos<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acos, a) },
        )
    }

    fn unary_acosh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Acosh, a) },
        )
    }

    fn unary_tan<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_tan_exec(a) })
    }

    fn unary_tanh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Tanh, a) },
        )
    }

    fn unary_atan<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atan, a) },
        )
    }
    fn unary_atanh<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Atanh, a) },
        )
    }

    fn unary_ceil<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Ceil, a) },
        )
    }

    fn unary_floor<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Floor, a) },
        )
    }

    fn unary_round<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Round, a) },
        )
    }

    fn round_digits<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        digits: i32,
        significant: bool,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.round_digits_exec(a, digits, significant).await })
    }

    fn unary_fix<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Fix, a) },
        )
    }

    fn unary_cos<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_cos_exec(a) })
    }

    fn unary_angle<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_angle_exec(a) })
    }

    fn unary_abs<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_abs_exec(a) })
    }

    fn unary_sign<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_sign_exec(a) })
    }

    fn unary_heaviside<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Heaviside, a)
        })
    }

    fn unary_conj<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            let result = (|| -> Result<Option<GpuTensorHandle>> {
                let entry = self.get_entry_raw(a)?;
                if entry.integer_type().is_some() || entry.precision != self.precision {
                    return Ok(None);
                }
                self.unary_conj_exec(a).map(Some)
            })();
            result.map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })
        })
    }

    fn unary_imag<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_imag_exec(a) })
    }

    fn unary_real<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_real_exec(a) })
    }

    fn unary_exp<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Exp, a) },
        )
    }

    fn unary_expm1<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Expm1, a) },
        )
    }

    fn activation_relu<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.relu_exec(a) })
    }

    fn activation_elu<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        alpha: f64,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.elu_exec(a, alpha) })
    }

    fn activation_adjoint_supported(
        &self,
        operation: runmat_accelerate_api::ProviderActivationAdjoint,
        precision: ProviderPrecision,
    ) -> bool {
        let precision_matches = matches!(
            (self.precision, precision),
            (NumericPrecision::F32, ProviderPrecision::F32)
                | (NumericPrecision::F64, ProviderPrecision::F64)
        );
        precision_matches
            && match operation {
                runmat_accelerate_api::ProviderActivationAdjoint::Relu => {
                    self.ieee_words_available(self.precision)
                }
                runmat_accelerate_api::ProviderActivationAdjoint::Elu { alpha } => {
                    alpha.is_finite()
                        && alpha >= 0.0
                        && (self.precision != NumericPrecision::F32 || (alpha as f32).is_finite())
                }
            }
    }

    fn activation_adjoint<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        upstream: &'a GpuTensorHandle,
        operation: runmat_accelerate_api::ProviderActivationAdjoint,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.activation_adjoint_exec(input, upstream, operation) })
    }

    fn activation_softmax_rows<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.softmax_rows_exec(a).await })
    }

    fn unary_log<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log, a) },
        )
    }

    fn unary_log2<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log2, input)
        })
    }

    fn unary_log10<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log10, input)
        })
    }

    fn unary_log1p<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Log1p, a) },
        )
    }

    fn unary_sqrt<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(
            async move { self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::Sqrt, a) },
        )
    }

    fn unary_double<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.floating_cast_exec_typed(a, NumericPrecision::F64)
                .await
        })
    }

    fn unary_double_admitted<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.floating_cast_exec_recorded(a, NumericPrecision::F64, admission, |_| Ok(()))
                .await
        })
    }

    fn unary_single<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.floating_cast_exec_typed(a, NumericPrecision::F32)
                .await
        })
    }

    fn unary_single_admitted<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.floating_cast_exec_recorded(a, NumericPrecision::F32, admission, |_| Ok(()))
                .await
        })
    }

    fn unary_pow2<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.unary_pow2_exec(a) })
    }

    fn unary_nextpow2<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.unary_op_exec(crate::backend::wgpu::types::UnaryOpCode::NextPow2, a)
        })
    }

    fn pow2_scale(
        &self,
        mantissa: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.pow2_scale_exec(mantissa, exponent)
    }

    fn scalar_rsub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::RSub, a, scalar)
    }

    fn scalar_rdiv(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::RDiv, a, scalar)
    }

    fn scalar_add(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Add, a, scalar)
    }

    fn scalar_sub(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Sub, a, scalar)
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Mul, a, scalar)
    }

    fn scalar_max(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Max, a, scalar)
    }

    fn scalar_min(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Min, a, scalar)
    }

    fn scalar_div(&self, a: &GpuTensorHandle, scalar: f64) -> Result<GpuTensorHandle> {
        self.scalar_op_exec(crate::backend::wgpu::types::ScalarOpCode::Div, a, scalar)
    }

    fn sort_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        order: SortOrder,
        comparison: SortComparison,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<SortResult>> {
        let pending = async move { self.sort_dim_exec(a, dim, order, comparison).await };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }
    fn sort_rows<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        columns: &'a [SortRowsColumnSpec],
        comparison: SortComparison,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<SortResult>> {
        let pending = async move { self.sort_rows_exec(a, columns, comparison).await };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }

    fn transpose(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.transpose_exec(a)
    }

    fn permute(
        &self,
        handle: &GpuTensorHandle,
        order: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let allocated = std::cell::Cell::new(false);
        self.permute_optional_exec(handle, order, &allocated, &|_| Ok(()))
            .map_err(|error| {
                let error = runmat_accelerate_api::ProviderExecutionError::from(error);
                if allocated.get() {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                }
            })
    }

    fn flip(
        &self,
        handle: &GpuTensorHandle,
        axes: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let allocated = std::cell::Cell::new(false);
        self.flip_optional_exec(handle, axes, &allocated, &|_| Ok(()))
            .map_err(|error| {
                let error = runmat_accelerate_api::ProviderExecutionError::from(error);
                if allocated.get() {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                }
            })
    }

    fn conv1d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        options: ProviderConv1dOptions,
    ) -> Result<GpuTensorHandle> {
        self.conv1d_exec(signal, kernel, options)
    }
    fn iir_filter<'a>(
        &'a self,
        coefficients: runmat_accelerate_api::ProviderIirCoefficientSource<'a>,
        x: &'a GpuTensorHandle,
        options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        Box::pin(async move { self.iir_filter_exec(coefficients, x, options).await })
    }
    fn iir_zero_phase<'a>(
        &'a self,
        request: &'a runmat_accelerate_api::ProviderIirZeroPhaseRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { crate::iir_zero_phase::execute(self, request).await })
    }
    fn moving_window<'a>(
        &'a self,
        request: &'a ProviderMovingWindowRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.moving_window_exec(request) })
    }
    fn uniform_spectral_estimate<'a>(
        &'a self,
        request: &'a ProviderSpectralRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderSpectralResult> {
        Box::pin(async move { self.uniform_spectral_estimate_exec(request).await })
    }
    fn signal_envelope<'a>(
        &'a self,
        request: &'a ProviderEnvelopeRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderEnvelopeResult> {
        Box::pin(async move { self.signal_envelope_exec(request).await })
    }
    fn supports_signal_envelope(&self, request: &ProviderEnvelopeRequest<'_>) -> bool {
        self.signal_envelope_supported_exec(request)
    }
    fn supports_signal_hilbert(&self, request: &ProviderHilbertRequest<'_>) -> bool {
        self.signal_hilbert_supported_exec(request)
    }

    fn signal_hilbert<'a>(
        &'a self,
        request: &'a ProviderHilbertRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.signal_hilbert_exec(request).await })
    }
    fn conv2d(
        &self,
        signal: &GpuTensorHandle,
        kernel: &GpuTensorHandle,
        mode: ProviderConvMode,
    ) -> Result<GpuTensorHandle> {
        self.conv2d_exec(signal, kernel, mode)
    }

    fn diff_dim(
        &self,
        handle: &GpuTensorHandle,
        order: usize,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle) != Some(self.precision())
            || runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
        {
            return Ok(None);
        }

        self.diff_exec(handle, dim, order)
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
    }

    fn gradient_dim(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        spacing: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle) != Some(self.precision())
        {
            return Ok(None);
        }

        self.gradient_exec(handle, dim, spacing)
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
    }

    fn gradient_dim_with_coordinates(
        &self,
        handle: &GpuTensorHandle,
        dim: usize,
        coordinates: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.gradient_exec_with_coordinates(handle, dim, coordinates)
    }

    fn cumsum_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if !cumulative_scan::admitted(
                self,
                input,
                dim,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )? {
                return Ok(None);
            }
            self.cumsum_exec(input, dim, direction, nan_mode)
                .map(Some)
                .map_err(Into::into)
        })
    }

    fn integer_cumsum_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if !cumulative_scan::admitted(
                self,
                input,
                dim,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )? {
                return Ok(None);
            }
            (|| -> anyhow::Result<GpuTensorHandle> {
                let (values, _) =
                    self.integer_cumulative_scan_exec(0, "cumsum", input, dim, direction)?;
                Ok(values)
            })()
            .map(Some)
            .map_err(Into::into)
        })
    }

    fn trapz_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        spacing: ProviderTrapezoidSpacing<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(input).is_some()
            || runmat_accelerate_api::handle_is_logical(input)
            || runmat_accelerate_api::handle_precision(input) != Some(self.precision())
        {
            return Ok(None);
        }

        self.trapz_exec(input, dim, spacing)
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
    }

    fn cumtrapz_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        spacing: ProviderTrapezoidSpacing<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(input).is_some()
            || runmat_accelerate_api::handle_is_logical(input)
            || runmat_accelerate_api::handle_precision(input) != Some(self.precision())
        {
            return Ok(None);
        }

        self.cumtrapz_exec(input, dim, spacing)
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
    }

    fn cumprod_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if !cumulative_scan::admitted(
                self,
                input,
                dim,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )? {
                return Ok(None);
            }
            self.cumprod_exec(input, dim, direction, nan_mode)
                .map(Some)
                .map_err(Into::into)
        })
    }

    fn integer_cumprod_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if !cumulative_scan::admitted(
                self,
                input,
                dim,
                runmat_accelerate_api::ProviderExtremaDemand::ValuesOnly,
            )? {
                return Ok(None);
            }
            (|| -> anyhow::Result<GpuTensorHandle> {
                let (values, _) =
                    self.integer_cumulative_scan_exec(1, "cumprod", input, dim, direction)?;
                Ok(values)
            })()
            .map(Some)
            .map_err(Into::into)
        })
    }

    fn cummin_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
        demand: runmat_accelerate_api::ProviderExtremaDemand,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        Option<runmat_accelerate_api::ProviderCumminResult>,
    > {
        Box::pin(async move {
            if !cumulative_scan::admitted(self, input, dim, demand)? {
                return Ok(None);
            }
            self.cummin_exec(input, dim, direction, nan_mode)
                .map(Some)
                .map_err(Into::into)
        })
    }

    fn integer_cummin_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        demand: runmat_accelerate_api::ProviderExtremaDemand,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        Option<runmat_accelerate_api::ProviderCumminResult>,
    > {
        Box::pin(async move {
            if !cumulative_scan::admitted(self, input, dim, demand)? {
                return Ok(None);
            }
            (|| -> anyhow::Result<runmat_accelerate_api::ProviderCumminResult> {
                let (values, indices) =
                    self.integer_cumulative_scan_exec(2, "cummin", input, dim, direction)?;
                Ok(runmat_accelerate_api::ProviderCumminResult {
                    values,
                    indices: indices.expect("cummin integer scan returns indices"),
                })
            })()
            .map(Some)
            .map_err(Into::into)
        })
    }

    fn cummax_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        nan_mode: ProviderNanMode,
        demand: runmat_accelerate_api::ProviderExtremaDemand,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        Option<runmat_accelerate_api::ProviderCummaxResult>,
    > {
        Box::pin(async move {
            if !cumulative_scan::admitted(self, input, dim, demand)? {
                return Ok(None);
            }
            self.cummax_exec(input, dim, direction, nan_mode)
                .map(Some)
                .map_err(Into::into)
        })
    }

    fn integer_cummax_scan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        direction: ProviderScanDirection,
        demand: runmat_accelerate_api::ProviderExtremaDemand,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<
        'a,
        Option<runmat_accelerate_api::ProviderCummaxResult>,
    > {
        Box::pin(async move {
            if !cumulative_scan::admitted(self, input, dim, demand)? {
                return Ok(None);
            }
            (|| -> anyhow::Result<runmat_accelerate_api::ProviderCummaxResult> {
                let (values, indices) =
                    self.integer_cumulative_scan_exec(3, "cummax", input, dim, direction)?;
                Ok(runmat_accelerate_api::ProviderCummaxResult {
                    values,
                    indices: indices.expect("cummax integer scan returns indices"),
                })
            })()
            .map(Some)
            .map_err(Into::into)
        })
    }

    fn supports_circshift(&self, handle: &GpuTensorHandle, shifts: &[isize]) -> bool {
        if handle.allocation_lease().validate_live().is_err()
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_transpose_info(handle).is_some()
            || !runmat_accelerate_api::provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, self))
        {
            return false;
        }
        let Ok(entry) = self.get_entry_for_storage_move(handle, "circshift") else {
            return false;
        };
        // The current kernel/register path produces real storage. Complex
        // interleaved inputs need the exact class-preserving fallback.
        if entry.storage != runmat_accelerate_api::GpuTensorStorage::Real || entry.logical {
            return false;
        }
        let mut shape = if entry.shape.is_empty() {
            vec![1]
        } else {
            entry.shape.clone()
        };
        shape.resize(shape.len().max(shifts.len()), 1);
        if shape.len() > crate::backend::wgpu::params::CIRCSHIFT_MAX_RANK
            || entry.len > u32::MAX as usize
            || shape
                .iter()
                .any(|dimension| *dimension > u32::MAX as usize || *dimension > isize::MAX as usize)
        {
            return false;
        }
        let mut stride = 1usize;
        for dimension in &shape {
            if stride > u32::MAX as usize {
                return false;
            }
            let Some(next) = stride.checked_mul((*dimension).max(1)) else {
                return false;
            };
            stride = next;
        }
        shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            == Some(entry.len)
            && entry
                .len
                .checked_mul(
                    if entry.integer_type().is_some_and(|integer| {
                        matches!(
                            integer,
                            runmat_accelerate_api::IntegerElementType::I64
                                | runmat_accelerate_api::IntegerElementType::U64
                        )
                    }) {
                        2
                    } else {
                        1
                    },
                )
                .is_some()
    }

    fn circshift(&self, handle: &GpuTensorHandle, shifts: &[isize]) -> Result<GpuTensorHandle> {
        self.circshift_exec(handle, shifts)
    }

    fn fft_dim<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft_dim_exec(handle, len, dim).await })
    }

    fn ifft_dim<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        len: Option<usize>,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.ifft_dim_exec(handle, len, dim).await })
    }

    fn fft_extract_real<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fft_extract_real_exec(handle).await })
    }

    fn unique<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: &'a UniqueOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<UniqueResult>> {
        let pending = async move { self.unique_exec(handle, options).await };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }
    fn ismember<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a IsMemberOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<IsMemberResult>> {
        let pending = async move { self.ismember_exec(a, b, options).await };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }

    fn union<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a UnionOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<UnionResult>> {
        let pending = async move { self.union_exec(a, b, options).await };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }
    fn setdiff<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        options: &'a SetdiffOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<SetdiffResult>> {
        let pending = async move { self.setdiff_exec(a, b, options).await };
        Box::pin(async move {
            pending.await.map(Some).map_err(
                |cause: runmat_accelerate_api::ProviderExecutionError| {
                    cause.with_unproven_effects()
                },
            )
        })
    }

    fn cat(&self, dim: usize, inputs: &[GpuTensorHandle]) -> Result<GpuTensorHandle> {
        self.cat_exec(dim, inputs)
    }

    fn cat_optional<'a>(
        &'a self,
        dim: usize,
        inputs: &'a [GpuTensorHandle],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.cat_optional_exec(dim, inputs).await })
    }

    fn repmat(
        &self,
        handle: &GpuTensorHandle,
        reps: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let allocated = std::cell::Cell::new(false);
        self.repmat_optional_exec(handle, reps, &allocated, &|_| Ok(()))
            .map_err(|error| {
                let error = runmat_accelerate_api::ProviderExecutionError::from(error);
                if allocated.get() {
                    error.after_effect()
                } else {
                    error.with_unproven_effects()
                }
            })
    }

    fn kron<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.kron_exec(a, b).await })
    }
    fn cross(
        &self,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        self.cross_exec(lhs, rhs, dim)
    }
    fn reshape(&self, handle: &GpuTensorHandle, new_shape: &[usize]) -> Result<GpuTensorHandle> {
        self.reshape_exec(handle, new_shape)
    }

    fn lu<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move { self.lu_exec(a).await })
    }

    fn chol<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        lower: bool,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderCholResult> {
        Box::pin(async move { self.chol_exec(a, lower).await })
    }
    fn qr<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
        options: ProviderQrOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderQrResult> {
        Box::pin(async move { self.qr_exec(handle, options).await })
    }

    fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        self.take_matmul_sources_exec(product)
    }

    fn qr_power_iter<'a>(
        &'a self,
        product: &'a GpuTensorHandle,
        product_lhs: Option<&'a GpuTensorHandle>,
        q_handle: &'a GpuTensorHandle,
        options: &'a ProviderQrOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<ProviderQrPowerIterResult>> {
        Box::pin(async move {
            self.qr_power_iter_exec(product, product_lhs, q_handle, options)
                .await
        })
    }
    fn matmul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_exec(a, b) })
    }

    fn syrk(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.syrk_exec(a)
    }
    fn matmul_epilogue<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        ep: &'a runmat_accelerate_api::MatmulEpilogue,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_epilogue_exec(a, b, ep).await })
    }

    fn pagefun(&self, request: &PagefunRequest) -> Result<GpuTensorHandle> {
        self.pagefun_exec(request)
    }
    fn image_normalize<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        desc: &'a runmat_accelerate_api::ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.image_normalize_exec(input, desc).await })
    }
    fn matmul_power_step<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        epilogue: &'a runmat_accelerate_api::PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.matmul_power_step_exec(lhs, rhs, epilogue).await })
    }
    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.covariance_with_optional_exec(matrix, second, weights, options)
                .await
                .map_err(Into::into)
        })
    }
    fn corrcoef<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: &'a CorrcoefOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.corrcoef_exec(matrix, options)
                .await
                .map_err(Into::into)
        })
    }

    fn covariance_to_correlation(
        &self,
        matrix: &GpuTensorHandle,
    ) -> Result<ProviderCovarianceToCorrelationResult> {
        self.covariance_to_correlation_exec(matrix)
    }

    fn linsolve<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        options: &'a ProviderLinsolveOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ProviderLinsolveResult> {
        Box::pin(async move { self.linsolve_exec(lhs, rhs, options).await })
    }
    fn inv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.inv_exec(matrix).await })
    }

    fn pinv<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        options: ProviderPinvOptions,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.pinv_exec(matrix, options).await })
    }

    fn cond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        norm: ProviderCondNorm,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.cond_exec(matrix, norm).await })
    }

    fn norm<'a>(
        &'a self,
        tensor: &'a GpuTensorHandle,
        order: ProviderNormOrder,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.norm_exec(tensor, order).await })
    }

    fn interp1<'a>(
        &'a self,
        request: &'a ProviderInterp1Request<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.interp1_exec(request).await })
    }

    fn rank<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        tolerance: Option<f64>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.rank_exec(matrix, tolerance).await })
    }

    fn rcond<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.rcond_exec(matrix).await })
    }

    fn mldivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.mldivide_exec(lhs, rhs).await })
    }

    fn mrdivide<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.mrdivide_exec(lhs, rhs).await })
    }

    fn dot<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        dim: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.dot_exec(lhs, rhs, dim) })
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let entry = self
                .get_entry_raw(a)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            if entry.storage == GpuTensorStorage::ComplexInterleaved {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "complex sum requires a complex floating reduction kernel",
                )
                .into());
            }
            if self.precision == crate::backend::wgpu::types::NumericPrecision::F32 {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "double-accumulating sum requires shader-f64 support",
                )
                .into());
            }
            if entry.integer_type().is_some() {
                let checkpoint = sum_submission_checkpoint();
                let result = self.integer_floating_sum_dim_exec(a, dim).await;
                if result.is_ok() && entry.len != 0 {
                    confirm_sum_kernel_submission(self, checkpoint);
                }
                return result;
            }
            let checkpoint = sum_submission_checkpoint();
            let result = if entry.precision == crate::backend::wgpu::types::NumericPrecision::F32 {
                self.single_double_sum_dim_exec(a, dim).await
            } else if a.shape().len() > 2 {
                self.reduce_nd_sum_f64_exec(a, dim).await
            } else {
                self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
                    .map_err(runmat_accelerate_api::ProviderExecutionError::from)
            };
            if (result.is_ok() && entry.len != 0)
                || (entry.precision == crate::backend::wgpu::types::NumericPrecision::F64
                    && result.is_err())
            {
                confirm_sum_kernel_submission(self, checkpoint);
            }
            result
        })
    }
    fn real_sum_native_dim_supported(&self, precision: ProviderPrecision) -> bool {
        matches!(
            (self.precision, precision),
            (
                crate::backend::wgpu::types::NumericPrecision::F32,
                ProviderPrecision::F32
            ) | (
                crate::backend::wgpu::types::NumericPrecision::F64,
                ProviderPrecision::F64
            )
        ) && self.pipelines.reduce_dim_sum_mean.preflight().is_ok()
    }
    fn reduce_real_sum_native_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let entry = self
                .get_entry_raw(a)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            if entry.storage != GpuTensorStorage::Real
                || entry.integer_type().is_some()
                || runmat_accelerate_api::handle_is_logical(a)
                || entry.precision != self.precision
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "native real sum requires the provider's real floating precision",
                )
                .into());
            }
            self.reduce_dim_sum_mean_exec(a, dim, crate::backend::wgpu::types::DimReduceOp::Sum)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)
        })
    }

    fn reduce_integer_sum_native<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.integer_reduce_sum_prod_global_exec(false, "sum", a) })
    }
    fn reduce_integer_sum_native_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.integer_reduce_sum_prod_dim_exec(false, dim, "sum", a) })
    }
    fn reduce_nnz_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(a).is_some()
            || runmat_accelerate_api::handle_is_logical(a)
            || runmat_accelerate_api::handle_precision(a) != Some(self.precision())
            || runmat_accelerate_api::handle_storage(a) != GpuTensorStorage::Real
            || dim >= a.shape().len()
        {
            return Box::pin(async { Ok(None) });
        }

        Box::pin(async move {
            (async {
                self.reduce_dim_sum_mean_exec(
                    a,
                    dim,
                    crate::backend::wgpu::types::DimReduceOp::CountNonZero,
                )
            })
            .await
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
    fn reduce_prod_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(a).is_some()
            || runmat_accelerate_api::handle_is_logical(a)
            || runmat_accelerate_api::handle_precision(a) != Some(self.precision())
            || runmat_accelerate_api::handle_storage(a) != GpuTensorStorage::Real
        {
            return Box::pin(async { Ok(None) });
        }

        Box::pin(async move {
            (async {
                self.reduce_dim_sum_mean_exec(
                    a,
                    dim,
                    crate::backend::wgpu::types::DimReduceOp::Prod,
                )
            })
            .await
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
    fn reduce_integer_prod_native<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            (async { self.integer_reduce_sum_prod_global_exec(true, "prod", a) })
                .await
                .map(Some)
                .map_err(|source| {
                    runmat_accelerate_api::ProviderExecutionError::from(source)
                        .with_unproven_effects()
                })
        })
    }
    fn reduce_integer_prod_native_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            (async { self.integer_reduce_sum_prod_dim_exec(true, dim, "prod", a) })
                .await
                .map(Some)
                .map_err(|source| {
                    runmat_accelerate_api::ProviderExecutionError::from(source)
                        .with_unproven_effects()
                })
        })
    }
    fn reduce_integer_mean_native<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.integer_reduce_mean_global_exec("mean", a) })
    }
    fn reduce_integer_mean_native_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.integer_reduce_mean_dim_exec(dim, "mean", a) })
    }
    fn reduce_integer_mean_native_dims<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.integer_reduce_mean_dims_exec(dims_zero_based, "mean", a) })
    }
    fn cast_to_integer<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        target: IntegerElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.cast_to_integer_exec(a, target).await })
    }
    fn reduce_mean_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            self.mean_dim_optional_exec(a, dim).await.map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })
        })
    }
    fn reduce_any_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_any_dim_exec(a, dim, omit_nan) })
    }
    fn reduce_any<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_any_exec(a, omit_nan) })
    }

    fn reduce_all_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_all_dim_exec(a, dim, omit_nan) })
    }

    fn reduce_all<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_all_exec(a, omit_nan) })
    }

    fn reduce_median<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_median_exec(a).await })
    }

    fn reduce_median_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reduce_median_dim_exec(a, dim).await })
    }

    fn preflight_double_sum(
        &self,
        a: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderDoubleSumPreflight {
        if self.precision == crate::backend::wgpu::types::NumericPrecision::F32
            || runmat_accelerate_api::handle_storage(a) == GpuTensorStorage::ComplexInterleaved
        {
            runmat_accelerate_api::ProviderDoubleSumPreflight::Unsupported
        } else {
            runmat_accelerate_api::ProviderDoubleSumPreflight::Unknown
        }
    }

    fn reduce_sum<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let entry = self
                .get_entry_raw(a)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            if entry.storage == GpuTensorStorage::ComplexInterleaved {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "complex sum requires a complex floating reduction kernel",
                )
                .into());
            }
            if self.precision == crate::backend::wgpu::types::NumericPrecision::F32 {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "double-accumulating sum requires shader-f64 support",
                )
                .into());
            }
            if entry.integer_type().is_some() {
                let checkpoint = sum_submission_checkpoint();
                let result = self.integer_floating_sum_global_exec(a).await;
                if result.is_ok() && entry.len > 1 {
                    confirm_sum_kernel_submission(self, checkpoint);
                }
                return result;
            }
            let checkpoint = sum_submission_checkpoint();
            let result = if entry.precision == crate::backend::wgpu::types::NumericPrecision::F32 {
                self.single_double_sum_global_exec(a).await
            } else {
                self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Sum)
                    .map_err(runmat_accelerate_api::ProviderExecutionError::from)
            };
            if (result.is_ok() && entry.len > 1)
                || (entry.precision == crate::backend::wgpu::types::NumericPrecision::F64
                    && result.is_err())
            {
                confirm_sum_kernel_submission(self, checkpoint);
            }
            result
        })
    }

    fn reduce_nnz<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(a).is_some()
            || runmat_accelerate_api::handle_is_logical(a)
            || runmat_accelerate_api::handle_precision(a) != Some(self.precision())
            || runmat_accelerate_api::handle_storage(a) != GpuTensorStorage::Real
        {
            return Box::pin(async { Ok(None) });
        }

        Box::pin(async move {
            (async {
                self.reduce_global_exec(
                    a,
                    crate::backend::wgpu::types::GlobalReduceOp::CountNonZero,
                )
            })
            .await
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }

    fn reduce_prod<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        // Pure metadata admission: decline before registry reads, staging, or execution.
        if runmat_accelerate_api::handle_integer_type(a).is_some()
            || runmat_accelerate_api::handle_is_logical(a)
            || runmat_accelerate_api::handle_precision(a) != Some(self.precision())
            || runmat_accelerate_api::handle_storage(a) != GpuTensorStorage::Real
        {
            return Box::pin(async { Ok(None) });
        }

        Box::pin(async move {
            (async {
                self.reduce_global_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Prod)
            })
            .await
            .map(Some)
            .map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }

    fn reduce_mean<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            self.mean_optional_exec(a).await.map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })
        })
    }
    fn reduce_std<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.typed_std_exec(a, None, normalization, nan_mode).await })
    }

    fn reduce_std_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.typed_std_exec(a, Some(dim), normalization, nan_mode)
                .await
        })
    }

    fn reduce_min<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.typed_global_extrema_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Min)
                .await
        })
    }

    fn reduce_max<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.typed_global_extrema_exec(a, crate::backend::wgpu::types::GlobalReduceOp::Max)
                .await
        })
    }

    fn reduce_min_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            let checkpoint =
                extrema_submission_checkpoint(runmat_accelerate_api::ProviderExtremaKind::Minimum);
            let result = self
                .typed_dimensional_extrema_exec(
                    a,
                    dim,
                    crate::backend::wgpu::types::DimReduceExtrema::Min,
                )
                .await;
            confirm_extrema_kernel_submission(self, checkpoint);
            result
        })
    }

    fn reduce_max_dim<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            let checkpoint =
                extrema_submission_checkpoint(runmat_accelerate_api::ProviderExtremaKind::Maximum);
            let result = self
                .typed_dimensional_extrema_exec(
                    a,
                    dim,
                    crate::backend::wgpu::types::DimReduceExtrema::Max,
                )
                .await;
            confirm_extrema_kernel_submission(self, checkpoint);
            result
        })
    }

    fn supports_find(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> bool {
        if !matches!(direction, FindDirection::First)
            || runmat_accelerate_api::handle_precision(a)
                != Some(runmat_accelerate_api::ProviderPrecision::F64)
            || runmat_accelerate_api::handle_storage(a)
                != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_integer_type(a).is_some()
            || runmat_accelerate_api::handle_is_logical(a)
        {
            return false;
        }
        let Some(total) = a
            .shape()
            .iter()
            .try_fold(1usize, |acc, extent| acc.checked_mul(*extent))
        else {
            return false;
        };
        let rows = a.shape().first().copied().unwrap_or(1).max(1);
        total <= u32::MAX as usize
            && rows <= u32::MAX as usize
            && limit.unwrap_or(total).min(total) <= u32::MAX as usize
    }

    fn find(
        &self,
        a: &GpuTensorHandle,
        limit: Option<usize>,
        direction: FindDirection,
    ) -> Result<ProviderFindResult> {
        self.find_exec(a, limit, direction)
    }
    fn issymmetric(
        &self,
        matrix: &GpuTensorHandle,
        kind: ProviderSymmetryKind,
        tolerance: f64,
    ) -> Result<bool> {
        self.issymmetric_exec(matrix, kind, tolerance)
    }

    fn ishermitian<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        kind: ProviderHermitianKind,
        tolerance: f64,
    ) -> AccelProviderFuture<'a, bool> {
        Box::pin(async move { self.ishermitian_exec(matrix, kind, tolerance).await })
    }

    fn bandwidth(&self, matrix: &GpuTensorHandle) -> Result<ProviderBandwidth> {
        self.bandwidth_exec(matrix)
    }

    fn sym_rcm<'a>(&'a self, matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move { self.sym_rcm_exec(matrix).await })
    }
    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> Result<f64> {
        self.read_scalar_exec(h, linear_index)
    }

    fn fused_elementwise<'a>(
        &'a self,
        shader: &'a str,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &'a [GpuTensorHandle],
        output_shape: &'a [usize],
        len: usize,
        output_class: runmat_accelerate_api::ProviderElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.require_selected_shader_requirements(requirements)?;
            let frame = self
                .plan_ordinary_single_fusion(shader, inputs, output_shape, len, output_class)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            let validated = self
                .prepare_validated_fusion_pipeline(
                    frame.recipe.pipeline(),
                    inputs.len(),
                    super::pipeline_recipe::FusionPipelineLayout::Single,
                )
                .await?;
            let observed = ElementwiseKernelObservation::begin(len);
            let output = self
                .fused_elementwise_validated_with_telemetry_exec(
                    shader,
                    inputs,
                    output_shape,
                    len,
                    &frame,
                    &validated,
                )
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            if let Some(observed) = observed {
                observed.submitted(self.device_id_exec());
            }
            Ok(output)
        })
    }

    fn supports_fused_elementwise_abi(
        &self,
        abi: runmat_accelerate_api::ProviderFusedElementwiseAbi,
    ) -> bool {
        abi.precision() == runmat_accelerate_api::ProviderPrecision::F32
            || self.precision == NumericPrecision::F64
    }

    fn fused_elementwise_typed<'a>(
        &'a self,
        request: runmat_accelerate_api::ProviderFusedElementwiseRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.execute_typed_elementwise(request).await })
    }
    fn fused_elementwise_multi_typed<'a>(
        &'a self,
        request: runmat_accelerate_api::ProviderFusedElementwiseRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move { self.execute_typed_elementwise_multi(request).await })
    }

    fn fused_elementwise_multi<'a>(
        &'a self,
        shader: &'a str,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &'a [GpuTensorHandle],
        output_shape: &'a [usize],
        len: usize,
        output_classes: &'a [runmat_accelerate_api::ProviderElementType],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move {
            self.require_selected_shader_requirements(requirements)?;
            let frame = self
                .plan_ordinary_multi_fusion(shader, inputs, output_shape, len, output_classes)
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            let validated = self
                .prepare_validated_fusion_pipeline(
                    &frame.recipe,
                    inputs.len(),
                    super::pipeline_recipe::FusionPipelineLayout::Multiple(output_classes.len()),
                )
                .await?;
            let observed = ElementwiseKernelObservation::begin(len);
            let outputs = self
                .fused_elementwise_multi_validated_with_telemetry_exec(
                    shader, inputs, &frame, &validated,
                )
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error)
                        .with_unproven_effects()
                })?;
            if let Some(observed) = observed {
                observed.submitted(self.device_id_exec());
            }
            Ok(outputs)
        })
    }

    fn map_nan_to_zero(
        &self,
        a: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.map_nan_to_zero_exec_typed(a)
    }
    fn not_nan_mask(&self, a: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        self.not_nan_mask_exec(a)
    }

    fn fused_reduction(
        &self,
        shader: &str,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
        inputs: &[GpuTensorHandle],
        output_shape: &[usize],
        reduce_len: usize,
        num_slices: usize,
        workgroup_size: u32,
        flavor: ReductionFlavor,
    ) -> Result<GpuTensorHandle> {
        self.fused_reduction_with_telemetry_exec(
            crate::backend::wgpu::provider::backend::telemetry::FusedReductionTelemetryRequest {
                shader,
                requirements,
                inputs,
                output_shape,
                reduce_len,
                num_slices,
                workgroup_size,
                flavor,
            },
        )
    }

    fn reduction_recipe_config(
        &self,
        reduce_elements: u32,
        slices: u32,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_accelerate_api::reduction::ReductionRecipeConfig,
    > {
        self.typed_reduction_config(reduce_elements, slices)
    }

    fn reduction_recipe_candidates(
        &self,
        recipe: &runmat_accelerate_api::reduction::ProviderReductionRecipe,
        host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        [Option<runmat_accelerate_api::reduction::ReductionStrategy>; 2],
    > {
        self.typed_reduction_candidates(recipe, host_metadata)
    }

    fn submit_reduction_recipe(
        &self,
        submission: &runmat_accelerate_api::reduction::ReductionSubmission<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.submit_reduction_recipe_exec(submission)
    }

    fn supports_shader_requirements(
        &self,
        requirements: runmat_accelerate_api::ProviderShaderRequirements,
    ) -> bool {
        self.supports_selected_shader_requirements(requirements)
    }
    fn warmup(&self) {
        self.warmup_exec();
    }
    fn fused_cache_counters(&self) -> (u64, u64) {
        self.fused_cache_counters_exec()
    }

    fn last_warmup_millis(&self) -> Option<u64> {
        self.last_warmup_millis_exec()
    }

    fn telemetry_snapshot(&self) -> runmat_accelerate_api::ProviderTelemetry {
        self.telemetry_snapshot_exec()
    }

    fn reset_telemetry(&self) {
        self.reset_telemetry_exec();
    }

    fn default_reduction_workgroup_size(&self) -> u32 {
        self.default_reduction_workgroup_size_exec()
    }

    fn two_pass_threshold(&self) -> usize {
        self.two_pass_threshold_exec()
    }

    fn reduction_two_pass_mode(&self) -> ReductionTwoPassMode {
        self.reduction_two_pass_mode_exec()
    }

    fn scatter_column<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        col_index: usize,
        values: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.scatter_column_exec(matrix, col_index, values).await })
    }
    fn scatter_row<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        row_index: usize,
        values: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.scatter_row_exec(matrix, row_index, values).await })
    }

    fn sub2ind<'a>(
        &'a self,
        dims: &'a [usize],
        strides: &'a [usize],
        inputs: &'a [&'a GpuTensorHandle],
        scalar_mask: &'a [bool],
        len: usize,
        output_shape: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.sub2ind_exec(dims, strides, inputs, scalar_mask, len, output_shape)
                .await
        })
    }

    fn supports_ind2sub(&self) -> bool {
        true
    }

    fn ind2sub<'a>(
        &'a self,
        dims: &'a [usize],
        strides: &'a [usize],
        indices: &'a GpuTensorHandle,
        total: usize,
        len: usize,
        output_shape: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move {
            self.ind2sub_exec(dims, strides, indices, total, len, output_shape)
                .await
        })
    }

    fn upload(&self, host: &HostTensorView) -> Result<GpuTensorHandle> {
        let output = self.upload_exec(host)?;
        complete_observed_upload(self)?;
        Ok(output)
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move { self.download_exec(h).await })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        upload_numeric_with_admission(self, host, None, None)
    }
    fn upload_numeric_raw_admitted(
        &self,
        host: &HostNumericTensorView,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        upload_numeric_with_admission(self, host, admission, None)
    }
    fn numeric_upload_encoding_words(
        &self,
        element: runmat_accelerate_api::NumericElementType,
        lanes: usize,
    ) -> Option<usize> {
        match element.integer_type() {
            Some(element) => super::io::integer_encoding_words(element, lanes),
            None => Some(0),
        }
    }
    fn upload_numeric_raw_admitted_with_scratch(
        &self,
        host: &HostNumericTensorView,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<runmat_accelerate_api::HostNumericUploadScratch<'_>>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        upload_numeric_with_admission(self, host, admission, scratch)
    }
    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.upload_logical_raw_admitted(host, None, None)
    }
    fn upload_logical_raw_admitted(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<runmat_accelerate_api::HostLogicalUploadScratch<'_>>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        let buffer_created = std::cell::Cell::new(false);
        let output = self
            .upload_logical_exec_observed_admitted(
                host,
                || {
                    buffer_created.set(true);
                    Ok(())
                },
                admission,
                scratch,
            )
            .map_err(|error| numeric_upload_failure(error, buffer_created.get()))?;
        if let Err(error) = complete_observed_upload(self) {
            let primary = runmat_accelerate_api::ProviderExecutionError::from(error).after_effect();
            let cleanup = output.release().err().into_iter().collect();
            return Err(runmat_accelerate_api::ProviderExecutionError::from_source(
                runmat_accelerate_api::ProviderExecutionCompletionError::new(
                    Some(primary),
                    cleanup,
                ),
            )
            .after_effect());
        }
        Ok(output)
    }
    fn download_numeric<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move { self.download_numeric_exec(h).await })
    }
    fn download_numeric_readback<'a>(
        &'a self,
        readback: &'a runmat_accelerate_api::NumericReadback<'_>,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(self.download_numeric_exec_with_claim_and_stage(
            readback.source(),
            None,
            None,
            Some(readback),
        ))
    }
    fn prepare_numeric_readback_stage(
        &self,
        readback: &runmat_accelerate_api::NumericReadback<'_>,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
        completion_claim: std::sync::Arc<
            dyn runmat_execution::resource::RetainedHostAllocationClaim,
        >,
    ) -> anyhow::Result<runmat_accelerate_api::PreparedNumericDownloadStage> {
        readback.authorize_provider(self)?;
        let source = readback.source();
        anyhow::ensure!(
            self.numeric_download_feasibility_exec(source, query)
                .is_supported(),
            "exact numeric readback staging source/query is unavailable"
        );
        let runmat_execution::ResourceByteBound::Known(scratch) =
            self.download_numeric_provider_scratch_bound(source)
        else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "numeric readback staging bound",
            ));
        };
        let entry = self.get_entry_raw(source)?;
        let staging = if entry.len == 0 {
            None
        } else {
            let physical = runmat_accelerate_api::WgpuNumericReadbackSource::new(
                &self.emission_authority,
                self,
                source,
                &entry.buffer,
            )?;
            Some(super::allocation_state::NumericStageBuffer::Readback(
                physical.create_stage(entry.allocated_bytes)?,
            ))
        };
        AllocationState::prepare_numeric_download_stage(
            &self.allocations,
            source,
            scratch,
            self.download_numeric_queue_slots(source),
            staging,
            completion_claim,
        )
    }
    fn download_numeric_readback_admitted_stage<'a>(
        &'a self,
        readback: &'a runmat_accelerate_api::NumericReadback<'_>,
        stage: runmat_accelerate_api::PreparedNumericDownloadStage,
    ) -> AccelNumericDownloadFuture<'a> {
        let claim = stage.completion_claim();
        Box::pin(self.download_numeric_exec_with_claim_and_stage(
            readback.source(),
            Some(claim),
            Some(stage),
            Some(readback),
        ))
    }
    fn download_numeric_admitted<'a>(
        &'a self,
        h: &'a GpuTensorHandle,
        completion_claim: std::sync::Arc<
            dyn runmat_execution::resource::RetainedHostAllocationClaim,
        >,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(self.download_numeric_exec_with_claim_and_stage(
            h,
            Some(completion_claim),
            None,
            None,
        ))
    }
    fn prepare_numeric_download_stage(
        &self,
        source: &GpuTensorHandle,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
        completion_claim: std::sync::Arc<
            dyn runmat_execution::resource::RetainedHostAllocationClaim,
        >,
    ) -> anyhow::Result<runmat_accelerate_api::PreparedNumericDownloadStage> {
        use runmat_execution::ResourceByteBound;
        if !self
            .numeric_download_feasibility_exec(source, query)
            .is_supported()
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "exact numeric download staging source/query",
            ));
        }
        let ResourceByteBound::Known(scratch) =
            self.download_numeric_provider_scratch_bound(source)
        else {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "numeric download staging bound",
            ));
        };
        let entry = self.get_entry_raw(source)?;
        let staging = if entry.len == 0 {
            None
        } else {
            Some(self.with_emission(|emission, _| {
                runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                    &wgpu::BufferDescriptor {
                        label: Some("runmat-numeric-download-staging"),
                        size: entry.allocated_bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                ))
            })?)
        };
        AllocationState::prepare_numeric_download_stage(
            &self.allocations,
            source,
            scratch,
            self.download_numeric_queue_slots(source),
            staging.map(Into::into),
            completion_claim,
        )
    }
    fn download_numeric_admitted_stage<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
        stage: runmat_accelerate_api::PreparedNumericDownloadStage,
    ) -> AccelNumericDownloadFuture<'a> {
        let claim = stage.completion_claim();
        Box::pin(self.download_numeric_exec_with_claim_and_stage(
            source,
            Some(claim),
            Some(stage),
            None,
        ))
    }
    fn download_numeric_completion_host_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        let future = self.download_numeric_exec_with_claim_and_stage(handle, None, None, None);
        let future_bytes = u64::try_from(std::mem::size_of_val(&future)).ok();
        super::core::map_completion_host_upper_bytes()
            .and_then(|callback| callback.checked_add(future_bytes?))
            .and_then(|bytes| {
                // Common map control already funds the one stage owner/fact.
                // A selected numeric stage adds just its two boxed handles.
                bytes.checked_add(
                    super::allocation_state::numeric_download_stage_host_bytes()?
                        .checked_sub(super::allocation_state::mapped_stage_host_bytes()?)?,
                )
            })
            .map(runmat_execution::ResourceByteBound::Known)
            .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            ))
    }
    fn download_numeric_host_workspace_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        use runmat_execution::{ResourceBoundUnavailable, ResourceByteBound};

        // Decoding allocates the returned native vector directly from mapped
        // staging bytes. A 2-D lazy transpose clones that vector once before
        // replacing it. Entry lookup constructs one logical shape Vec, which
        // overlaps the returned native shape until this call returns. The
        // mapped wgpu buffer is provider-owned scratch.
        let Some(payload) =
            runmat_accelerate_api::checked_numeric_download_host_payload_bytes(handle)
        else {
            return ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        };
        let extra =
            if handle.shape().len() == 2 && runmat_accelerate_api::handle_is_transposed(handle) {
                payload
            } else {
                0
            };
        let shapes = handle
            .shape()
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| u64::try_from(bytes).ok());
        payload
            .checked_add(extra)
            .and_then(|bytes| bytes.checked_add(shapes?))
            .map(ResourceByteBound::Known)
            .unwrap_or(ResourceByteBound::Unavailable(
                ResourceBoundUnavailable::NoScratchBound,
            ))
    }
    fn download_numeric_provider_scratch_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        // download_numeric_exec maps one staging buffer sized to the owning
        // source buffer. Empty inputs allocate no staging buffer.
        runmat_execution::ResourceByteBound::Known(if handle.physical().element_capacity() == 0 {
            0
        } else {
            handle.physical().requested_bytes()
        })
    }
    fn download_numeric_queue_slots(&self, handle: &GpuTensorHandle) -> u32 {
        // download_numeric_exec returns directly for the provider's genuine
        // empty payload; no staging, encoder, or submission is created.
        u32::from(handle.physical().payload_bytes() != 0)
    }

    fn produced_numeric_readback_evidence(
        &self,
        output: &runmat_accelerate_api::ProviderRepresentation,
    ) -> Option<runmat_accelerate_api::ProviderProducedNumericReadbackEvidence> {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderLayout, ProviderResidency, ProviderStorage,
        };
        use runmat_execution::ResourceByteBound;
        if output.residency != ProviderResidency::Device
            || output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || !matches!(
                output.element_type,
                ProviderElementType::F32 | ProviderElementType::F64
            )
        {
            return None;
        }
        let bytes = output.checked_byte_len()?;
        // The selected route must validate the actual produced handle before
        // readback. For a nonempty contiguous native buffer, the WGPU staging
        // copy has precisely its requested physical extent and one submission.
        if bytes == 0 || bytes > self.adapter_limits.max_buffer_size {
            return None;
        }
        Some(
            runmat_accelerate_api::ProviderProducedNumericReadbackEvidence {
                source_requested_bytes: bytes,
                transfer_bytes: bytes,
                host_workspace_bytes: ResourceByteBound::Known(bytes),
                provider_scratch_bytes: ResourceByteBound::Known(bytes),
                queue_slots: 1,
            },
        )
    }
    fn upload_integer(&self, host: &HostIntegerTensorView) -> Result<GpuTensorHandle> {
        let output = self.upload_integer_exec(host)?;
        complete_observed_upload(self)?;
        Ok(output)
    }
    fn download_integer<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelIntegerDownloadFuture<'a> {
        Box::pin(async move { self.download_integer_exec(h).await })
    }

    fn device_info(&self) -> String {
        self.device_info_exec()
    }

    fn acquire_live(
        &self,
        wire: &runmat_accelerate_api::GpuTensorWireDescriptor,
    ) -> std::result::Result<
        GpuTensorHandle,
        runmat_accelerate_api::allocation::AllocationAcquisitionError,
    > {
        self.allocations.acquire_tensor(wire)
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        self.device_info_struct_exec()
    }
}
