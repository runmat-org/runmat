use super::*;

impl SelectedStochasticProvider {
    /// One selected provider route. The original source stores t and the full
    /// logical Z before any tile upload or arithmetic; a later provider error
    /// therefore leaves exactly that completed prefix visible to the VM.
    pub(in super::super) async fn execute(
        mut self,
        context: &RuntimeContext,
        mut prepared: StochasticPreparedSource,
        publish: &mut impl FnMut(StochasticCheckpoint) -> BuiltinResult<()>,
    ) -> BuiltinResult<StochasticSourceExecution> {
        StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
        let mut turn = random::acquire_turn(Some(context), "stochastic_evolution").await?;
        StochasticPreparedSource::checkpoint_receipt(prepared.receipt(), context)?;
        if random::normal_source_algorithm_in_turn(&mut turn)? != self.algorithm {
            // No draw, source publication, transfer, or submission has yet
            // occurred. Dropping selection abandons feedback rather than
            // pretending a provider operation completed.
            return Ok(StochasticSourceExecution::Declined);
        }
        let recorder = self.recorder();
        let (initial_state, pattern, preparation_receipt) = prepared.parts_mut();
        let dtype = match pattern.noise_class() {
            NumericClass::Single => NumericDType::F32,
            NumericClass::Double => NumericDType::F64,
            _ => return Err(source_contract_error()),
        };
        let mut initial = Some(self.initial.take().ok_or_else(source_contract_error)?);
        let mut current = match initial.as_ref().ok_or_else(source_contract_error)? {
            InitialProviderSource::Resident(handle) => Some(handle.clone()),
            InitialProviderSource::HostUpload(_) => None,
        };
        if current.is_some() {
            // The active handle now owns the resident source lease. Do not
            // pin a second copy of the original input through later steps.
            initial = None;
        }
        let mut retained_initial = if current.is_none() {
            Some(RetainedSourceNoise::capture(
                initial_state,
                pattern.shape(),
                pattern.state_class(),
            )?)
        } else {
            None
        };
        // Only the exact shared numeric backing or resident handle remains.
        // The VM's original source binding is not consumed until publication.
        *initial_state = Value::Num(0.0);
        // One operation-owned scope covers every serial upload and compute
        // tile. The canonical recorder then retains one bounded scope record,
        // not an allocation-bearing history proportional to tile count.
        let scope_limit = selected_scope_limit(&self.resources)?;
        let scope = recorder
            .begin_resource_scope(
                ProviderResourcePhasePlan::new([scope_limit]).map_err(upload_plan_error)?,
            )
            .map_err(upload_plan_error)?;
        scope.begin_next_phase().map_err(upload_plan_error)?;
        let result = async {
            for step in 0..pattern.steps() {
                StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
                StochasticPreparedSource::checkpoint_receipt(self.plan.metadata.as_ref(), context)?;
                StochasticPreparedSource::checkpoint_receipt(
                    self.noise_upload.query_receipt.as_ref(),
                    context,
                )?;
                if let Some(receipt) = self.host_receipt.as_ref() {
                    receipt
                        .checkpoint(context)
                        .map_err(|cause| upload_admission_error(context, cause))?;
                }
                publish(StochasticCheckpoint::IterationBinding(f64::from(step + 1)))?;

                // One complete ordered CPU-stream draw per source iteration.
                // The selected tile capacity never changes draw order or
                // advances a separate provider random stream.
                let noise =
                    draw_source_normal(&mut turn, pattern.shape(), dtype, self.mode).await?;
                StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
                let retained_noise =
                    RetainedSourceNoise::capture(&noise, pattern.shape(), pattern.noise_class())?;
                publish(StochasticCheckpoint::Noise(noise))?;

                if current.is_none() {
                    let Some(InitialProviderSource::HostUpload(initial_plan)) = initial.as_ref()
                    else {
                        return Err(source_contract_error());
                    };
                    StochasticPreparedSource::checkpoint_receipt(
                        initial_plan.get().query_receipt.as_ref(),
                        context,
                    )?;
                    let view = retained_initial
                        .as_ref()
                        .ok_or_else(source_contract_error)?
                        .full_view(pattern.shape())?;
                    let uploaded = upload_selected_phase(
                        self.plan.provider,
                        &initial_plan.get().phase,
                        &view,
                        &[],
                        self.metadata.as_deref(),
                        &recorder,
                        &scope,
                    )
                    .await?;
                    current = Some(uploaded);
                    retained_initial = None;
                    initial = None;
                }
                let source = current.as_ref().ok_or_else(source_contract_error)?;
                let total = self
                    .plan
                    .selected
                    .plan
                    .validate()
                    .ok_or_else(source_contract_error)?;
                let mut offset = 0_u64;
                let mut pending: Option<ProviderStochasticEvolutionPending> = None;
                let output = loop {
                    StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
                    let count = self
                        .plan
                        .selected
                        .plan
                        .noise_tile_elements
                        .min(total - offset);
                    let count_usize =
                        usize::try_from(count).map_err(|_| source_contract_error())?;
                    let tile_shape = [count_usize, 1];
                    let tile = retained_noise.tile_view(
                        usize::try_from(offset).map_err(|_| source_contract_error())?,
                        count_usize,
                        &tile_shape,
                    )?;
                    let upload_phase = self
                        .noise_upload
                        .phase_for(count)
                        .ok_or_else(source_contract_error)?;
                    let uploaded_noise = upload_selected_phase(
                        self.plan.provider,
                        upload_phase,
                        &tile,
                        &[source],
                        self.metadata.as_deref(),
                        &recorder,
                        &scope,
                    )
                    .await?;
                    let request = ProviderStochasticEvolutionTileRequest {
                        plan: &self.plan.selected.plan,
                        proposal: &self.plan.selected.proposal,
                        source,
                        noise: &uploaded_noise,
                        step,
                        offset,
                        pending: pending.as_ref(),
                    };
                    let progress = execute_selected_tile(
                        self.plan.provider,
                        request,
                        self.metadata.as_deref(),
                        self.completion_control.clone(),
                        &recorder,
                        &scope,
                    )
                    .await?;
                    if progress.is_complete() {
                        break progress
                            .into_completed_output(
                                self.plan.provider,
                                &self.plan.selected.plan,
                                &self.plan.selected.proposal,
                                step,
                            )
                            .map_err(|cause| {
                                provider_failure("selected stochastic step did not complete", cause)
                            })?;
                    }
                    offset = progress.pending().next_offset();
                    pending = Some(progress.pending().clone());
                };
                StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
                // A handle clone retains the same physical allocation; it
                // does not materialize another numeric payload or VM value.
                let next = output.clone();
                publish(StochasticCheckpoint::State(Value::GpuTensor(output)))?;
                current = Some(next);
            }
            StochasticPreparedSource::checkpoint_receipt(preparation_receipt, context)?;
            if let Some(receipt) = self.host_receipt.as_ref() {
                receipt
                    .checkpoint(context)
                    .map_err(|cause| upload_admission_error(context, cause))?;
            }
            Ok(())
        }
        .await;
        drop(turn);
        match result {
            Ok(()) => {
                if let Err(cause) = scope.finish() {
                    self.finish(Err(cause)).await;
                    return Err(upload_plan_error(cause));
                }
                self.finish(Ok(true)).await;
                Ok(StochasticSourceExecution::Completed)
            }
            Err(error) => {
                let failure = if context.is_cancelled() {
                    ProviderObservationFailure::Cancelled
                } else {
                    ProviderObservationFailure::ExecutionUnproven
                };
                scope.fail(failure);
                let _ = scope.finish();
                self.finish(Err(failure)).await;
                Err(error)
            }
        }
    }
}

fn selected_scope_limit(resources: &StochasticProviderResources) -> BuiltinResult<u64> {
    let mut maximum = resources.phase_limit_bytes;
    for bound in [
        Some(resources.upload_phase_limit_bytes_full),
        resources.upload_phase_limit_bytes_remainder,
        resources.upload_phase_limit_bytes_initial,
    ]
    .into_iter()
    .flatten()
    {
        match bound {
            ResourceByteBound::Known(bytes) if bytes != 0 => maximum = maximum.max(bytes),
            _ => {
                return Err(upload_plan_error(
                    "selected transfer has no physical scope bound",
                ));
            }
        }
    }
    (maximum != 0)
        .then_some(maximum)
        .ok_or_else(|| upload_plan_error("selected compute has no physical scope bound"))
}

/// One exact tile of the immutable provider-selected arithmetic, under the
/// whole-call recorder and a separate operation-owned physical resource
/// scope. A returned prefix is not source publication or feedback completion.
async fn execute_selected_tile(
    provider: &'static dyn AccelProvider,
    request: ProviderStochasticEvolutionTileRequest<'_>,
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
    completion_control: Option<Arc<dyn RetainedHostAllocationClaim>>,
    recorder: &ProviderOperationRecorder,
    scope: &ProviderResourceScope,
) -> BuiltinResult<ProviderStochasticEvolutionProgress> {
    request
        .validate_for(provider)
        .map_err(|cause| provider_failure("selected stochastic tile request changed", cause))?;
    let progress = runmat_accelerate_api::observation::observe_provider_future(
        Some(recorder.clone()),
        observe_provider_resource_future(
            Some(scope.clone()),
            provider.execute_stochastic_evolution_tile(request, metadata, completion_control),
        ),
    )
    .await
    .map_err(|cause| {
        scope.fail(ProviderObservationFailure::ProviderFailed);
        provider_failure("selected stochastic tile failed", cause)
    })?;
    Ok(progress)
}
