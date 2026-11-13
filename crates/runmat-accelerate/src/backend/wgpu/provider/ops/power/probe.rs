//! Constructor-frozen ordinary POWER probe, shared by legacy classification
//! and selected compound execution. Preparation validates physical sources and
//! freezes actual flag/reduction geometry; it submits no provider work.
use super::super::reduction::{FrozenAnyPipeline, FrozenAnySnapshot};
use super::*;
use runmat_accelerate_api::allocation::TensorViewIdentity;

pub(super) struct PreparedPowerDomainProbe {
    source_views: [TensorViewIdentity; 2],
    shape: Vec<usize>,
    len: usize,
    input_precision: NumericPrecision,
    snapshot: FrozenAnySnapshot,
    any_pipeline: FrozenAnyPipeline,
    probe: probe_emission::ProbePreflight,
    prepared_flag: flag_fusion::PreparedPowerFlagFusion,
}

impl WgpuProvider {
    /// None is only the already-validated empty output, not an unsupported
    /// physical route. No numerical-domain scan or device submission occurs.
    pub(super) fn prepare_power_domain_probe_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
    ) -> Result<Option<PreparedPowerDomainProbe>> {
        self.prepare_power_domain_probe_with_snapshot_exec(
            base,
            exponent,
            FrozenAnySnapshot::capture(),
        )
    }

    fn prepare_power_domain_probe_with_snapshot_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        snapshot: FrozenAnySnapshot,
    ) -> Result<Option<PreparedPowerDomainProbe>> {
        let input_precision = self.get_entry_raw(base)?.precision;
        if input_precision != self.precision
            && !(input_precision == NumericPrecision::F32
                && self.precision == NumericPrecision::F64)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "power classification requires exact floating input conversion",
            ));
        }
        let (shape, len) = self.admitted_power_shape(base, exponent, input_precision)?;
        if len == 0 {
            return Ok(None);
        }
        self.require_ieee_words(self.precision)?;
        let any_pipeline = self.prepare_power_any_pipeline_exec()?;
        let probe = probe_emission::ProbePreflight::plan(probe_emission::ProbePlanRequest {
            base,
            exponent,
            owner_device_id: self.device_id(),
            len,
            precision: self.precision,
            snapshot,
            limits: &self.adapter_limits,
            shader_workgroup: any_pipeline.shader_workgroup(),
        })?;
        let prepared_flag =
            self.prepare_power_flag_fusion_exec(base, exponent, &shape, len, input_precision)?;
        Ok(Some(PreparedPowerDomainProbe {
            source_views: [base.view_identity(), exponent.view_identity()],
            shape,
            len,
            input_precision,
            snapshot,
            any_pipeline,
            probe,
            prepared_flag,
        }))
    }

    pub(crate) fn propose_power_probe_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        frozen: Option<&runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        Option<runmat_accelerate_api::power_service::PowerProbePhysicalProposal>,
    > {
        let snapshot = frozen.map_or_else(FrozenAnySnapshot::capture, |frozen| FrozenAnySnapshot {
            enabled: frozen.snapshot_copy,
        });
        self.prepare_power_domain_probe_with_snapshot_exec(base, exponent, snapshot)?
            .map(|prepared| prepared.physical())
            .transpose()
            .map_err(Into::into)
    }

    pub(crate) async fn execute_power_probe_recipe_exec(
        &self,
        recipe: &runmat_accelerate_api::power_service::PowerProbeRecipe,
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_accelerate_api::power_service::PowerProbeObserved,
    > {
        use runmat_accelerate_api::{power_service::PowerProbeObserved, ProviderExecutionError};
        if !recipe.is_owned_by(self) {
            return Err(anyhow!("POWER probe owner changed").into());
        }
        let [base, exponent] = recipe.sources();
        let prepared = self
            .prepare_power_domain_probe_with_snapshot_exec(
                base,
                exponent,
                FrozenAnySnapshot {
                    enabled: recipe.physical().snapshot_copy,
                },
            )?
            .ok_or_else(|| anyhow!("selected POWER probe became empty"))?;
        if prepared.physical()? != *recipe.physical() {
            return Err(anyhow!("selected POWER probe recipe changed").into());
        }
        let (domain, _) = self
            .execute_power_domain_probe_prepared_exec(base, exponent, prepared)
            .await
            .map_err(|cause| ProviderExecutionError::from(cause).with_unproven_effects())?;
        // The consuming runner checked every flag, Any-stage, reshape alias,
        // readback extent and submission observation against this frozen plan.
        Ok(PowerProbeObserved {
            domain,
            dispatches: recipe.physical().dispatches,
            queue_submissions: recipe.physical().queue_submissions,
        })
    }

    pub(super) async fn execute_power_domain_probe_prepared_exec(
        &self,
        base: &GpuTensorHandle,
        exponent: &GpuTensorHandle,
        prepared: PreparedPowerDomainProbe,
    ) -> Result<(PowerResultDomain, ProbeRecordResult)> {
        let PreparedPowerDomainProbe {
            source_views,
            shape,
            len,
            input_precision,
            snapshot,
            any_pipeline,
            probe,
            prepared_flag,
        } = prepared;
        let (current_shape, current_len) =
            self.admitted_power_shape(base, exponent, input_precision)?;
        ensure!(
            source_views == [base.view_identity(), exponent.view_identity()]
                && current_shape == shape
                && current_len == len,
            "prepared POWER probe source or geometry changed"
        );
        probe.require_sources(base, exponent)?;
        let phase_limits = probe.resource_phase_limits(&prepared_flag.resource_work())?;
        let resource_scope = runmat_accelerate_api::observation::current_provider_observation()
            .map(|recorder| recorder.begin_resource_scope(phase_limits))
            .transpose()?;
        let classified = runmat_accelerate_api::observation::observe_provider_resource_future(
            resource_scope.clone(),
            async {
                if let Some(scope) =
                    runmat_accelerate_api::observation::current_provider_resource_scope()
                {
                    PowerProbeResourcePhase::Flag.begin(&scope)?;
                }
                // Input buffers remain borrowed, including F32 on an F64 provider.
                // Exact widening occurs in registers. Flags and the existing scalar
                // reductions retain native precision, so probe allocations/readback
                // geometry is unchanged: no input staging or hidden layout dispatch.
                let (flags, flag_receipt) =
                    self.execute_power_flag_fusion_prepared_exec(base, exponent, prepared_flag)?;
                flag_receipt.require_output(&flags)?;
                probe.require_flag(&flags)?;
                // The flag receipt covers emission only. The outer canonical
                // operation receipt, if issued, owns fence completion and the
                // explicit resource scope across the remaining stages.
                // This buffer is owned here, so reshaping it cannot change an input's
                // metadata. Existing two-dimensional full reduction handles all ranks.
                let flat = self.reshape_exec(&flags, &[len, 1])?;
                probe.require_flat(&flags, &flat)?;
                if let Some(scope) =
                    runmat_accelerate_api::observation::current_provider_resource_scope()
                {
                    PowerProbeResourcePhase::AnyRows.begin(&scope)?;
                }
                let (reduced, stages) =
                    self.reduce_power_any_omit_observed_exec(&flat, snapshot, &any_pipeline)?;
                probe.require_stages(&flat, &reduced, &stages)?;
                if let Some(scope) =
                    runmat_accelerate_api::observation::current_provider_resource_scope()
                {
                    PowerProbeResourcePhase::Readback.begin(&scope)?;
                }
                let (result, readback) = self.download_power_probe_observed_exec(&reduced).await?;
                probe.require_readback(&reduced, &readback)?;
                #[cfg(test)]
                let submissions =
                    probe.collect_submissions(flag_receipt.tickets(), &stages, &readback)?;
                ensure!(
                    result.data.len() == 1 && (result.data[0] == 0.0 || result.data[0] == 1.0),
                    "power domain reduction returned an invalid flag"
                );
                let domain = if result.data[0] == 0.0 {
                    PowerResultDomain::Real
                } else {
                    PowerResultDomain::Complex
                };
                #[cfg(test)]
                let record = Some(submissions);
                #[cfg(not(test))]
                let record = ();
                Ok((domain, record))
            },
        )
        .await;
        if let Some(scope) = resource_scope {
            if classified.is_ok() {
                scope.finish()?;
            } else {
                scope.fail(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                );
            }
        }
        classified
    }
}

impl PreparedPowerDomainProbe {
    fn physical(&self) -> Result<runmat_accelerate_api::power_service::PowerProbePhysicalProposal> {
        let flag = self.prepared_flag.resource_work();
        physical_proposal(
            &self.shape,
            self.snapshot,
            &flag,
            self.prepared_flag.pipeline_key(),
            self.prepared_flag.physical_identity()?,
            &self.any_pipeline,
            self.probe.data_resource_limits(&flag)?,
        )
    }
}

impl WgpuProvider {
    /// Handle-free geometry for an admitted allocator projection. The future
    /// source is authenticated separately before any child can execute.
    pub(super) fn project_power_probe_geometry(
        &self,
        input_precision: NumericPrecision,
        shape: &[usize],
        len: usize,
        snapshot: FrozenAnySnapshot,
    ) -> Result<runmat_accelerate_api::power_service::PowerProbePhysicalProposal> {
        self.require_ieee_words(self.precision)?;
        let any = self.prepare_power_any_pipeline_exec()?;
        let flag = flag_fusion::PowerFlagFusionRecipe::plan(
            input_precision,
            self.precision,
            shape,
            self.linear_launch,
            &self.adapter_limits,
            self.enabled_features,
        )?;
        let geometry = probe_emission::ProbeGeometry::plan(
            len,
            self.precision,
            snapshot,
            &self.adapter_limits,
            any.shader_workgroup(),
        )?;
        let work = flag.resource_work();
        physical_proposal(
            shape,
            snapshot,
            &work,
            flag.pipeline_key(),
            flag.physical_identity()?,
            &any,
            geometry.resource_limits_with_fences(&work, false)?,
        )
    }
}

fn physical_proposal(
    shape: &[usize],
    snapshot: FrozenAnySnapshot,
    flag: &runmat_accelerate_api::ProviderRealElementwiseObserved,
    flag_pipeline_key: u64,
    flag_identity: runmat_execution::Digest,
    any: &FrozenAnyPipeline,
    phase_data_limits: [u64; 4],
) -> Result<runmat_accelerate_api::power_service::PowerProbePhysicalProposal> {
    use runmat_accelerate_api::power_service::PowerProbePhysicalProposal;
    let reduction_queues = if snapshot.enabled { 4_u32 } else { 2_u32 };
    Ok(PowerProbePhysicalProposal {
        output_shape: shape.to_vec(),
        snapshot_copy: snapshot.enabled,
        flag_pipeline_key,
        shader_identity: runmat_execution::Digest::sha256_serialized(&(
            "wgpu-power-probe-shaders-v1",
            flag_identity,
            any.physical_identity()?,
        ))?,
        phase_data_limits,
        dispatches: flag
            .dispatches
            .checked_add(2)
            .ok_or_else(|| anyhow!("POWER probe dispatch count overflow"))?,
        queue_submissions: flag
            .queue_submissions
            .checked_add(reduction_queues)
            .and_then(|queues| queues.checked_add(1))
            .ok_or_else(|| anyhow!("POWER probe submission count overflow"))?,
        host_workspace_bytes: super::super::io::power_probe_host_workspace_bytes()?,
    })
}
