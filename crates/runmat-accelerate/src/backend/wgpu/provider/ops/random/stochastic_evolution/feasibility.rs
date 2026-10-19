use super::super::super::{core, WgpuProvider};
use runmat_accelerate_api::stochastic_evolution::{
    ProviderStochasticEvolutionPhysicalProposal, ProviderStochasticEvolutionPlanRequest,
    ProviderStochasticEvolutionSelection, ProviderStochasticEvolutionWorkRequest,
    PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};
use runmat_accelerate_api::{
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderOperationFamily, ProviderRejectionCode,
    ProviderResourceEstimate,
};

fn observation_host_control_bytes() -> Option<u64> {
    let dispatch =
        crate::backend::wgpu::dispatch::observation::observed_submission_dispatch_host_control_upper_bytes()?;
    #[cfg(not(target_arch = "wasm32"))]
    let dispatch = dispatch
        .checked_add(super::super::super::observation_progress::reservation_host_upper_bytes()?)?;
    Some(dispatch)
}

pub(super) fn physical_width(class: runmat_types::NumericClass) -> Option<usize> {
    match class {
        runmat_types::NumericClass::Single => Some(4),
        runmat_types::NumericClass::Double => Some(8),
        _ => None,
    }
}

fn authentication_workspace_bytes(state_rank: usize) -> Option<u64> {
    let noise_rank = 2_usize;
    // `entry_for_handle` first clones the retained BufferEntry shape and then
    // replaces it with the requested view shape. Source and noise entries stay
    // live together. A continuation adds the pending-output clone/replacement;
    // first-tile publication has the same peak while cloning its output view:
    // source + noise + retained output shape + output view shape.
    let source_peak = state_rank.checked_mul(2)?;
    let noise_peak = state_rank.checked_add(noise_rank.checked_mul(2)?)?;
    let output_peak = state_rank.checked_mul(3)?.checked_add(noise_rank)?;
    let peak_elements = source_peak.max(noise_peak).max(output_peak);
    u64::try_from(peak_elements.checked_mul(std::mem::size_of::<usize>())?).ok()
}

fn supports_stochastic_bindings(limits: &wgpu::Limits) -> bool {
    limits.max_bind_groups >= 1
        && limits.max_bindings_per_bind_group >= 4
        && limits.max_storage_buffers_per_shader_stage >= 3
        && limits.max_uniform_buffers_per_shader_stage >= 1
}

impl WgpuProvider {
    pub(crate) fn stochastic_evolution_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::observation::ProviderObservationCapability> {
        use runmat_accelerate_api::observation::{
            ProviderObservationCapability, ProviderObservationOverhead,
        };

        if query.operation.0 != PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION {
            return None;
        }
        let unsupported = || Some(ProviderObservationCapability::Unsupported);
        let Some(workload) = query.workload.stochastic_evolution else {
            return unsupported();
        };
        if query.family != ProviderOperationFamily::Library
            || query.inputs.len() != 1
            || query.outputs.len() != 1
        {
            return unsupported();
        }
        let request = ProviderStochasticEvolutionPlanRequest {
            state: query.inputs[0].clone(),
            output: query.outputs[0].clone(),
            first_arithmetic: workload.first_arithmetic,
            continuing_arithmetic: workload.continuing_arithmetic,
            steps: workload.steps,
            noise_tile_elements: workload.noise_tile_elements,
        };
        if request.identity() != Some(workload.plan_identity) {
            return unsupported();
        }
        let proposal = self.propose_stochastic_evolution_step_exec(&request)?;
        // Every compute tile has one exactly retired noise upload. A host
        // source adds one initial upload; the semantic query intentionally
        // describes target residency, so quote that conditional upload as an
        // upper bound rather than inferring source residency from it.
        let submissions = proposal
            .full_call_queue_submissions
            .checked_mul(2)?
            .checked_add(1)
            .and_then(|count| u32::try_from(count).ok())?;
        Some(ProviderObservationCapability::Supported(
            ProviderObservationOverhead {
                submissions,
                // Source transfers and tiles await exact retirement before the
                // next submission, so only one observation fence is live.
                requested_peak_buffer_bytes: proposal.observation_fence_bytes_per_tile,
                clear_commands: submissions,
                map_callbacks: submissions,
                wgpu_compute_query:
                    crate::backend::wgpu::dispatch::observation::prospective_encoder_query(1, 1),
            },
        ))
    }

    pub(crate) fn plan_stochastic_evolution_exec(
        &self,
        work: &ProviderStochasticEvolutionWorkRequest,
    ) -> Option<ProviderStochasticEvolutionSelection> {
        let count = work.validate()?;
        let first_pipeline = self
            .pipelines
            .stochastic_evolution
            .get(work.first_arithmetic.class_identity())?;
        let continuing_pipeline = self
            .pipelines
            .stochastic_evolution
            .get(work.continuing_arithmetic.class_identity())?;
        if (work.first_arithmetic.requires_f64() || work.continuing_arithmetic.requires_f64())
            && self.precision != crate::backend::wgpu::types::NumericPrecision::F64
        {
            return None;
        }
        let noise_width =
            u64::try_from(physical_width(work.first_arithmetic.noise_class())?).ok()?;
        let launch_capacity = u64::try_from(
            first_pipeline
                .linear_chunk_capacity()
                .ok()?
                .min(continuing_pipeline.linear_chunk_capacity().ok()?),
        )
        .ok()?;
        let binding_capacity = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .checked_div(noise_width)?;
        let buffer_capacity = self
            .adapter_limits
            .max_buffer_size
            .checked_div(noise_width)?;
        let tile = count
            .min(u64::from(u32::MAX))
            .min(launch_capacity)
            .min(binding_capacity)
            .min(buffer_capacity);
        let plan = work.with_noise_tile(tile)?;
        let proposal = self.propose_stochastic_evolution_step_exec(&plan)?;
        let selection = ProviderStochasticEvolutionSelection { plan, proposal };
        selection.validate_for(work).then_some(selection)
    }

    pub(crate) fn stochastic_evolution_feasibility(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderFeasibility> {
        if query.operation.0 != PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION {
            return None;
        }
        let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
        let Some(workload) = query.workload.stochastic_evolution else {
            return Some(reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "random.stochastic_evolution.missing_workload",
            ));
        };
        if query.family != ProviderOperationFamily::Library
            || query.inputs.len() != 1
            || query.outputs.len() != 1
        {
            return Some(reject(
                ProviderRejectionCode::InvalidShape,
                "random.stochastic_evolution.query",
            ));
        }
        let request = ProviderStochasticEvolutionPlanRequest {
            state: query.inputs[0].clone(),
            output: query.outputs[0].clone(),
            first_arithmetic: workload.first_arithmetic,
            continuing_arithmetic: workload.continuing_arithmetic,
            steps: workload.steps,
            noise_tile_elements: workload.noise_tile_elements,
        };
        if request.identity() != Some(workload.plan_identity) {
            return Some(reject(
                ProviderRejectionCode::InvalidShape,
                "random.stochastic_evolution.identity",
            ));
        }
        let Some(physical) = self.propose_stochastic_evolution_step_exec(&request) else {
            return Some(reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "random.stochastic_evolution.physical",
            ));
        };
        let transient_bytes = physical
            .first_output_buffer_bytes
            .max(physical.continuing_output_buffer_bytes)
            .checked_add(physical.max_noise_tile_buffer_bytes)
            .and_then(|bytes| bytes.checked_add(physical.uniform_buffer_bytes_per_tile))
            .and_then(|bytes| bytes.checked_add(physical.completion_fence_bytes_per_tile))
            .and_then(|bytes| bytes.checked_add(physical.observation_fence_bytes_per_tile));
        let dispatches = u32::try_from(physical.full_call_dispatches).ok();
        Some(match (transient_bytes, dispatches) {
            (Some(transient_bytes), Some(dispatches)) => {
                ProviderFeasibility::supported(ProviderResourceEstimate {
                    transient_bytes: Some(transient_bytes),
                    output_bytes: Some(physical.final_output_buffer_bytes),
                    dispatches: Some(dispatches),
                })
            }
            _ => reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "random.stochastic_evolution.full_call_bound",
            ),
        })
    }

    pub(crate) fn propose_stochastic_evolution_step_exec(
        &self,
        request: &ProviderStochasticEvolutionPlanRequest,
    ) -> Option<ProviderStochasticEvolutionPhysicalProposal> {
        let count = request.validate()?;
        if !supports_stochastic_bindings(&self.adapter_limits) {
            return None;
        }
        let first_pipeline = self
            .pipelines
            .stochastic_evolution
            .get(request.first_arithmetic.class_identity())?;
        let continuing_pipeline = self
            .pipelines
            .stochastic_evolution
            .get(request.continuing_arithmetic.class_identity())?;
        if (request.first_arithmetic.requires_f64() || request.continuing_arithmetic.requires_f64())
            && self.precision != crate::backend::wgpu::types::NumericPrecision::F64
        {
            return None;
        }
        let count_usize = usize::try_from(count).ok()?;
        let tile_usize = usize::try_from(request.noise_tile_elements).ok()?;
        if count > u64::from(u32::MAX)
            || request.noise_tile_elements > u64::from(u32::MAX)
            || tile_usize > first_pipeline.linear_chunk_capacity().ok()?
            || tile_usize > continuing_pipeline.linear_chunk_capacity().ok()?
        {
            return None;
        }
        let first_output_buffer_bytes = count.checked_mul(
            u64::try_from(physical_width(request.first_arithmetic.result_class())?).ok()?,
        )?;
        let continuing_output_buffer_bytes = count.checked_mul(
            u64::try_from(physical_width(
                request.continuing_arithmetic.result_class(),
            )?)
            .ok()?,
        )?;
        let final_output_buffer_bytes = request.output.checked_byte_len()?;
        let max_noise_tile_buffer_bytes = request.noise_tile_elements.checked_mul(
            u64::try_from(physical_width(request.first_arithmetic.noise_class())?).ok()?,
        )?;
        let uniform_buffer_bytes = std::mem::size_of::<
            crate::backend::wgpu::params::StochasticEvolutionStepParamsF64F64,
        >() as u64;
        let completion_fence_bytes = 8_u64;
        let observation_fence_bytes =
            crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES;
        if [
            first_output_buffer_bytes,
            continuing_output_buffer_bytes,
            max_noise_tile_buffer_bytes,
        ]
        .into_iter()
        .any(|bytes| {
            bytes > self.adapter_limits.max_buffer_size
                || bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        }) || uniform_buffer_bytes > self.adapter_limits.max_buffer_size
            || uniform_buffer_bytes > u64::from(self.adapter_limits.max_uniform_buffer_binding_size)
            || completion_fence_bytes > self.adapter_limits.max_buffer_size
            || observation_fence_bytes > self.adapter_limits.max_buffer_size
        {
            return None;
        }
        let tiles = u32::try_from(count_usize.div_ceil(tile_usize)).ok()?;
        let full = u64::from(tiles).checked_mul(u64::from(request.steps))?;
        let output_view_host_bytes = runmat_accelerate_api::GpuTensorHandle::view_host_upper_bytes(
            request.output.shape.len(),
        )?;
        let authentication_host_bytes_per_tile =
            authentication_workspace_bytes(request.state.shape.len())?;
        Some(ProviderStochasticEvolutionPhysicalProposal {
            plan_identity: request.identity()?,
            first_output_buffer_bytes,
            continuing_output_buffer_bytes,
            final_output_buffer_bytes,
            max_noise_tile_buffer_bytes,
            uniform_buffer_bytes_per_tile: uniform_buffer_bytes,
            completion_fence_bytes_per_tile: completion_fence_bytes,
            observation_fence_bytes_per_tile: observation_fence_bytes,
            completion_control_host_bytes: core::map_completion_host_upper_bytes()?,
            authentication_host_bytes_per_tile,
            submission_retention_host_bytes_per_tile:
                super::submission::StochasticTileSubmittedAllocations::host_upper_bytes()?,
            source_transfer_retirement_host_bytes:
                runmat_accelerate_api::observation::stochastic_evolution_source_transfer_retirement_host_upper_bytes()?,
            observation_host_control_bytes_per_tile: observation_host_control_bytes()?,
            output_view_host_bytes,
            tiles_per_step: tiles,
            dispatches_per_step: tiles,
            queue_submissions_per_step: tiles,
            full_call_dispatches: full,
            full_call_queue_submissions: full,
            peak_provisional_output_buffers: 1,
            frozen_workgroup_size: first_pipeline.linear_launch()?.workgroup(),
            frozen_tile_capacity: u64::try_from(
                first_pipeline
                    .linear_chunk_capacity()
                    .ok()?
                    .min(continuing_pipeline.linear_chunk_capacity().ok()?),
            )
            .ok()?,
        })
    }
}
