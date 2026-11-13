//! Cache/progress facts for the same actual paired POWER physical recipe.
use super::*;
use runmat_accelerate_api::{
    observation::{ProviderObservationCapability as Capability, ProviderObservationOverhead},
    power_service::{PowerPairCompound, PowerTypedScalarProjection},
};

impl WgpuProvider {
    pub(super) fn power_scalar_preparation_snapshot(
        &self,
        projection: &PowerTypedScalarProjection,
    ) -> Option<ProviderPreparationSnapshot> {
        if !projection.is_current_for(self) {
            return None;
        }
        let physical = projection.pair.physical();
        let mut keys = vec![physical.probe.flag_pipeline_key];
        keys.push(physical.real.work.pipeline_key?);
        keys.push(physical.complex.work.pipeline_key?);
        if projection.widened.is_some() {
            keys.push(
                PipelineRecipe::floating_cast(
                    NumericPrecision::F32,
                    NumericPrecision::F64,
                    self.workgroup_config.scalar,
                )
                .key(),
            );
        }
        if projection.narrow_real.is_some() || projection.narrow_complex.is_some() {
            keys.push(
                PipelineRecipe::floating_cast(
                    NumericPrecision::F64,
                    NumericPrecision::F32,
                    self.workgroup_config.scalar,
                )
                .key(),
            );
        }
        keys.sort_unstable();
        keys.dedup();
        let mut snapshot =
            self.preparation_keys_snapshot("wgpu-power-scalar-preparation-v1", keys, None)?;
        let any_warm = self.pipelines.reduce_dim_sum_mean.is_initialized();
        snapshot.identity =
            Digest::sha256_serialized(&(projection.identity().ok()?, snapshot.identity, any_warm))
                .ok()?;
        if any_warm {
            snapshot.warm_pipelines = snapshot.warm_pipelines.checked_add(1)?;
        } else {
            snapshot.cold_pipelines = snapshot.cold_pipelines.checked_add(1)?;
        }
        snapshot.state = snapshot.state.map(|_| {
            if snapshot.cold_pipelines == 0 {
                CandidatePreparationState::Warm
            } else {
                CandidatePreparationState::Cold
            }
        });
        Some(snapshot)
    }
    fn current_power_pair_recipe(&self, recipe: &PowerPairCompound) -> Option<()> {
        if !recipe.is_owned_by(self) {
            return None;
        }
        let [base, exponent] = recipe.probe().sources();
        if self
            .propose_power_probe(base, exponent, Some(recipe.probe().physical()))
            .ok()?
            .as_ref()
            != Some(recipe.probe().physical())
        {
            return None;
        }
        for evaluation in [recipe.real(), recipe.complex()] {
            if self
                .propose_power_evaluation(base, exponent, evaluation.domain())
                .ok()?
                .as_ref()
                != Some(evaluation.physical())
            {
                return None;
            }
        }
        Some(())
    }

    pub(super) fn power_pair_preparation_snapshot(
        &self,
        recipe: &PowerPairCompound,
    ) -> Option<ProviderPreparationSnapshot> {
        self.current_power_pair_recipe(recipe)?;
        let mut keys = vec![recipe.probe().physical().flag_pipeline_key];
        for evaluation in [recipe.real(), recipe.complex()] {
            keys.push(evaluation.physical().work.pipeline_key?);
        }
        keys.sort_unstable();
        keys.dedup();
        let mut snapshot =
            self.preparation_keys_snapshot("wgpu-power-pair-preparation-v1", keys, None)?;
        // Any's constructor-frozen PipelineBundle is lazy: inspect its actual
        // OnceLock rather than assuming it is warm or querying another cache.
        let any_warm = self.pipelines.reduce_dim_sum_mean.is_initialized();
        snapshot.identity =
            Digest::sha256_serialized(&(recipe.identity().ok()?, snapshot.identity, any_warm))
                .ok()?;
        if any_warm {
            snapshot.warm_pipelines = snapshot.warm_pipelines.checked_add(1)?;
        } else {
            snapshot.cold_pipelines = snapshot.cold_pipelines.checked_add(1)?;
        }
        snapshot.state = snapshot.state.map(|_| {
            if snapshot.cold_pipelines == 0 {
                CandidatePreparationState::Warm
            } else {
                CandidatePreparationState::Cold
            }
        });
        Some(snapshot)
    }

    pub(crate) fn power_pair_observation_capability_exec(
        &self,
        recipe: &PowerPairCompound,
    ) -> Capability {
        #[cfg(not(target_arch = "wasm32"))]
        if !self
            .observation_progress
            .as_ref()
            .is_some_and(|port| port.is_available())
        {
            return Capability::Unsupported;
        }
        if self.current_power_pair_recipe(recipe).is_none() {
            return Capability::Unsupported;
        }
        let Some(submissions) = recipe.probe().physical().queue_submissions.checked_add(
            recipe
                .real()
                .physical()
                .work
                .queue_submissions
                .max(recipe.complex().physical().work.queue_submissions),
        ) else {
            return Capability::Unsupported;
        };
        let Some(requested_peak_buffer_bytes) = u64::from(submissions)
            .checked_mul(crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES)
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
}
