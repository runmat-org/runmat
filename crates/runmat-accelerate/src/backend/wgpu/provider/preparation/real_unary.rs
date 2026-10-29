//! Complete compound inventory at the actual pipeline/transfer/progress owner.
use super::*;
use runmat_accelerate_api::observation::{
    ProviderObservationCapability as Capability, ProviderObservationOverhead,
    ProviderObservationSupport, ProviderOperationReceipt,
};
use runmat_accelerate_api::{
    ProviderRealUnaryCompound, ProviderRealUnaryContinuations, ProviderRealUnaryPublication,
    ProviderStorage,
};

impl WgpuProvider {
    fn current_real_unary_recipe(&self, recipe: &ProviderRealUnaryCompound) -> Option<()> {
        if !recipe.is_owned_by(self)
            || recipe.source().allocation_lease().validate_live().is_err()
            || recipe
                .source()
                .view()
                .validate(recipe.source().physical())
                .is_err()
            || runmat_accelerate_api::provider_for_handle(recipe.source())
                .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, self))
            || self
                .propose_real_elementwise(recipe.real().request())
                .ok()?
                .as_ref()
                != Some(recipe.real().physical())
        {
            return None;
        }
        for probe in recipe.probes() {
            if self.propose_global_extrema(probe.request()).ok()?.as_ref() != Some(probe.physical())
            {
                return None;
            }
        }
        Some(())
    }

    pub(super) fn real_unary_preparation_snapshot(
        &self,
        recipe: &ProviderRealUnaryCompound,
    ) -> Option<ProviderPreparationSnapshot> {
        self.current_real_unary_recipe(recipe)?;
        let mut keys = recipe
            .probes()
            .filter(|probe| probe.physical().dispatches != 0)
            .map(|probe| probe.physical().pipeline_key)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        // Both min/max use this same selected validated global tree.
        let validated = self.preparation_keys_snapshot(
            "wgpu-unary-domain-probe-preparation-v1",
            keys.clone(),
            keys.first().copied(),
        )?;
        let real = self.preparation_keys_snapshot(
            "wgpu-unary-real-continuation-preparation-v1",
            recipe.real().physical().pipeline_key.as_slice(),
            None,
        )?;
        let cold_pipelines = validated.cold_pipelines.checked_add(real.cold_pipelines)?;
        let warm_pipelines = validated.warm_pipelines.checked_add(real.warm_pipelines)?;
        Some(ProviderPreparationSnapshot {
            identity: Digest::sha256_serialized(&(
                recipe.identity().ok()?,
                validated.identity,
                real.identity,
            ))
            .ok()?,
            state: validated.state.zip(real.state).map(|_| {
                if cold_pipelines == 0 {
                    CandidatePreparationState::Warm
                } else {
                    CandidatePreparationState::Cold
                }
            }),
            cold_pipelines,
            warm_pipelines,
        })
    }

    pub(crate) fn real_unary_observation_capability_exec(
        &self,
        recipe: &ProviderRealUnaryCompound,
        continuations: ProviderRealUnaryContinuations,
    ) -> Capability {
        #[cfg(not(target_arch = "wasm32"))]
        if !self
            .observation_progress
            .as_ref()
            .is_some_and(|port| port.is_available())
        {
            return Capability::Unsupported;
        }
        let inventory = || -> Option<u32> {
            self.current_real_unary_recipe(recipe)?;
            let mut probes = 0_u32;
            for probe in recipe.probes() {
                let readback = self.produced_numeric_readback_evidence(&probe.output())?;
                if readback.source_requested_bytes != probe.physical().output_buffer_bytes {
                    return None;
                }
                probes = probes
                    .checked_add(probe.physical().queue_submissions)?
                    .checked_add(readback.queue_slots)?;
            }
            let mut restore = 0;
            for (domain, publication) in [
                (runmat_types::NumericDomain::Real, continuations.real),
                (runmat_types::NumericDomain::Complex, continuations.complex),
            ] {
                if publication == ProviderRealUnaryPublication::Host {
                    continue;
                }
                let mut output = recipe.real().output().clone();
                if domain == runmat_types::NumericDomain::Complex {
                    output.element_type = match output.element_type {
                        ProviderElementType::F32 => ProviderElementType::ComplexF32,
                        ProviderElementType::F64 => ProviderElementType::ComplexF64,
                        _ => return None,
                    };
                    output.storage = ProviderStorage::DenseComplexInterleaved;
                }
                let query = runmat_accelerate_api::class_preserving_upload_query(vec![output])?;
                restore = restore.max(
                    self.upload_physical_evidence(&query)?
                        .completion_queue_slots,
                );
            }
            let host = self
                .download_numeric_queue_slots(recipe.source())
                .checked_add(restore)?;
            probes.checked_add(recipe.real().physical().queue_submissions.max(host))
        };
        let Some(submissions) = inventory() else {
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

    pub(in crate::backend::wgpu::provider) fn owned_operation_observation(
        &self,
    ) -> ProviderObservationSupport {
        #[cfg(not(target_arch = "wasm32"))]
        let receipt = match &self.observation_progress {
            Some(progress) => ProviderOperationReceipt::with_wgpu_progress(
                self.observation_budget.clone(),
                progress.clone(),
            ),
            None => return ProviderObservationSupport::Unsupported,
        };
        #[cfg(target_arch = "wasm32")]
        let receipt = ProviderOperationReceipt::with_budget(self.observation_budget.clone());
        ProviderObservationSupport::Observed(receipt)
    }
}
