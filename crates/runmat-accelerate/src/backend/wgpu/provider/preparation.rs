//! Side-effect-free lookup of the complete audited fusion pipeline inventory.
use super::{
    backend_types::WgpuProvider,
    pipeline_recipe::{FusionPipelineLayout, PipelineRecipe},
};
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::{
    AccelProvider, ProviderElementType, ProviderFeasibilityQuery, ProviderPreparationProgram,
    ProviderPreparationSnapshot, ProviderResidency,
};
use runmat_execution::{CandidatePreparationState, Digest};

impl WgpuProvider {
    pub(super) fn fusion_preparation_snapshot(
        &self,
        query: &ProviderFeasibilityQuery,
        program: ProviderPreparationProgram<'_>,
    ) -> Option<ProviderPreparationSnapshot> {
        let work = crate::placement::fusion_physical_work(self, query)?;
        let ProviderPreparationProgram::Elementwise { shader } = program;
        let mut keys = Vec::new();
        if work.estimate.dispatches != Some(0) {
            let outputs = if query.outputs.len() == 1 {
                FusionPipelineLayout::Single
            } else {
                FusionPipelineLayout::Multiple(query.outputs.len())
            };
            keys.push(
                PipelineRecipe::fusion(
                    shader,
                    query.inputs.len(),
                    outputs,
                    self.workgroup_config.scalar,
                )
                .key(),
            );
            let native = ProviderElementType::from(AccelProvider::precision(self));
            for input in &query.inputs {
                if input.residency == ProviderResidency::Device
                    && input.element_type != native
                    && input.checked_element_count()? != 0
                {
                    keys.push(
                        PipelineRecipe::floating_cast(
                            NumericPrecision::F32,
                            self.precision,
                            self.workgroup_config.scalar,
                        )
                        .key(),
                    );
                }
            }
            for output in &query.outputs {
                if output.element_type != native {
                    keys.push(
                        PipelineRecipe::floating_cast(
                            self.precision,
                            NumericPrecision::F32,
                            self.workgroup_config.scalar,
                        )
                        .key(),
                    );
                }
            }
        }
        // Materialization uses immutable prebuilt permute pipelines; no dynamic
        // cache key is missing here. Repeated casts compile a shared key once.
        keys.sort_unstable();
        keys.dedup();
        let identity = Digest::sha256(
            serde_json::to_vec(&(
                "wgpu-fusion-preparation-v1",
                self.workgroup_config.scalar,
                &keys,
            ))
            .ok()?,
        );
        let cache = self.fused_pipeline_cache.try_lock().ok();
        let (state, cold_pipelines, warm_pipelines) = match cache {
            Some(cache) => {
                let cold =
                    u32::try_from(keys.iter().filter(|key| !cache.contains_key(*key)).count())
                        .ok()?;
                let warm = u32::try_from(keys.len()).ok()?.checked_sub(cold)?;
                (
                    Some(if keys.is_empty() {
                        CandidatePreparationState::Ready
                    } else if cold == 0 {
                        CandidatePreparationState::Warm
                    } else {
                        CandidatePreparationState::Cold
                    }),
                    cold,
                    warm,
                )
            }
            None => (None, 0, 0),
        };
        Some(ProviderPreparationSnapshot {
            identity,
            state,
            cold_pipelines,
            warm_pipelines,
        })
    }
}

#[cfg(test)]
#[path = "preparation/tests.rs"]
mod tests;
