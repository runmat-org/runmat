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
        let shader = match program {
            ProviderPreparationProgram::Elementwise { shader } => shader,
            ProviderPreparationProgram::FloatingCast { target } => {
                let work = crate::placement::floating_cast_physical_work(self, query)?;
                if work.target != target {
                    return None;
                }
                let keys = if work.estimate.dispatches == Some(0) {
                    vec![]
                } else {
                    let precision = |precision| match precision {
                        runmat_accelerate_api::ProviderPrecision::F32 => NumericPrecision::F32,
                        runmat_accelerate_api::ProviderPrecision::F64 => NumericPrecision::F64,
                    };
                    vec![PipelineRecipe::floating_cast(
                        precision(work.source),
                        precision(work.target),
                        work.workgroup,
                    )
                    .key()]
                };
                return self.preparation_keys_snapshot("wgpu-floating-cast-preparation-v1", keys);
            }
        };
        let work = crate::placement::fusion_physical_work(self, query)?;
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
        self.preparation_keys_snapshot("wgpu-fusion-preparation-v1", keys)
    }

    fn preparation_keys_snapshot(
        &self,
        kind: &str,
        keys: Vec<u64>,
    ) -> Option<ProviderPreparationSnapshot> {
        let identity =
            Digest::sha256(serde_json::to_vec(&(kind, self.workgroup_config.scalar, &keys)).ok()?);
        // No shader is used by a same-type alias or empty cast. Cache contention
        // cannot make that route's known zero preparation work unknown.
        if keys.is_empty() {
            return Some(ProviderPreparationSnapshot {
                identity,
                state: Some(CandidatePreparationState::Ready),
                cold_pipelines: 0,
                warm_pipelines: 0,
            });
        }
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
