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
#[path = "preparation/power_pair.rs"]
mod power_pair;
#[path = "preparation/real_unary.rs"]
mod real_unary;

fn preparation_key_is_warm(
    key: u64,
    validated_fusion_key: Option<u64>,
    legacy_has_key: bool,
    validated_has_key: bool,
) -> bool {
    if validated_fusion_key == Some(key) {
        validated_has_key
    } else {
        legacy_has_key
    }
}

impl WgpuProvider {
    pub(super) fn fusion_preparation_snapshot(
        &self,
        query: &ProviderFeasibilityQuery,
        program: ProviderPreparationProgram<'_>,
    ) -> Option<ProviderPreparationSnapshot> {
        let shader = match program {
            ProviderPreparationProgram::PowerPair { recipe } => {
                return self.power_pair_preparation_snapshot(recipe)
            }
            ProviderPreparationProgram::PowerScalar { projection } => {
                return self.power_scalar_preparation_snapshot(projection)
            }
            ProviderPreparationProgram::RealUnary { recipe } => {
                return self.real_unary_preparation_snapshot(recipe)
            }
            ProviderPreparationProgram::TypedReduction {
                selection,
                host_metadata,
            } => {
                let work = query.workload.typed_reduction.as_ref()?;
                if !work.matches_selection(selection)
                    || crate::placement::typed_reduction_physical_work(query).is_none()
                    || self
                        .admit_reduction_selection(selection, host_metadata)
                        .is_err()
                {
                    return None;
                }
                let keys = self.typed_reduction_pipeline_keys(selection)?;
                return self.preparation_keys_snapshot(
                    "wgpu-typed-reduction-preparation-v1",
                    keys,
                    None,
                );
            }
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
                // Floating-cast execution uses the validated recipe cache,
                // never the legacy fusion cache, even for an ordinary cast.
                let validated_cast_key = keys.first().copied();
                return self.preparation_keys_snapshot(
                    "wgpu-floating-cast-preparation-v1",
                    keys,
                    validated_cast_key,
                );
            }
            ProviderPreparationProgram::Sum => {
                return self.sum_preparation_snapshot(query);
            }
        };
        let work = crate::placement::fusion_physical_work(self, query)?;
        let mut keys = Vec::new();
        let mut validated_fusion_key = None;
        if work.estimate.dispatches != Some(0) {
            let outputs = if query.outputs.len() == 1 {
                FusionPipelineLayout::Single
            } else {
                FusionPipelineLayout::Multiple(query.outputs.len())
            };
            let fusion_key = PipelineRecipe::fusion(
                shader,
                query.inputs.len(),
                outputs,
                self.workgroup_config.scalar,
            )
            .key();
            keys.push(fusion_key);
            validated_fusion_key = Some(fusion_key);
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
        self.preparation_keys_snapshot("wgpu-fusion-preparation-v1", keys, validated_fusion_key)
    }

    fn sum_preparation_snapshot(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderPreparationSnapshot> {
        let work = crate::placement::sum_physical_work(self, query)?;
        let prebuilt = match work.kernel {
            crate::placement::SumKernelRoute::Host => return None,
            crate::placement::SumKernelRoute::Global => {
                Some(self.pipelines.reduce_global.is_initialized())
            }
            crate::placement::SumKernelRoute::Dimension2 => {
                Some(self.pipelines.reduce_dim_sum_mean.is_initialized())
            }
            crate::placement::SumKernelRoute::DimensionN => None,
        };
        let mut keys = Vec::new();
        let mut validated_keys = Vec::new();
        if work.kernel == crate::placement::SumKernelRoute::DimensionN
            && work.estimate.dispatches != Some(0)
        {
            let key = self.nd_sum_pipeline_key();
            keys.push(key);
            validated_keys.push(key);
        }
        if work.input_cast {
            let key = PipelineRecipe::floating_cast(
                NumericPrecision::F32,
                NumericPrecision::F64,
                self.workgroup_config.scalar,
            )
            .key();
            keys.push(key);
            validated_keys.push(key);
        }
        if work.integer_input_cast {
            let key = PipelineRecipe::integer_sum_cast(self.workgroup_config.scalar).key();
            keys.push(key);
            validated_keys.push(key);
        }
        if work.nan_map {
            keys.push(
                PipelineRecipe::fusion(
                    &crate::backend::wgpu::shaders::nan::map_source(NumericPrecision::F64),
                    1,
                    FusionPipelineLayout::Single,
                    self.workgroup_config.scalar,
                )
                .key(),
            );
        }
        if work.output_cast {
            let key = PipelineRecipe::floating_cast(
                NumericPrecision::F64,
                NumericPrecision::F32,
                self.workgroup_config.scalar,
            )
            .key();
            keys.push(key);
            validated_keys.push(key);
        }
        keys.sort_unstable();
        keys.dedup();
        let identity = Digest::sha256(
            serde_json::to_vec(&(
                "wgpu-sum-admitted-preparation-v3",
                query.workload.sum.as_ref()?,
                work.kernel,
                self.device_id_exec(),
                &keys,
            ))
            .ok()?,
        );
        let cache = self.fused_pipeline_cache.try_lock().ok();
        let validated = self.validated_fusion_cache.try_lock().ok();
        if !keys.is_empty()
            && (cache.is_none() || (!validated_keys.is_empty() && validated.is_none()))
        {
            return Some(ProviderPreparationSnapshot {
                identity,
                state: None,
                cold_pipelines: 0,
                warm_pipelines: 0,
            });
        }
        let dynamic_cold = u32::try_from(
            keys.iter()
                .filter(|key| {
                    if validated_keys.contains(key) {
                        !validated
                            .as_ref()
                            .is_some_and(|cache| cache.contains_key(key))
                    } else {
                        !cache.as_ref().is_some_and(|cache| cache.contains_key(key))
                    }
                })
                .count(),
        )
        .ok()?;
        let dynamic_warm = u32::try_from(keys.len()).ok()?.checked_sub(dynamic_cold)?;
        let cold_pipelines = dynamic_cold.checked_add(u32::from(prebuilt == Some(false)))?;
        let warm_pipelines = dynamic_warm.checked_add(u32::from(prebuilt == Some(true)))?;
        Some(ProviderPreparationSnapshot {
            identity,
            state: Some(if cold_pipelines == 0 {
                CandidatePreparationState::Warm
            } else {
                CandidatePreparationState::Cold
            }),
            cold_pipelines,
            warm_pipelines,
        })
    }

    fn preparation_keys_snapshot(
        &self,
        kind: &str,
        keys: impl AsRef<[u64]>,
        validated_fusion_key: Option<u64>,
    ) -> Option<ProviderPreparationSnapshot> {
        let keys = keys.as_ref();
        let identity =
            Digest::sha256_serialized(&(kind, self.workgroup_config.scalar, keys)).ok()?;
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
        let validated =
            validated_fusion_key.and_then(|_| self.validated_fusion_cache.try_lock().ok());
        let (state, cold_pipelines, warm_pipelines) =
            match (cache, validated_fusion_key.is_none() || validated.is_some()) {
                (Some(cache), true) => {
                    let cold = u32::try_from(
                        keys.iter()
                            .filter(|key| {
                                !preparation_key_is_warm(
                                    **key,
                                    validated_fusion_key,
                                    cache.contains_key(key),
                                    validated
                                        .as_ref()
                                        .is_some_and(|cache| cache.contains_key(key)),
                                )
                            })
                            .count(),
                    )
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
                _ => (None, 0, 0),
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
