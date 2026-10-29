//! Read-only preparation facts for an exact admitted network forward route.

use super::{
    backend_types::WgpuProvider,
    linalg::{selected_matmul_pipeline, MatmulPipelineRoute},
    pipeline_cache::PipelineCache,
    pipeline_recipe::{FusionPipelineLayout, PipelineRecipe},
};
use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::{
    DeepLearningFeedForwardStage, DeepLearningForwardCapability, DeepLearningForwardOperation,
    DeepLearningForwardParameters, DeepLearningForwardRoute, GpuTensorLayout,
    ProviderPreparationSnapshot, ProviderViewChain, ProviderViewSource, ProviderViewStep,
};
use runmat_execution::{CandidatePreparationState, Digest};

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum DynamicPipeline {
    Legacy(u64),
    Validated(u64),
}

impl DynamicPipeline {
    fn is_warm<L, V>(
        self,
        legacy: Option<&PipelineCache<L>>,
        validated: Option<&std::collections::HashMap<u64, V>>,
    ) -> Option<bool> {
        match self {
            Self::Legacy(key) => legacy.map(|cache| cache.contains_key(&key)),
            Self::Validated(key) => validated.map(|cache| cache.contains_key(&key)),
        }
    }
}

#[cfg(test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dynamic_pipeline_warmth_uses_its_typed_cache_owner() {
    let mut legacy = PipelineCache::default();
    legacy
        .insert_unbounded(7, &super::pipeline_cache::PipelineOwner::unbounded(()))
        .unwrap();
    let validated = std::collections::HashMap::from([(8u64, ())]);
    assert_eq!(
        DynamicPipeline::Legacy(7)
            .is_warm(Some(&legacy), None::<&std::collections::HashMap<u64, ()>>),
        Some(true)
    );
    assert_eq!(
        DynamicPipeline::Validated(7).is_warm(None::<&PipelineCache<()>>, Some(&validated)),
        Some(false)
    );
    assert_eq!(
        DynamicPipeline::Validated(8).is_warm(None::<&PipelineCache<()>>, Some(&validated)),
        Some(true)
    );
    assert_eq!(
        DynamicPipeline::Legacy(8)
            .is_warm(Some(&legacy), None::<&std::collections::HashMap<u64, ()>>),
        Some(false)
    );
    assert_eq!(
        DynamicPipeline::Legacy(7).is_warm(None::<&PipelineCache<()>>, Some(&validated)),
        None
    );
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum FixedPipeline {
    Fill,
    ScalarMax,
    Binary,
    BinaryBroadcast,
    Matmul,
    MatmulVec4,
    Permute,
    LayerNormStatistics,
    LayerNormAffine,
    StablePool1D,
    Conv1DForward,
}

impl FixedPipeline {
    const fn tag(self) -> u8 {
        match self {
            Self::Fill => 0,
            Self::ScalarMax => 1,
            Self::Binary => 2,
            Self::BinaryBroadcast => 3,
            Self::Matmul => 4,
            Self::MatmulVec4 => 5,
            Self::Permute => 6,
            Self::LayerNormStatistics => 7,
            Self::LayerNormAffine => 8,
            Self::StablePool1D => 9,
            Self::Conv1DForward => 10,
        }
    }

    fn is_initialized(self, provider: &WgpuProvider) -> bool {
        match self {
            Self::Fill => provider.pipelines.fill.is_initialized(),
            Self::ScalarMax => provider.pipelines.scalar_max.is_initialized(),
            Self::Binary => provider.pipelines.binary.is_initialized(),
            Self::BinaryBroadcast => provider.pipelines.binary_broadcast.is_initialized(),
            Self::Matmul => provider.pipelines.matmul.is_initialized(),
            Self::MatmulVec4 => provider.pipelines.matmul_vec4.is_initialized(),
            Self::Permute => provider.pipelines.permute.is_initialized(),
            Self::LayerNormStatistics => provider
                .pipelines
                .deep_learning_layer_norm_statistics
                .is_initialized(),
            Self::LayerNormAffine => provider
                .pipelines
                .deep_learning_layer_norm_affine
                .is_initialized(),
            Self::StablePool1D => provider
                .pipelines
                .deep_learning_stable_pool_1d
                .is_initialized(),
            Self::Conv1DForward => provider
                .pipelines
                .deep_learning_conv1d_forward
                .is_initialized(),
        }
    }
}

impl WgpuProvider {
    pub(crate) fn deep_learning_forward_preparation_snapshot_exec(
        &self,
        route: &DeepLearningForwardRoute<'_>,
    ) -> Option<ProviderPreparationSnapshot> {
        if !matches!(
            self.deep_learning_forward_route_capability_exec(route)
                .ok()?,
            DeepLearningForwardCapability::Ready(_)
        ) {
            return None;
        }
        let geometry = route.analyze().ok()?;
        let mut fixed = Vec::new();
        let mut dynamic = Vec::new();
        let element_type = route.input.element_type();
        let parameter_views = route
            .parameters
            .iter()
            .map(|parameter| {
                let outcome = ProviderViewChain {
                    source: ProviderViewSource::Current,
                    steps: parameter.view_steps.clone(),
                }
                .analyze(&parameter.source.view(), element_type)
                .ok()?;
                if outcome.materialization_bytes != 0 {
                    fixed.push(FixedPipeline::Permute);
                }
                Some(outcome.view)
            })
            .collect::<Option<Vec<_>>>()?;
        for (step, input) in route.steps.iter().zip(&geometry.stage_inputs) {
            match &step.operation {
                DeepLearningForwardOperation::View(view) => {
                    let outcome = ProviderViewChain {
                        source: ProviderViewSource::Current,
                        steps: vec![view.clone()],
                    }
                    .analyze(input, element_type)
                    .ok()?;
                    if outcome.materialization_bytes != 0 {
                        fixed.push(FixedPipeline::Permute);
                    }
                }
                DeepLearningForwardOperation::FeedForward(stage) => match stage {
                    DeepLearningFeedForwardStage::Transpose2D { .. }
                    | DeepLearningFeedForwardStage::Permute2D { .. } => {
                        let view_step = match stage {
                            DeepLearningFeedForwardStage::Transpose2D { .. } => {
                                ProviderViewStep::Transpose2D
                            }
                            DeepLearningFeedForwardStage::Permute2D { .. } => {
                                ProviderViewStep::Permute2D
                            }
                            _ => unreachable!(),
                        };
                        let outcome = ProviderViewChain {
                            source: ProviderViewSource::Current,
                            steps: vec![view_step],
                        }
                        .analyze(input, element_type)
                        .ok()?;
                        if outcome.materialization_bytes != 0 {
                            fixed.push(FixedPipeline::Permute);
                        }
                    }
                    DeepLearningFeedForwardStage::Dense {
                        batch,
                        input_channels,
                        output_channels,
                    } => {
                        let DeepLearningForwardParameters::Dense { weights, bias } =
                            step.parameters
                        else {
                            return None;
                        };
                        let weights_view = parameter_views.get(weights)?;
                        let bias_view = parameter_views.get(bias)?;
                        if bias_view.shape() != [1, *output_channels] {
                            return None;
                        }
                        match selected_matmul_pipeline(
                            self.precision,
                            *batch,
                            *output_channels,
                            *input_channels,
                            input.layout() != GpuTensorLayout::Contiguous,
                            weights_view.layout() != GpuTensorLayout::Contiguous,
                        ) {
                            MatmulPipelineRoute::Empty => {}
                            MatmulPipelineRoute::Scalar => fixed.push(FixedPipeline::Matmul),
                            MatmulPipelineRoute::Vec4 => fixed.push(FixedPipeline::MatmulVec4),
                            MatmulPipelineRoute::Chunked => {
                                fixed.push(FixedPipeline::Matmul);
                                fixed.push(FixedPipeline::Binary);
                            }
                        }
                        if *batch != 0 && *output_channels != 0 {
                            fixed.push(if *batch == 1 {
                                FixedPipeline::Binary
                            } else {
                                FixedPipeline::BinaryBroadcast
                            });
                        }
                    }
                    DeepLearningFeedForwardStage::Relu { elements } if *elements != 0 => {
                        fixed.push(FixedPipeline::ScalarMax);
                    }
                    DeepLearningFeedForwardStage::Relu { .. } => {}
                    DeepLearningFeedForwardStage::Elu { elements } if *elements != 0 => {
                        fixed.push(FixedPipeline::Fill);
                        dynamic.push(DynamicPipeline::Legacy(
                            PipelineRecipe::fusion(
                                super::deep_learning::elu_shader(self.precision),
                                2,
                                FusionPipelineLayout::Single,
                                self.workgroup_config.scalar,
                            )
                            .key(),
                        ));
                    }
                    DeepLearningFeedForwardStage::Elu { .. } => {}
                    DeepLearningFeedForwardStage::SoftmaxRows { rows, columns }
                        if *rows != 0 && *columns != 0 =>
                    {
                        dynamic.push(DynamicPipeline::Validated(
                            super::deep_learning::softmax_rows_pipeline_key(
                                self.precision,
                                self.workgroup_config.scalar,
                            ),
                        ));
                    }
                    DeepLearningFeedForwardStage::SoftmaxRows { .. } => {}
                },
                DeepLearningForwardOperation::Pool1D { .. } => {
                    if input.layout() != GpuTensorLayout::Contiguous {
                        fixed.push(FixedPipeline::Permute);
                    }
                    if input.shape().iter().all(|extent| *extent != 0) {
                        fixed.push(FixedPipeline::StablePool1D);
                    }
                }
                DeepLearningForwardOperation::LayerNorm(request) => {
                    let DeepLearningForwardParameters::LayerNorm { scale, offset } =
                        step.parameters
                    else {
                        return None;
                    };
                    if input.layout() != GpuTensorLayout::Contiguous
                        || [scale, offset].into_iter().any(|index| {
                            parameter_views[index].layout() != GpuTensorLayout::Contiguous
                        })
                    {
                        fixed.push(FixedPipeline::Permute);
                    }
                    if request.shape.iter().all(|extent| *extent != 0) {
                        fixed.push(FixedPipeline::LayerNormStatistics);
                        fixed.push(FixedPipeline::LayerNormAffine);
                    }
                }
                DeepLearningForwardOperation::Conv1D(request) => {
                    if request.output_shape.iter().all(|extent| *extent != 0) {
                        fixed.push(FixedPipeline::Conv1DForward);
                    }
                }
            }
        }
        fixed.sort_unstable();
        fixed.dedup();
        dynamic.sort_unstable();
        dynamic.dedup();

        let mut bytes = b"runmat.wgpu.network-preparation.v1\0".to_vec();
        bytes.extend_from_slice(route.timing_route_digest().bytes());
        bytes.push(match self.precision {
            NumericPrecision::F32 => 0,
            NumericPrecision::F64 => 1,
        });
        bytes.extend_from_slice(&self.workgroup_config.scalar.to_le_bytes());
        bytes.extend(fixed.iter().map(|pipeline| pipeline.tag()));
        for pipeline in &dynamic {
            let (key, validated) = match pipeline {
                DynamicPipeline::Legacy(key) => (key, false),
                DynamicPipeline::Validated(key) => (key, true),
            };
            bytes.push(u8::from(validated));
            bytes.extend_from_slice(&key.to_le_bytes());
        }
        let identity = Digest::sha256(bytes);
        let fixed_cold = fixed
            .iter()
            .filter(|pipeline| !pipeline.is_initialized(self))
            .count();
        let fixed_warm = fixed.len().checked_sub(fixed_cold)?;
        let (state, cold_pipelines, warm_pipelines) = if dynamic.is_empty() {
            (Some(()), fixed_cold, fixed_warm)
        } else {
            let legacy = dynamic
                .iter()
                .any(|pipeline| matches!(pipeline, DynamicPipeline::Legacy(_)))
                .then(|| self.fused_pipeline_cache.try_lock().ok())
                .flatten();
            let validated = dynamic
                .iter()
                .any(|pipeline| matches!(pipeline, DynamicPipeline::Validated(_)))
                .then(|| self.validated_fusion_cache.try_lock().ok())
                .flatten();
            let dynamic_cold = dynamic.iter().try_fold(0usize, |cold, pipeline| {
                Some(
                    cold + usize::from(!pipeline.is_warm(legacy.as_deref(), validated.as_deref())?),
                )
            });
            if let Some(dynamic_cold) = dynamic_cold {
                (
                    Some(()),
                    fixed_cold.checked_add(dynamic_cold)?,
                    fixed_warm.checked_add(dynamic.len().checked_sub(dynamic_cold)?)?,
                )
            } else {
                (None, 0, 0)
            }
        };
        Some(ProviderPreparationSnapshot {
            identity,
            state: state.map(|_| {
                if fixed.is_empty() && dynamic.is_empty() {
                    CandidatePreparationState::Ready
                } else if cold_pipelines == 0 {
                    CandidatePreparationState::Warm
                } else {
                    CandidatePreparationState::Cold
                }
            }),
            cold_pipelines: u32::try_from(cold_pipelines).ok()?,
            warm_pipelines: u32::try_from(warm_pipelines).ok()?,
        })
    }
}

#[cfg(test)]
#[path = "deep_learning/tests.rs"]
mod tests;
