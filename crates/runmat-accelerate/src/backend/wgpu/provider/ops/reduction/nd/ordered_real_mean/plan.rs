//! The one provider-owned, handle-free ordered real mean derivation. A recipe
//! is only a proposal until a live source view is authenticated and matched.

use super::{physical_precision, provider_precision, NdReductionGeometry, WgpuProvider, LANES};
use crate::backend::wgpu::{params::ReduceNdParams, types::NumericPrecision};
use anyhow::Result;
use runmat_accelerate_api::{
    ProviderElementType, ProviderLayout, ProviderOrderedRealMeanControl,
    ProviderOrderedRealMeanLimits, ProviderOrderedRealMeanRecipe, ProviderOrderedRealMeanResources,
    ProviderOrderedRealMeanSource, ProviderOrderedRealMeanStep, ProviderPrecision,
    ProviderRealMeanNaNMode, ProviderRepresentation, ProviderResidency, ProviderStorage,
};

#[derive(Clone, Copy)]
struct OrderedMeanCapabilities {
    limits: ProviderOrderedRealMeanLimits,
    provider_precision: NumericPrecision,
    pipeline_available: bool,
}

impl WgpuProvider {
    pub(super) fn ordered_mean_limits(&self) -> ProviderOrderedRealMeanLimits {
        ProviderOrderedRealMeanLimits {
            max_buffer_size: self.adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: self.adapter_limits.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: self.adapter_limits.max_uniform_buffer_binding_size,
            max_compute_invocations_per_workgroup: self
                .adapter_limits
                .max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: self.adapter_limits.max_compute_workgroup_size_x,
            max_compute_workgroups_per_dimension: self
                .adapter_limits
                .max_compute_workgroups_per_dimension,
        }
    }

    pub(super) fn derive_ordered_mean_recipe(
        &self,
        source: &ProviderRepresentation,
        source_buffer_bytes: u64,
        axes: &[usize],
        output_precision: ProviderPrecision,
        nan_mode: ProviderRealMeanNaNMode,
    ) -> Result<Option<ProviderOrderedRealMeanRecipe>> {
        derive_recipe(
            OrderedMeanCapabilities {
                limits: self.ordered_mean_limits(),
                provider_precision: self.precision,
                pipeline_available: self.ordered_real_mean.is_some()
                    && self.ieee_words_available(NumericPrecision::F64),
            },
            ProviderOrderedRealMeanSource {
                representation: source.clone(),
                buffer_bytes: source_buffer_bytes,
            },
            ProviderOrderedRealMeanControl {
                axes: axes.to_vec(),
                output_precision,
                nan_mode,
            },
        )
    }
}

fn derive_recipe(
    capabilities: OrderedMeanCapabilities,
    source_facts: ProviderOrderedRealMeanSource,
    control: ProviderOrderedRealMeanControl,
) -> Result<Option<ProviderOrderedRealMeanRecipe>> {
    let limits = capabilities.limits;
    let source = &source_facts.representation;
    let source_buffer_bytes = source_facts.buffer_bytes;
    let axes = &control.axes;
    if capabilities.provider_precision != NumericPrecision::F64
        || !capabilities.pipeline_available
        || source.storage != ProviderStorage::DenseReal
        || source.layout != ProviderLayout::ColumnMajorContiguous
        || source.residency != ProviderResidency::Device
        || !matches!(
            source.element_type,
            ProviderElementType::F32 | ProviderElementType::F64
        )
        || axes.is_empty()
        || axes.windows(2).any(|pair| pair[0] >= pair[1])
        || axes.iter().any(|&axis| axis >= source.shape.len())
        || source.shape.contains(&0)
        || LANES > limits.max_compute_invocations_per_workgroup
        || LANES > limits.max_compute_workgroup_size_x
    {
        return Ok(None);
    }
    let Some(logical_bytes) = source.checked_byte_len() else {
        return Ok(None);
    };
    let uniform_bytes = std::mem::size_of::<ReduceNdParams>() as u64;
    if source_buffer_bytes < logical_bytes
        || source_buffer_bytes > limits.max_buffer_size
        || source_buffer_bytes > u64::from(limits.max_storage_buffer_binding_size)
        || uniform_bytes > limits.max_buffer_size
        || uniform_bytes > u64::from(limits.max_uniform_buffer_binding_size)
    {
        return Ok(None);
    }
    let Ok(mut shape) = source
        .shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
    else {
        return Ok(None);
    };
    let mut input_precision = match source.element_type {
        ProviderElementType::F32 => NumericPrecision::F32,
        ProviderElementType::F64 => NumericPrecision::F64,
        _ => unreachable!("checked real floating source"),
    };
    let mut input_bytes = source_buffer_bytes;
    let mut steps = Vec::with_capacity(axes.len());
    let mut incremental_peak_bytes = 0u64;
    let mut queue_submissions = 0u32;
    for (ordinal, &axis) in axes.iter().enumerate() {
        let Ok(geometry) = NdReductionGeometry::new(&shape, &[axis], "ordered real mean") else {
            return Ok(None);
        };
        let output_precision = if ordinal + 1 == axes.len() {
            physical_precision(control.output_precision)
        } else {
            NumericPrecision::F64
        };
        let bytes_per_element = match output_precision {
            NumericPrecision::F32 => 4u64,
            NumericPrecision::F64 => 8u64,
        };
        let output_bytes = u64::try_from(geometry.cols)
            .ok()
            .and_then(|cols| cols.checked_mul(bytes_per_element));
        let workgroups = u32::try_from(geometry.cols.div_ceil(LANES as usize)).ok();
        let (Some(output_bytes), Some(workgroups)) = (output_bytes, workgroups) else {
            return Ok(None);
        };
        if input_bytes > limits.max_buffer_size
            || input_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || output_bytes > limits.max_buffer_size
            || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || workgroups > limits.max_compute_workgroups_per_dimension
        {
            return Ok(None);
        }
        let Some(next_peak) = incremental_peak_bytes
            .checked_add(output_bytes)
            .and_then(|bytes| bytes.checked_add(uniform_bytes))
        else {
            return Ok(None);
        };
        incremental_peak_bytes = next_peak;
        steps.push(
            ProviderOrderedRealMeanStep::new(
                axis,
                shape.clone(),
                geometry.out_shape.clone(),
                provider_precision(input_precision),
                provider_precision(output_precision),
            )
            .with_physical_resources(
                input_bytes,
                output_bytes,
                uniform_bytes,
                workgroups,
            ),
        );
        shape = geometry.out_shape;
        input_precision = output_precision;
        input_bytes = output_bytes;
        // execute_ordered_mean_plan_with invokes elementwise::run once for
        // this step; that owner submits exactly one command encoder.
        let Some(next) = queue_submissions.checked_add(1) else {
            return Ok(None);
        };
        queue_submissions = next;
    }
    Ok(Some(ProviderOrderedRealMeanRecipe::new(
        source_facts,
        control,
        ProviderOrderedRealMeanResources {
            limits,
            steps,
            incremental_peak_bytes,
            queue_submissions,
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ProviderOrderedRealMeanLimits {
        ProviderOrderedRealMeanLimits {
            max_buffer_size: 1 << 20,
            max_storage_buffer_binding_size: 1 << 20,
            max_uniform_buffer_binding_size: 1 << 16,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroups_per_dimension: 1 << 16,
        }
    }

    fn source() -> ProviderRepresentation {
        ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 4, 5],
            residency: ProviderResidency::Device,
        }
    }

    fn plan(
        source: &ProviderRepresentation,
        bytes: u64,
        axes: &[usize],
        nan_mode: ProviderRealMeanNaNMode,
    ) -> Option<ProviderOrderedRealMeanRecipe> {
        derive_recipe(
            OrderedMeanCapabilities {
                limits: limits(),
                provider_precision: NumericPrecision::F64,
                pipeline_available: true,
            },
            ProviderOrderedRealMeanSource {
                representation: source.clone(),
                buffer_bytes: bytes,
            },
            ProviderOrderedRealMeanControl {
                axes: axes.to_vec(),
                output_precision: ProviderPrecision::F32,
                nan_mode,
            },
        )
        .unwrap()
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn future_source_recipe_keeps_ordered_double_scratch_and_physical_extents() {
        let recipe = plan(&source(), 256, &[1, 2], ProviderRealMeanNaNMode::Include).unwrap();
        assert_eq!(recipe.source(), &source());
        assert_eq!(recipe.source_buffer_bytes(), 256);
        assert_eq!(recipe.axes(), &[1, 2]);
        assert_eq!(recipe.steps().len(), 2);
        assert_eq!(recipe.steps()[0].input_bytes(), 256);
        assert_eq!(recipe.steps()[0].input_precision(), ProviderPrecision::F32);
        assert_eq!(recipe.steps()[0].output_precision(), ProviderPrecision::F64);
        assert_eq!(recipe.steps()[0].output_shape(), &[2, 1, 5]);
        assert_eq!(recipe.steps()[0].output_bytes(), 80);
        assert_eq!(recipe.steps()[1].input_bytes(), 80);
        assert_eq!(recipe.steps()[1].input_precision(), ProviderPrecision::F64);
        assert_eq!(recipe.steps()[1].output_precision(), ProviderPrecision::F32);
        assert_eq!(recipe.steps()[1].output_shape(), &[2, 1, 1]);
        assert_eq!(recipe.steps()[1].output_bytes(), 8);
        assert_eq!(
            recipe.requested_incremental_peak_bytes(),
            88 + 2 * std::mem::size_of::<ReduceNdParams>() as u64
        );
        assert_ne!(
            recipe,
            plan(&source(), 256, &[1, 2], ProviderRealMeanNaNMode::Omit).unwrap()
        );
        assert_ne!(
            recipe,
            plan(&source(), 160, &[1, 2], ProviderRealMeanNaNMode::Include).unwrap()
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn symbolic_recipe_rejects_unrepresentable_source_and_capability() {
        let valid = source();
        assert!(plan(&valid, 159, &[1, 2], ProviderRealMeanNaNMode::Include).is_none());
        assert!(plan(&valid, 160, &[2, 1], ProviderRealMeanNaNMode::Include).is_none());
        assert!(plan(&valid, 160, &[1, 1], ProviderRealMeanNaNMode::Include).is_none());
        assert!(plan(&valid, 160, &[], ProviderRealMeanNaNMode::Include).is_none());
        assert!(plan(
            &ProviderRepresentation {
                shape: vec![u64::MAX, 2],
                ..valid.clone()
            },
            u64::MAX,
            &[1],
            ProviderRealMeanNaNMode::Include,
        )
        .is_none());
        for changed in [
            ProviderRepresentation {
                element_type: ProviderElementType::Logical,
                ..valid.clone()
            },
            ProviderRepresentation {
                storage: ProviderStorage::Sparse,
                ..valid.clone()
            },
            ProviderRepresentation {
                layout: ProviderLayout::Strided,
                ..valid.clone()
            },
            ProviderRepresentation {
                residency: ProviderResidency::Host,
                ..valid.clone()
            },
            ProviderRepresentation {
                shape: vec![2, 0, 5],
                ..valid.clone()
            },
        ] {
            assert!(plan(&changed, 160, &[1, 2], ProviderRealMeanNaNMode::Include).is_none());
        }
        let mut too_small = limits();
        too_small.max_storage_buffer_binding_size = 159;
        let source_facts = ProviderOrderedRealMeanSource {
            representation: valid.clone(),
            buffer_bytes: 160,
        };
        let control = ProviderOrderedRealMeanControl {
            axes: vec![1, 2],
            output_precision: ProviderPrecision::F32,
            nan_mode: ProviderRealMeanNaNMode::Include,
        };
        assert!(derive_recipe(
            OrderedMeanCapabilities {
                limits: too_small,
                provider_precision: NumericPrecision::F64,
                pipeline_available: true,
            },
            source_facts.clone(),
            control.clone(),
        )
        .unwrap()
        .is_none());
        assert!(derive_recipe(
            OrderedMeanCapabilities {
                limits: limits(),
                provider_precision: NumericPrecision::F32,
                pipeline_available: true,
            },
            source_facts.clone(),
            control.clone(),
        )
        .unwrap()
        .is_none());
        assert!(derive_recipe(
            OrderedMeanCapabilities {
                limits: limits(),
                provider_precision: NumericPrecision::F64,
                pipeline_available: false,
            },
            source_facts,
            control,
        )
        .unwrap()
        .is_none());
    }
}
