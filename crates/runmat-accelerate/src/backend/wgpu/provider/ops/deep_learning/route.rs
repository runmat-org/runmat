//! Effect-free WGPU admission of an already-lowered whole forward route.

use super::*;
use runmat_accelerate_api::{
    DeepLearningConv1DCapability, DeepLearningFeedForwardCapability, DeepLearningForwardCapability,
    DeepLearningForwardOperation, DeepLearningForwardResources, DeepLearningForwardRoute,
    DeepLearningForwardSource, DeepLearningLayerNormCapability, DeepLearningPool1DCapability,
    ProviderViewCapability, ProviderViewChain, ProviderViewSource,
};
use std::collections::HashSet;

fn checked_shape_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| anyhow!("forward source shape product overflows"))
}

#[derive(Clone, Copy)]
enum SourceCapability {
    Ready,
    Unsupported,
    ResourceLimit,
}

impl WgpuProvider {
    fn forward_source_capability(
        &self,
        source: &DeepLearningForwardSource<'_>,
    ) -> Result<SourceCapability> {
        let element_type = source.element_type();
        let matching_precision = self.supports_native_real_element_type(element_type);
        match source {
            DeepLearningForwardSource::Host { shape, .. } => {
                if !matching_precision {
                    return Ok(SourceCapability::Unsupported);
                }
                let count = match checked_shape_count(shape) {
                    Ok(count) => count,
                    Err(_) => return Ok(SourceCapability::ResourceLimit),
                };
                if !self.within_binding_limit(count)
                    || shape.iter().any(|extent| *extent > u32::MAX as usize)
                {
                    return Ok(SourceCapability::ResourceLimit);
                }
                Ok(SourceCapability::Ready)
            }
            DeepLearningForwardSource::Resident(handle) => {
                let entry = self.get_entry_raw(handle)?;
                ensure!(
                    entry.element_type == handle.physical().element_type()
                        && entry.storage == handle.physical().storage()
                        && entry.len == checked_shape_count(handle.shape())?
                        && entry.logical == runmat_accelerate_api::handle_is_logical(handle)
                        && entry.integer_type()
                            == runmat_accelerate_api::handle_integer_type(handle),
                    "forward source differs from provider-owned descriptor"
                );
                if matching_precision
                    && entry.storage == GpuTensorStorage::Real
                    && !entry.logical
                    && entry.integer_type().is_none()
                {
                    Ok(SourceCapability::Ready)
                } else {
                    Ok(SourceCapability::Unsupported)
                }
            }
        }
    }

    /// No upload, allocation, dispatch, provider timing, or schedule decision
    /// occurs here. Only the actual WGPU owner's limits and implementations
    /// can declare a route ready.
    pub(crate) fn deep_learning_forward_route_capability_exec(
        &self,
        route: &DeepLearningForwardRoute<'_>,
    ) -> Result<DeepLearningForwardCapability> {
        let geometry = match route.analyze() {
            Ok(geometry) => geometry,
            Err(runmat_accelerate_api::DeepLearningForwardRouteError::SizeOverflow) => {
                return Ok(DeepLearningForwardCapability::ResourceLimit)
            }
            Err(error) => return Err(anyhow!("invalid physical forward route: {error:?}")),
        };
        let sources =
            std::iter::once(&route.input).chain(route.parameters.iter().map(|p| &p.source));
        let mut allocations = HashSet::new();
        let mut upload_bytes = 0_u64;
        let mut resident_bytes = 0_u64;
        for source in sources {
            match self.forward_source_capability(source)? {
                SourceCapability::Ready => {}
                SourceCapability::Unsupported => {
                    return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                }
                SourceCapability::ResourceLimit => {
                    return Ok(DeepLearningForwardCapability::ResourceLimit)
                }
            }
            match source {
                DeepLearningForwardSource::Host { .. } => {
                    upload_bytes =
                        upload_bytes
                            .checked_add(source.logical_bytes().map_err(|error| {
                                anyhow!("invalid host upload geometry: {error:?}")
                            })?)
                            .ok_or_else(|| anyhow!("forward upload byte sum overflows"))?;
                }
                DeepLearningForwardSource::Resident(handle) => {
                    if allocations.insert(handle.allocation_identity()) {
                        resident_bytes = resident_bytes
                            .checked_add(handle.physical().requested_bytes())
                            .ok_or_else(|| anyhow!("forward resident byte sum overflows"))?;
                    }
                }
            }
        }

        let element_type = route.input.element_type();
        for parameter in &route.parameters {
            let query = ProviderViewChain {
                source: ProviderViewSource::Current,
                steps: parameter.view_steps.clone(),
            };
            match self.provider_view_chain_capability_for_type(
                &parameter.source.view(),
                element_type,
                &query,
            )? {
                ProviderViewCapability::Ready(_) => {}
                ProviderViewCapability::UnsupportedOperation => {
                    return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                }
                ProviderViewCapability::ResourceLimit => {
                    return Ok(DeepLearningForwardCapability::ResourceLimit)
                }
            }
        }
        let mut scratch_bytes = 0_u64;
        for (step, input_view) in route.steps.iter().zip(&geometry.stage_inputs) {
            match &step.operation {
                DeepLearningForwardOperation::View(operation) => {
                    let query = ProviderViewChain {
                        source: ProviderViewSource::Current,
                        steps: vec![operation.clone()],
                    };
                    match self.provider_view_chain_capability_for_type(
                        input_view,
                        element_type,
                        &query,
                    )? {
                        ProviderViewCapability::Ready(_) => {}
                        ProviderViewCapability::UnsupportedOperation => {
                            return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                        }
                        ProviderViewCapability::ResourceLimit => {
                            return Ok(DeepLearningForwardCapability::ResourceLimit)
                        }
                    }
                }
                DeepLearningForwardOperation::FeedForward(stage) => {
                    match self
                        .deep_learning_feed_forward_stage_capability_for_type(element_type, *stage)
                    {
                        DeepLearningFeedForwardCapability::Ready => {}
                        DeepLearningFeedForwardCapability::UnsupportedOperation => {
                            return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                        }
                        DeepLearningFeedForwardCapability::ResourceLimit => {
                            return Ok(DeepLearningForwardCapability::ResourceLimit)
                        }
                    }
                }
                DeepLearningForwardOperation::Pool1D { input_shape, route } => {
                    match self.deep_learning_global_average_pool_1d_capability_for_type(
                        element_type,
                        input_shape,
                        route,
                    )? {
                        DeepLearningPool1DCapability::Ready => {}
                        DeepLearningPool1DCapability::UnsupportedOperation => {
                            return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                        }
                        DeepLearningPool1DCapability::ResourceLimit => {
                            return Ok(DeepLearningForwardCapability::ResourceLimit)
                        }
                    }
                }
                DeepLearningForwardOperation::LayerNorm(request) => {
                    match self
                        .deep_learning_layer_norm_capability_for_type(element_type, request)?
                    {
                        DeepLearningLayerNormCapability::Ready => {
                            scratch_bytes = scratch_bytes
                                .max(self.deep_learning_layer_norm_scratch_bytes(request)?);
                        }
                        DeepLearningLayerNormCapability::UnsupportedOperation => {
                            return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                        }
                        DeepLearningLayerNormCapability::ResourceLimit => {
                            return Ok(DeepLearningForwardCapability::ResourceLimit)
                        }
                    }
                }
                DeepLearningForwardOperation::Conv1D(request) => {
                    match self.deep_learning_conv1d_capability_for_type(element_type, request)? {
                        DeepLearningConv1DCapability::Ready => {}
                        DeepLearningConv1DCapability::UnsupportedOperation => {
                            return Ok(DeepLearningForwardCapability::UnsupportedOperation)
                        }
                        DeepLearningConv1DCapability::ResourceLimit => {
                            return Ok(DeepLearningForwardCapability::ResourceLimit)
                        }
                    }
                }
            }
        }
        let new_buffer_bytes_upper_bound = match self.forward_buffer_bound(route, &geometry) {
            Ok(bound) => bound,
            Err(super::deep_learning_resource_bound::BufferBoundError::Overflow) => {
                return Ok(DeepLearningForwardCapability::ResourceLimit)
            }
        };
        Ok(DeepLearningForwardCapability::Ready(
            DeepLearningForwardResources {
                output_bytes: geometry.output_bytes,
                largest_stage_value_bytes: geometry.largest_stage_value_bytes,
                known_normalization_scratch_bytes: scratch_bytes,
                upload_bytes,
                resident_bytes,
                new_buffer_bytes_upper_bound,
            },
        ))
    }
}

#[cfg(test)]
#[path = "route/tests.rs"]
mod tests;
