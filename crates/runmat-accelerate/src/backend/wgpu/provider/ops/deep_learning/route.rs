//! Effect-free WGPU admission of an already-lowered whole forward route.

use super::*;
use runmat_accelerate_api::{
    DeepLearningFeedForwardCapability, DeepLearningForwardCapability, DeepLearningForwardOperation,
    DeepLearningForwardResources, DeepLearningForwardRoute, DeepLearningForwardSource,
    DeepLearningLayerNormCapability, DeepLearningPool1DCapability, NumericElementType,
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
            }
        }
        Ok(DeepLearningForwardCapability::Ready(
            DeepLearningForwardResources {
                output_bytes: geometry.output_bytes,
                largest_intermediate_bytes: geometry.largest_intermediate_bytes,
                known_normalization_scratch_bytes: scratch_bytes,
                upload_bytes,
                resident_bytes,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        AccelProvider, DeepLearningFeedForwardStage, DeepLearningForwardParameter,
        DeepLearningForwardParameters, DeepLearningForwardStep, GpuTensorView, HostNumericDataView,
        HostNumericTensorView, ProviderViewStep,
    };

    fn element_type(provider: &WgpuProvider) -> NumericElementType {
        match provider.precision() {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        }
    }

    fn route<'a>(source: DeepLearningForwardSource<'a>) -> DeepLearningForwardRoute<'a> {
        DeepLearningForwardRoute {
            input: source,
            parameters: Vec::new(),
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::Relu { elements: 6 },
                ),
                parameters: DeepLearningForwardParameters::None,
            }],
            output_view: GpuTensorView::contiguous(vec![2, 3]),
        }
    }

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for effect-free forward-route admission"]
    fn host_and_resident_routes_are_admitted_without_provider_effects() {
        let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
        let class = element_type(&provider);
        let input_f32 = [1.0_f32; 6];
        let input_f64 = [1.0_f64; 6];
        let resident = provider
            .upload_numeric(&HostNumericTensorView {
                data: match class {
                    NumericElementType::F32 => HostNumericDataView::F32(&input_f32),
                    NumericElementType::F64 => HostNumericDataView::F64(&input_f64),
                    _ => unreachable!("provider precision is floating"),
                },
                shape: &[2, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let parameter_f32 = [2.0_f32];
        let parameter_f64 = [2.0_f64];
        let shared_parameter = provider
            .upload_numeric(&HostNumericTensorView {
                data: match class {
                    NumericElementType::F32 => HostNumericDataView::F32(&parameter_f32),
                    NumericElementType::F64 => HostNumericDataView::F64(&parameter_f64),
                    _ => unreachable!("provider precision is floating"),
                },
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let small_input = provider
            .upload_numeric(&HostNumericTensorView {
                data: match class {
                    NumericElementType::F32 => HostNumericDataView::F32(&input_f32[..2]),
                    NumericElementType::F64 => HostNumericDataView::F64(&input_f64[..2]),
                    _ => unreachable!("provider precision is floating"),
                },
                shape: &[2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        let host = route(DeepLearningForwardSource::Host {
            element_type: class,
            shape: vec![2, 3],
        });
        let resident_route = route(DeepLearningForwardSource::Resident(&resident));
        for candidate in [&host, &resident_route] {
            let Some(DeepLearningForwardCapability::Ready(resources)) = provider
                .deep_learning_forward_route_capability(candidate)
                .unwrap()
            else {
                panic!("legal physical route should be ready");
            };
            assert_eq!(resources.output_bytes, 6 * class.element_size() as u64);
        }
        let shared = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Resident(&small_input),
            parameters: vec![
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Resident(&shared_parameter),
                    view_steps: vec![ProviderViewStep::Transpose2D],
                },
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Resident(&shared_parameter),
                    view_steps: vec![ProviderViewStep::Reshape(vec![1, 1])],
                },
            ],
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::Dense {
                        batch: 2,
                        input_channels: 1,
                        output_channels: 1,
                    },
                ),
                parameters: DeepLearningForwardParameters::Dense {
                    weights: 0,
                    bias: 1,
                },
            }],
            output_view: GpuTensorView::contiguous(vec![2, 1]),
        };
        let Some(DeepLearningForwardCapability::Ready(shared_resources)) = provider
            .deep_learning_forward_route_capability(&shared)
            .unwrap()
        else {
            panic!("same allocation may supply two admitted parameter views")
        };
        assert_eq!(
            shared_resources.resident_bytes,
            small_input.physical().requested_bytes()
                + shared_parameter.physical().requested_bytes()
        );
        let mut unsupported = host.clone();
        unsupported.input = DeepLearningForwardSource::Host {
            element_type: NumericElementType::I32,
            shape: vec![2, 3],
        };
        assert_eq!(
            provider
                .deep_learning_forward_route_capability(&unsupported)
                .unwrap(),
            Some(DeepLearningForwardCapability::UnsupportedOperation)
        );
        let mut malformed = host;
        malformed.steps[0].parameters = DeepLearningForwardParameters::Dense {
            weights: 0,
            bias: 1,
        };
        assert!(provider
            .deep_learning_forward_route_capability(&malformed)
            .is_err());
        assert_eq!(provider.telemetry_snapshot(), before);
        resident.release().unwrap();
        small_input.release().unwrap();
        shared_parameter.release().unwrap();
    }
}
