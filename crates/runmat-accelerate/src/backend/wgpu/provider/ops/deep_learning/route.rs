//! Effect-free WGPU admission of an already-lowered whole forward route.

use super::*;
use runmat_accelerate_api::{
    DeepLearningFeedForwardCapability, DeepLearningForwardCapability, DeepLearningForwardOperation,
    DeepLearningForwardResources, DeepLearningForwardRoute, DeepLearningForwardSource,
    DeepLearningLayerNormCapability, DeepLearningPool1DCapability, ProviderViewCapability,
    ProviderViewChain, ProviderViewSource,
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
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        AccelProvider, DeepLearningFeedForwardStage, DeepLearningForwardBufferBound,
        DeepLearningForwardParameter, DeepLearningForwardParameters, DeepLearningForwardStep,
        GpuTensorView, HostNumericDataView, HostNumericTensorView, NumericElementType,
        ProviderViewStep,
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
        for (candidate, expected_upload) in [
            (&host, 6 * class.element_size() as u64),
            (&resident_route, 0),
        ] {
            let Some(DeepLearningForwardCapability::Ready(resources)) = provider
                .deep_learning_forward_route_capability(candidate)
                .unwrap()
            else {
                panic!("legal physical route should be ready");
            };
            assert_eq!(resources.output_bytes, 6 * class.element_size() as u64);
            assert_eq!(
                resources.new_buffer_bytes_upper_bound,
                DeepLearningForwardBufferBound::Known(
                    expected_upload + 6 * class.element_size() as u64 + 256
                )
            );
        }
        let mut two_relu = host.clone();
        two_relu.steps.push(two_relu.steps[0].clone());
        let Some(DeepLearningForwardCapability::Ready(two_relu_resources)) = provider
            .deep_learning_forward_route_capability(&two_relu)
            .unwrap()
        else {
            panic!("two admitted ReLU stages should remain ready")
        };
        assert_eq!(
            two_relu_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(3 * 6 * class.element_size() as u64 + 512),
            "pending earlier stages cannot be omitted from the conservative sum"
        );
        let mut softmax = host.clone();
        softmax.steps[0].operation =
            DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::SoftmaxRows {
                rows: 2,
                columns: 3,
            });
        let Some(DeepLearningForwardCapability::Ready(softmax_resources)) = provider
            .deep_learning_forward_route_capability(&softmax)
            .unwrap()
        else {
            panic!("admitted softmax should remain ready")
        };
        assert_eq!(
            softmax_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(2 * 6 * class.element_size() as u64 + 8 + 16),
        );
        let mut elu = host.clone();
        elu.steps[0].operation =
            DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::Elu {
                elements: 6,
            });
        let Some(DeepLearningForwardCapability::Ready(elu_resources)) = provider
            .deep_learning_forward_route_capability(&elu)
            .unwrap()
        else {
            panic!("admitted ELU should remain ready")
        };
        let fill_uniform = match class {
            NumericElementType::F32 => {
                std::mem::size_of::<crate::backend::wgpu::params::FillParamsF32>() as u64
            }
            NumericElementType::F64 => {
                std::mem::size_of::<crate::backend::wgpu::params::FillParamsF64>() as u64
            }
            _ => unreachable!("provider precision is floating"),
        };
        assert_eq!(
            elu_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(
                3 * 6 * class.element_size() as u64 + fill_uniform + 16
            ),
        );
        let pooling = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: class,
                shape: vec![2, 3],
            },
            parameters: Vec::new(),
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::Pool1D {
                    input_shape: vec![2, 3],
                    route: runmat_accelerate_api::DeepLearningPool1DRoute {
                        reduction: runmat_accelerate_api::DeepLearningPool1DRequest {
                            axis_zero_based: 1,
                        },
                        result_axis:
                            runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
                        source_shape: vec![2, 1],
                        target_shape: vec![2, 1],
                        target_to_source: vec![0, 1],
                    },
                },
                parameters: DeepLearningForwardParameters::None,
            }],
            output_view: GpuTensorView::contiguous(vec![2, 1]),
        };
        let Some(DeepLearningForwardCapability::Ready(pool_resources)) = provider
            .deep_learning_forward_route_capability(&pooling)
            .unwrap()
        else {
            panic!("admitted pooling should remain ready")
        };
        assert_eq!(
            pool_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(8 * class.element_size() as u64 + 16),
        );
        let lazy_pool_input = provider.transpose(&resident).unwrap();
        let lazy_pool_route = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Resident(&lazy_pool_input),
            parameters: Vec::new(),
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::Pool1D {
                    input_shape: vec![3, 2],
                    route: runmat_accelerate_api::DeepLearningPool1DRoute {
                        reduction: runmat_accelerate_api::DeepLearningPool1DRequest {
                            axis_zero_based: 1,
                        },
                        result_axis:
                            runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
                        source_shape: vec![3, 1],
                        target_shape: vec![3, 1],
                        target_to_source: vec![0, 1],
                    },
                },
                parameters: DeepLearningForwardParameters::None,
            }],
            output_view: GpuTensorView::contiguous(vec![3, 1]),
        };
        assert!(
            provider
                .deep_learning_forward_route_capability(&lazy_pool_route)
                .is_err(),
            "a lazy transpose is not an admitted pooling stage input"
        );
        let mut materialized_pool = lazy_pool_route.clone();
        materialized_pool.steps.insert(
            0,
            DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::View(ProviderViewStep::Reshape(vec![
                    3, 2,
                ])),
                parameters: DeepLearningForwardParameters::None,
            },
        );
        let Some(DeepLearningForwardCapability::Ready(materialized_pool_resources)) = provider
            .deep_learning_forward_route_capability(&materialized_pool)
            .unwrap()
        else {
            panic!("explicit view materialization makes pooling admissible")
        };
        assert_eq!(
            materialized_pool_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(
                9 * class.element_size() as u64
                    + std::mem::size_of::<crate::backend::wgpu::params::PermuteParams>() as u64
                    + std::mem::size_of::<[u32; 4]>() as u64
            ),
            "the view materialization and pool output/uniform are distinct owned buffers"
        );
        let empty_pool = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: class,
                shape: vec![0, 3],
            },
            parameters: Vec::new(),
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::Pool1D {
                    input_shape: vec![0, 3],
                    route: runmat_accelerate_api::DeepLearningPool1DRoute {
                        reduction: runmat_accelerate_api::DeepLearningPool1DRequest {
                            axis_zero_based: 1,
                        },
                        result_axis:
                            runmat_accelerate_api::DeepLearningPool1DResultAxis::KeepSingleton,
                        source_shape: vec![0, 1],
                        target_shape: vec![0, 1],
                        target_to_source: vec![0, 1],
                    },
                },
                parameters: DeepLearningForwardParameters::None,
            }],
            output_view: GpuTensorView::contiguous(vec![0, 1]),
        };
        let Some(DeepLearningForwardCapability::Ready(empty_pool_resources)) = provider
            .deep_learning_forward_route_capability(&empty_pool)
            .unwrap()
        else {
            panic!("empty non-reduced axis must remain admissible")
        };
        assert_eq!(
            empty_pool_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(2 * class.element_size() as u64),
            "an empty upload and an empty pool output each allocate one physical lane"
        );
        let physical_permute = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: class,
                shape: vec![2, 3],
            },
            parameters: Vec::new(),
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::View(ProviderViewStep::Permute2D),
                parameters: DeepLearningForwardParameters::None,
            }],
            output_view: GpuTensorView::contiguous(vec![3, 2]),
        };
        let Some(DeepLearningForwardCapability::Ready(permutation_resources)) = provider
            .deep_learning_forward_route_capability(&physical_permute)
            .unwrap()
        else {
            panic!("physical permutation should remain ready")
        };
        assert_eq!(
            permutation_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(
                12 * class.element_size() as u64
                    + std::mem::size_of::<crate::backend::wgpu::params::PermuteParams>() as u64
            ),
        );
        let normalization = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: class,
                shape: vec![2, 3],
            },
            parameters: (0..2)
                .map(|_| DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: class,
                        shape: vec![2, 1],
                    },
                    view_steps: Vec::new(),
                })
                .collect(),
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::LayerNorm(
                    runmat_accelerate_api::DeepLearningLayerNormRequest {
                        shape: vec![2, 3],
                        reduce_mask: 1,
                        channel_axis: 0,
                        scale_shape: vec![2, 1],
                        offset_shape: vec![2, 1],
                        epsilon: 1.0e-5,
                    },
                ),
                parameters: DeepLearningForwardParameters::LayerNorm {
                    scale: 0,
                    offset: 1,
                },
            }],
            output_view: GpuTensorView::contiguous(vec![2, 3]),
        };
        let Some(DeepLearningForwardCapability::Ready(norm_resources)) = provider
            .deep_learning_forward_route_capability(&normalization)
            .unwrap()
        else {
            panic!("host-origin normalization should remain admitted")
        };
        assert_eq!(
            norm_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(25 * class.element_size() as u64 + 112),
            "uploads, output, three statistics per group, and two 56-byte uniforms"
        );
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
        assert_eq!(
            shared_resources.new_buffer_bytes_upper_bound,
            DeepLearningForwardBufferBound::Known(
                provider.matmul_buffer_demand(2, 1, 1).unwrap()
                    + provider.binary_broadcast_buffer_demand(2).unwrap()
            ),
            "dense route accounts for product and bias output buffers"
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
