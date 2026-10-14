//! Effect-free geometry validation for a lowered forward route.

use super::*;
use crate::{GpuTensorLayout, ProviderViewChain, ProviderViewPlanError, ProviderViewSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningForwardRouteError {
    InvalidGeometry,
    InvalidParameter,
    SizeOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepLearningForwardGeometry {
    pub output_bytes: u64,
    /// Largest single stage value, including input and output. This is not
    /// a newly allocated intermediate or whole-route peak.
    pub largest_stage_value_bytes: u64,
    /// The verified physical view immediately before each operation.
    pub stage_inputs: Vec<GpuTensorView>,
}

fn elements(shape: &[usize]) -> Result<u64, DeepLearningForwardRouteError> {
    shape.iter().try_fold(1_u64, |count, extent| {
        count
            .checked_mul(
                u64::try_from(*extent).map_err(|_| DeepLearningForwardRouteError::SizeOverflow)?,
            )
            .ok_or(DeepLearningForwardRouteError::SizeOverflow)
    })
}

fn bytes(
    shape: &[usize],
    element: NumericElementType,
) -> Result<u64, DeepLearningForwardRouteError> {
    elements(shape)?
        .checked_mul(element.element_size() as u64)
        .ok_or(DeepLearningForwardRouteError::SizeOverflow)
}

impl DeepLearningForwardSource<'_> {
    pub fn element_type(&self) -> NumericElementType {
        match self {
            Self::Host { element_type, .. } => *element_type,
            Self::Resident(handle) => handle.physical().element_type(),
        }
    }

    pub fn view(&self) -> GpuTensorView {
        match self {
            Self::Host { shape, .. } => GpuTensorView::contiguous(shape.clone()),
            Self::Resident(handle) => handle.view().clone(),
        }
    }

    pub fn logical_bytes(&self) -> Result<u64, DeepLearningForwardRouteError> {
        bytes(self.view().shape(), self.element_type())
    }
}

impl DeepLearningForwardRoute<'_> {
    fn parameter_view(
        &self,
        index: usize,
        element_type: NumericElementType,
    ) -> Result<GpuTensorView, DeepLearningForwardRouteError> {
        let parameter = self
            .parameters
            .get(index)
            .ok_or(DeepLearningForwardRouteError::InvalidParameter)?;
        if parameter.source.element_type() != element_type {
            return Err(DeepLearningForwardRouteError::InvalidParameter);
        }
        let source = parameter.source.view();
        ProviderViewChain {
            source: ProviderViewSource::Current,
            steps: parameter.view_steps.clone(),
        }
        .analyze(&source, element_type)
        .map(|outcome| outcome.view)
        .map_err(|error| match error {
            ProviderViewPlanError::SizeOverflow => DeepLearningForwardRouteError::SizeOverflow,
            _ => DeepLearningForwardRouteError::InvalidParameter,
        })
    }

    /// Validate stage sequence, parameter bindings, and final presentation
    /// without inventing a device handle or interpreting semantic layer names.
    pub fn analyze(&self) -> Result<DeepLearningForwardGeometry, DeepLearningForwardRouteError> {
        let element = self.input.element_type();
        let mut view = self.input.view();
        let mut largest = bytes(view.shape(), element)?;
        let mut stage_inputs = Vec::with_capacity(self.steps.len());
        let mut used_parameters = vec![false; self.parameters.len()];
        for step in &self.steps {
            stage_inputs.push(view.clone());
            match (&step.operation, step.parameters) {
                (
                    DeepLearningForwardOperation::View(operation),
                    DeepLearningForwardParameters::None,
                ) => {
                    let outcome = ProviderViewChain {
                        source: ProviderViewSource::Current,
                        steps: vec![operation.clone()],
                    }
                    .analyze(&view, element)
                    .map_err(|error| match error {
                        ProviderViewPlanError::SizeOverflow => {
                            DeepLearningForwardRouteError::SizeOverflow
                        }
                        _ => DeepLearningForwardRouteError::InvalidGeometry,
                    })?;
                    largest = largest.max(outcome.materialization_bytes);
                    view = outcome.view;
                }
                (
                    DeepLearningForwardOperation::FeedForward(
                        DeepLearningFeedForwardStage::Dense {
                            batch,
                            input_channels,
                            output_channels,
                        },
                    ),
                    DeepLearningForwardParameters::Dense { weights, bias },
                ) => {
                    // Dense execution uses provider matmul, which accepts
                    // the validated lazy 2-D transpose view without a copy.
                    if view.shape() != [*batch, *input_channels]
                        || self.parameter_view(weights, element)?.shape()
                            != [*input_channels, *output_channels]
                        || self.parameter_view(bias, element)?.shape() != [1, *output_channels]
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                    used_parameters[weights] = true;
                    used_parameters[bias] = true;
                    view = GpuTensorView::contiguous(vec![*batch, *output_channels]);
                }
                (
                    DeepLearningForwardOperation::FeedForward(
                        DeepLearningFeedForwardStage::Relu { elements }
                        | DeepLearningFeedForwardStage::Elu { elements },
                    ),
                    DeepLearningForwardParameters::None,
                ) => {
                    if view.layout() != GpuTensorLayout::Contiguous
                        || u64::try_from(*elements).ok() != Some(self::elements(view.shape())?)
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                }
                (
                    DeepLearningForwardOperation::FeedForward(
                        DeepLearningFeedForwardStage::SoftmaxRows { rows, columns },
                    ),
                    DeepLearningForwardParameters::None,
                ) => {
                    if view.layout() != GpuTensorLayout::Contiguous
                        || view.shape() != [*rows, *columns]
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                }
                (
                    DeepLearningForwardOperation::Pool1D { input_shape, route },
                    DeepLearningForwardParameters::None,
                ) => {
                    if view.layout() != GpuTensorLayout::Contiguous
                        || view.shape() != input_shape.as_slice()
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                    let axis = route.reduction.axis_zero_based;
                    if axis >= input_shape.len() || input_shape[axis] == 0 {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                    let mut source = input_shape.clone();
                    source[axis] = 1;
                    if route.result_axis == DeepLearningPool1DResultAxis::RemoveReduced {
                        if source.len() == 1 {
                            return Err(DeepLearningForwardRouteError::InvalidGeometry);
                        }
                        source.remove(axis);
                    }
                    if route.source_shape != source
                        || route.target_shape.len() != source.len()
                        || route.target_to_source.len() != source.len()
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                    let mut seen = vec![false; source.len()];
                    for (position, &source_axis) in route.target_to_source.iter().enumerate() {
                        if source_axis >= source.len()
                            || seen[source_axis]
                            || route.target_shape[position] != source[source_axis]
                        {
                            return Err(DeepLearningForwardRouteError::InvalidGeometry);
                        }
                        seen[source_axis] = true;
                    }
                    view = GpuTensorView::contiguous(route.target_shape.clone());
                }
                (
                    DeepLearningForwardOperation::Conv1D(request),
                    DeepLearningForwardParameters::Conv1D { weights, bias },
                ) => {
                    if view.layout() != GpuTensorLayout::Contiguous
                        || view.shape() != request.input_shape.as_slice()
                        || self.parameter_view(weights, element)?.shape()
                            != request.weights_shape.as_slice()
                        || self.parameter_view(bias, element)?.shape()
                            != request.bias_shape.as_slice()
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                    used_parameters[weights] = true;
                    used_parameters[bias] = true;
                    view = GpuTensorView::contiguous(request.output_shape.clone());
                }
                (
                    DeepLearningForwardOperation::LayerNorm(request),
                    DeepLearningForwardParameters::LayerNorm { scale, offset },
                ) => {
                    if view.layout() != GpuTensorLayout::Contiguous
                        || view.shape() != request.shape.as_slice()
                        || self.parameter_view(scale, element)?.shape()
                            != request.scale_shape.as_slice()
                        || self.parameter_view(offset, element)?.shape()
                            != request.offset_shape.as_slice()
                    {
                        return Err(DeepLearningForwardRouteError::InvalidGeometry);
                    }
                    used_parameters[scale] = true;
                    used_parameters[offset] = true;
                }
                _ => return Err(DeepLearningForwardRouteError::InvalidParameter),
            }
            largest = largest.max(bytes(view.shape(), element)?);
        }
        if used_parameters.iter().any(|used| !used) {
            return Err(DeepLearningForwardRouteError::InvalidParameter);
        }
        if view.shape() != self.output_view.shape()
            || view.layout() != self.output_view.layout()
            || view.is_logical() != self.output_view.is_logical()
        {
            return Err(DeepLearningForwardRouteError::InvalidGeometry);
        }
        Ok(DeepLearningForwardGeometry {
            output_bytes: bytes(view.shape(), element)?,
            largest_stage_value_bytes: largest,
            stage_inputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_support::TestAllocationOwner, GpuTensorStorage};

    fn dense_route<'a>(input: DeepLearningForwardSource<'a>) -> DeepLearningForwardRoute<'a> {
        DeepLearningForwardRoute {
            input,
            parameters: vec![
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: NumericElementType::F32,
                        shape: vec![3, 4],
                    },
                    view_steps: Vec::new(),
                },
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: NumericElementType::F32,
                        shape: vec![1, 4],
                    },
                    view_steps: Vec::new(),
                },
            ],
            steps: vec![
                DeepLearningForwardStep {
                    operation: DeepLearningForwardOperation::FeedForward(
                        DeepLearningFeedForwardStage::Dense {
                            batch: 2,
                            input_channels: 3,
                            output_channels: 4,
                        },
                    ),
                    parameters: DeepLearningForwardParameters::Dense {
                        weights: 0,
                        bias: 1,
                    },
                },
                DeepLearningForwardStep {
                    operation: DeepLearningForwardOperation::FeedForward(
                        DeepLearningFeedForwardStage::Relu { elements: 8 },
                    ),
                    parameters: DeepLearningForwardParameters::None,
                },
            ],
            output_view: GpuTensorView::contiguous(vec![2, 4]),
        }
    }

    #[test]
    fn host_forward_geometry_and_parameter_views_are_checked_without_a_handle() {
        let route = dense_route(DeepLearningForwardSource::Host {
            element_type: NumericElementType::F32,
            shape: vec![2, 3],
        });
        let geometry = route.analyze().unwrap();
        assert_eq!(geometry.output_bytes, 32);
        assert_eq!(geometry.stage_inputs[1].shape(), [2, 4]);

        let mut wrong_parameter = route.clone();
        wrong_parameter.steps[0].parameters = DeepLearningForwardParameters::Dense {
            weights: 0,
            bias: 2,
        };
        assert_eq!(
            wrong_parameter.analyze(),
            Err(DeepLearningForwardRouteError::InvalidParameter)
        );
        let mut unused_parameter = route.clone();
        unused_parameter
            .parameters
            .push(unused_parameter.parameters[1].clone());
        assert_eq!(
            unused_parameter.analyze(),
            Err(DeepLearningForwardRouteError::InvalidParameter)
        );
        let mut wrong_output = route;
        wrong_output.output_view = GpuTensorView::contiguous(vec![99, 1]);
        assert_eq!(
            wrong_output.analyze(),
            Err(DeepLearningForwardRouteError::InvalidGeometry)
        );
    }

    #[test]
    fn convolution_route_checks_exact_physical_parameter_binding_and_output() {
        let request = DeepLearningConv1DRequest {
            input_shape: vec![4, 2, 3],
            output_shape: vec![2, 5, 3],
            weights_shape: [2, 2, 5],
            bias_shape: [1, 5],
            span_axis: 0,
            channel_axis: 1,
            batch_axis: Some(2),
            time_axis: None,
            stride: 2,
            dilation: 2,
            padding_left: 0,
            padding_right: 1,
            padding_value: DeepLearningConv1DPaddingValue::SymmetricIncludeEdge,
        };
        let mut route = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: NumericElementType::F32,
                shape: request.input_shape.clone(),
            },
            parameters: vec![
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: NumericElementType::F32,
                        shape: request.weights_shape.to_vec(),
                    },
                    view_steps: Vec::new(),
                },
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: NumericElementType::F32,
                        shape: request.bias_shape.to_vec(),
                    },
                    view_steps: Vec::new(),
                },
            ],
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::Conv1D(request.clone()),
                parameters: DeepLearningForwardParameters::Conv1D {
                    weights: 0,
                    bias: 1,
                },
            }],
            output_view: GpuTensorView::contiguous(request.output_shape.clone()),
        };
        let geometry = route.analyze().unwrap();
        assert_eq!(geometry.output_bytes, 2 * 5 * 3 * 4);
        assert_eq!(geometry.stage_inputs.len(), 1);
        assert_eq!(geometry.stage_inputs[0].shape(), request.input_shape);

        route.steps[0].parameters = DeepLearningForwardParameters::Conv1D {
            weights: 1,
            bias: 0,
        };
        assert_eq!(
            route.analyze(),
            Err(DeepLearningForwardRouteError::InvalidGeometry)
        );
        route.steps[0].parameters = DeepLearningForwardParameters::Conv1D {
            weights: 0,
            bias: 1,
        };
        route.output_view = GpuTensorView::contiguous(vec![3, 5, 2]);
        assert_eq!(
            route.analyze(),
            Err(DeepLearningForwardRouteError::InvalidGeometry)
        );
    }

    #[test]
    fn resident_route_uses_the_actual_admitted_input_view() {
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let route = dense_route(DeepLearningForwardSource::Resident(&input.handle));
        assert_eq!(route.analyze().unwrap().output_bytes, 32);
        assert_eq!(input.probe.retirements(), 0);
    }

    #[test]
    fn nonsquare_lazy_transpose_is_a_legal_dense_input() {
        let mut route = dense_route(DeepLearningForwardSource::Host {
            element_type: NumericElementType::F32,
            shape: vec![3, 2],
        });
        route.steps.insert(
            0,
            DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::View(ProviderViewStep::Transpose2D),
                parameters: DeepLearningForwardParameters::None,
            },
        );
        let geometry = route.analyze().unwrap();
        assert_eq!(geometry.stage_inputs[1].shape(), [2, 3]);
        assert!(matches!(
            geometry.stage_inputs[1].layout(),
            GpuTensorLayout::Transpose(_)
        ));
        assert_eq!(geometry.output_bytes, 32);
        route.steps[1].operation =
            DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::Dense {
                batch: 3,
                input_channels: 2,
                output_channels: 4,
            });
        assert_eq!(
            route.analyze(),
            Err(DeepLearningForwardRouteError::InvalidGeometry)
        );
    }

    #[test]
    fn normalization_then_activation_consumes_both_affine_parameters() {
        let mut route = DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: NumericElementType::F32,
                shape: vec![2, 3],
            },
            parameters: vec![
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: NumericElementType::F32,
                        shape: vec![3, 1],
                    },
                    view_steps: Vec::new(),
                },
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Host {
                        element_type: NumericElementType::F32,
                        shape: vec![3, 1],
                    },
                    view_steps: Vec::new(),
                },
            ],
            steps: vec![
                DeepLearningForwardStep {
                    operation: DeepLearningForwardOperation::LayerNorm(
                        DeepLearningLayerNormRequest {
                            shape: vec![2, 3],
                            reduce_mask: 0b10,
                            channel_axis: 1,
                            scale_shape: vec![3, 1],
                            offset_shape: vec![3, 1],
                            epsilon: 1.0e-5,
                        },
                    ),
                    parameters: DeepLearningForwardParameters::LayerNorm {
                        scale: 0,
                        offset: 1,
                    },
                },
                DeepLearningForwardStep {
                    operation: DeepLearningForwardOperation::FeedForward(
                        DeepLearningFeedForwardStage::Relu { elements: 6 },
                    ),
                    parameters: DeepLearningForwardParameters::None,
                },
            ],
            output_view: GpuTensorView::contiguous(vec![2, 3]),
        };
        assert_eq!(route.analyze().unwrap().output_bytes, 24);
        route.steps[0].parameters = DeepLearningForwardParameters::LayerNorm {
            scale: 0,
            offset: 2,
        };
        assert_eq!(
            route.analyze(),
            Err(DeepLearningForwardRouteError::InvalidParameter)
        );
    }
}
