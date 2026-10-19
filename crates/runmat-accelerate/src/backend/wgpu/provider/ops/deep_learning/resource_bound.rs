//! Conservative buffer demand for an admitted physical forward route.
//!
//! This is the checked sum of route-created WGPU buffers, not a measured
//! residency peak or a claim about driver-private allocations. Summing rather
//! than sliding a two-stage window remains safe while pending submissions
//! retain an earlier stage's storage. An operation without an owner-certified
//! allocation formula makes the entire route unavailable to a budget selector.

use super::*;
use runmat_accelerate_api::{
    DeepLearningFeedForwardStage, DeepLearningForwardBufferBound, DeepLearningForwardGeometry,
    DeepLearningForwardOperation, DeepLearningForwardParameters, DeepLearningForwardRoute,
    DeepLearningForwardSource, GpuTensorLayout, ProviderViewChain, ProviderViewSource,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferBoundError {
    Overflow,
}

fn checked_add(total: &mut u64, bytes: u64) -> Result<(), BufferBoundError> {
    *total = total.checked_add(bytes).ok_or(BufferBoundError::Overflow)?;
    Ok(())
}

impl WgpuProvider {
    pub(super) fn forward_buffer_bound(
        &self,
        route: &DeepLearningForwardRoute<'_>,
        geometry: &DeepLearningForwardGeometry,
    ) -> Result<DeepLearningForwardBufferBound, BufferBoundError> {
        let element = route.input.element_type();
        let mut total = 0_u64;
        for source in std::iter::once(&route.input)
            .chain(route.parameters.iter().map(|parameter| &parameter.source))
        {
            if let DeepLearningForwardSource::Host { .. } = source {
                // Empty WGPU storage buffers still allocate one physical lane.
                let bytes = source
                    .logical_bytes()
                    .map_err(|_| BufferBoundError::Overflow)?
                    .max(element.element_size() as u64);
                checked_add(&mut total, bytes)?;
            }
        }

        // Count every materialization, rather than using the view-plan summary
        // that records only the largest single allocation in a chain.
        let mut parameter_views = Vec::with_capacity(route.parameters.len());
        for parameter in &route.parameters {
            let mut view = parameter.source.view();
            for operation in &parameter.view_steps {
                let outcome = ProviderViewChain {
                    source: ProviderViewSource::Current,
                    steps: vec![operation.clone()],
                }
                .analyze(&view, element)
                .map_err(|_| BufferBoundError::Overflow)?;
                if outcome.materialization_bytes != 0 {
                    checked_add(
                        &mut total,
                        self.real_permute_buffer_demand(
                            outcome.materialization_bytes / element.element_size() as u64,
                        )
                        .ok_or(BufferBoundError::Overflow)?,
                    )?;
                }
                view = outcome.view;
            }
            parameter_views.push(view);
        }
        for (step, input) in route.steps.iter().zip(&geometry.stage_inputs) {
            let demand = match &step.operation {
                DeepLearningForwardOperation::View(operation) => {
                    let outcome = ProviderViewChain {
                        source: ProviderViewSource::Current,
                        steps: vec![operation.clone()],
                    }
                    .analyze(input, element)
                    .map_err(|_| BufferBoundError::Overflow)?;
                    if outcome.materialization_bytes == 0 {
                        0
                    } else {
                        let size = element.element_size() as u64;
                        self.real_permute_buffer_demand(outcome.materialization_bytes / size)
                            .ok_or(BufferBoundError::Overflow)?
                    }
                }
                // ReLU uses one fused unary output and one per-chunk uniform;
                // the scalar-op bound has the same aligned buffer geometry.
                DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::Relu {
                    elements,
                }) => self
                    .scalar_op_buffer_demand(*elements as u64)
                    .ok_or(BufferBoundError::Overflow)?,
                DeepLearningForwardOperation::FeedForward(DeepLearningFeedForwardStage::Elu {
                    elements,
                }) => self
                    .elu_buffer_demand(*elements as u64)
                    .ok_or(BufferBoundError::Overflow)?,
                DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::Dense {
                        batch,
                        input_channels,
                        output_channels,
                    },
                ) => {
                    let output = (*batch as u64)
                        .checked_mul(*output_channels as u64)
                        .ok_or(BufferBoundError::Overflow)?;
                    let matmul = self
                        .matmul_buffer_demand(
                            *batch as u64,
                            *output_channels as u64,
                            *input_channels as u64,
                        )
                        .ok_or(BufferBoundError::Overflow)?;
                    let bias = if *batch == 1 {
                        self.binary_same_shape_buffer_demand(output)
                    } else {
                        self.binary_broadcast_buffer_demand(output)
                    }
                    .ok_or(BufferBoundError::Overflow)?;
                    matmul.checked_add(bias).ok_or(BufferBoundError::Overflow)?
                }
                DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::SoftmaxRows { rows, columns },
                ) => self
                    .softmax_rows_buffer_demand(
                        u64::try_from(*rows)
                            .ok()
                            .and_then(|rows| rows.checked_mul(*columns as u64))
                            .ok_or(BufferBoundError::Overflow)?,
                        *rows as u64,
                    )
                    .ok_or(BufferBoundError::Overflow)?,
                DeepLearningForwardOperation::Pool1D { route, .. } => self
                    .pool_1d_buffer_demand(input, route)
                    .ok_or(BufferBoundError::Overflow)?,
                DeepLearningForwardOperation::Conv1D(request) => self
                    .deep_learning_conv1d_forward_buffer_demand(request)
                    .ok_or(BufferBoundError::Overflow)?,
                DeepLearningForwardOperation::LayerNorm(request) => {
                    let DeepLearningForwardParameters::LayerNorm { scale, offset } =
                        step.parameters
                    else {
                        return Ok(DeepLearningForwardBufferBound::Unavailable);
                    };
                    let mut demand = self
                        .layer_norm_buffer_demand(request)
                        .ok_or(BufferBoundError::Overflow)?;
                    for view in [input, &parameter_views[scale], &parameter_views[offset]] {
                        if matches!(view.layout(), GpuTensorLayout::Transpose(_)) {
                            let count = view.shape().iter().try_fold(1_u64, |product, extent| {
                                product.checked_mul(*extent as u64)
                            });
                            let bytes = self
                                .real_permute_buffer_demand(
                                    count.ok_or(BufferBoundError::Overflow)?,
                                )
                                .ok_or(BufferBoundError::Overflow)?;
                            demand = demand
                                .checked_add(bytes)
                                .ok_or(BufferBoundError::Overflow)?;
                        }
                    }
                    demand
                }
                // The admitted route uses View for these layout operations.
                // If that contract changes, their physical allocation must
                // be accounted for by the operation owner first.
                DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::Transpose2D { .. }
                    | DeepLearningFeedForwardStage::Permute2D { .. },
                ) => return Ok(DeepLearningForwardBufferBound::Unavailable),
            };
            checked_add(&mut total, demand)?;
        }
        Ok(DeepLearningForwardBufferBound::Known(total))
    }
}
