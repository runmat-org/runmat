//! Stable physical route facts for shared placement timing cohorts.
//!
//! This never reads tensor payloads, backend labels, or ephemeral allocation
//! numbers. Admission still validates live owner leases separately.

use runmat_execution::identity::Digest;

use super::*;
use crate::{
    allocation::AllocationIdentity, GpuHandleProvenance, GpuTensorLayout, GpuTensorStorage,
};

struct Encoder {
    bytes: Vec<u8>,
    aliases: Vec<AllocationIdentity>,
}

impl Encoder {
    fn new() -> Self {
        Self {
            bytes: b"runmat.network.physical-route.v1\0".to_vec(),
            aliases: Vec::new(),
        }
    }

    fn tag(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn count(&mut self, value: usize) {
        self.bytes.extend_from_slice(
            &u64::try_from(value)
                .expect("addressable size fits u64")
                .to_le_bytes(),
        );
    }

    fn number(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn shape(&mut self, shape: &[usize]) {
        self.count(shape.len());
        for extent in shape {
            self.count(*extent);
        }
    }

    fn element(&mut self, element: NumericElementType) {
        self.tag(match element {
            NumericElementType::F64 => 0,
            NumericElementType::F32 => 1,
            NumericElementType::I8 => 2,
            NumericElementType::I16 => 3,
            NumericElementType::I32 => 4,
            NumericElementType::I64 => 5,
            NumericElementType::U8 => 6,
            NumericElementType::U16 => 7,
            NumericElementType::U32 => 8,
            NumericElementType::U64 => 9,
        });
    }

    fn layout(&mut self, layout: GpuTensorLayout) {
        match layout {
            GpuTensorLayout::Contiguous => self.tag(0),
            GpuTensorLayout::Transpose(info) => {
                self.tag(1);
                self.count(info.base_rows);
                self.count(info.base_cols);
            }
        }
    }

    fn view(&mut self, view: &GpuTensorView) {
        self.shape(view.shape());
        self.layout(view.layout());
        self.tag(match view.provenance() {
            None => 0,
            Some(GpuHandleProvenance::Automatic) => 1,
            Some(GpuHandleProvenance::Explicit) => 2,
        });
        self.tag(u8::from(view.is_logical()));
    }

    fn source(&mut self, source: &DeepLearningForwardSource<'_>) {
        match source {
            DeepLearningForwardSource::Host {
                element_type,
                shape,
            } => {
                self.tag(0);
                self.element(*element_type);
                self.shape(shape);
            }
            DeepLearningForwardSource::Resident(handle) => {
                self.tag(1);
                let allocation = handle.allocation_identity();
                let alias = self
                    .aliases
                    .iter()
                    .position(|identity| *identity == allocation)
                    .unwrap_or_else(|| {
                        self.aliases.push(allocation);
                        self.aliases.len() - 1
                    });
                self.count(alias);
                self.view(handle.view());
                let physical = handle.physical();
                self.element(physical.element_type());
                self.tag(match physical.storage() {
                    GpuTensorStorage::Real => 0,
                    GpuTensorStorage::ComplexInterleaved => 1,
                });
                self.number(physical.requested_bytes());
            }
        }
    }

    fn view_step(&mut self, step: &ProviderViewStep) {
        match step {
            ProviderViewStep::Reshape(shape) => {
                self.tag(0);
                self.shape(shape);
            }
            ProviderViewStep::Transpose2D => self.tag(1),
            ProviderViewStep::Permute2D => self.tag(2),
        }
    }

    fn feed_forward(&mut self, stage: DeepLearningFeedForwardStage) {
        match stage {
            DeepLearningFeedForwardStage::Transpose2D { rows, columns } => {
                self.tag(0);
                self.count(rows);
                self.count(columns);
            }
            DeepLearningFeedForwardStage::Permute2D { rows, columns } => {
                self.tag(1);
                self.count(rows);
                self.count(columns);
            }
            DeepLearningFeedForwardStage::Dense {
                batch,
                input_channels,
                output_channels,
            } => {
                self.tag(2);
                self.count(batch);
                self.count(input_channels);
                self.count(output_channels);
            }
            DeepLearningFeedForwardStage::Relu { elements } => {
                self.tag(3);
                self.count(elements);
            }
            DeepLearningFeedForwardStage::Elu { elements } => {
                self.tag(4);
                self.count(elements);
            }
            DeepLearningFeedForwardStage::SoftmaxRows { rows, columns } => {
                self.tag(5);
                self.count(rows);
                self.count(columns);
            }
        }
    }

    fn parameter_binding(&mut self, binding: DeepLearningForwardParameters) {
        match binding {
            DeepLearningForwardParameters::None => self.tag(0),
            DeepLearningForwardParameters::Dense { weights, bias } => {
                self.tag(1);
                self.count(weights);
                self.count(bias);
            }
            DeepLearningForwardParameters::Conv1D { weights, bias } => {
                self.tag(3);
                self.count(weights);
                self.count(bias);
            }
            DeepLearningForwardParameters::LayerNorm { scale, offset } => {
                self.tag(2);
                self.count(scale);
                self.count(offset);
            }
        }
    }

    fn operation(&mut self, operation: &DeepLearningForwardOperation) {
        match operation {
            DeepLearningForwardOperation::View(step) => {
                self.tag(0);
                self.view_step(step);
            }
            DeepLearningForwardOperation::FeedForward(stage) => {
                self.tag(1);
                self.feed_forward(*stage);
            }
            DeepLearningForwardOperation::Pool1D { input_shape, route } => {
                self.tag(2);
                self.shape(input_shape);
                self.count(route.reduction.axis_zero_based);
                self.tag(match route.result_axis {
                    DeepLearningPool1DResultAxis::KeepSingleton => 0,
                    DeepLearningPool1DResultAxis::RemoveReduced => 1,
                });
                self.shape(&route.source_shape);
                self.shape(&route.target_shape);
                self.shape(&route.target_to_source);
            }
            DeepLearningForwardOperation::LayerNorm(request) => {
                self.tag(3);
                self.shape(&request.shape);
                self.tag(request.reduce_mask);
                self.count(request.channel_axis);
                self.shape(&request.scale_shape);
                self.shape(&request.offset_shape);
                self.number(request.epsilon.to_bits());
            }
            DeepLearningForwardOperation::Conv1D(request) => {
                self.tag(4);
                self.shape(&request.input_shape);
                self.shape(&request.output_shape);
                self.shape(&request.weights_shape);
                self.shape(&request.bias_shape);
                self.count(request.span_axis);
                self.count(request.channel_axis);
                for axis in [request.batch_axis, request.time_axis] {
                    match axis {
                        Some(index) => {
                            self.tag(1);
                            self.count(index);
                        }
                        None => self.tag(0),
                    }
                }
                self.count(request.stride);
                self.count(request.dilation);
                self.count(request.padding_left);
                self.count(request.padding_right);
                match request.padding_value {
                    DeepLearningConv1DPaddingValue::Constant(value) => {
                        self.tag(0);
                        self.number(value.to_bits());
                    }
                    DeepLearningConv1DPaddingValue::Replicate => self.tag(1),
                    DeepLearningConv1DPaddingValue::SymmetricIncludeEdge => self.tag(2),
                    DeepLearningConv1DPaddingValue::SymmetricExcludeEdge => self.tag(3),
                }
            }
        }
    }
}

impl DeepLearningForwardRoute<'_> {
    /// Physical-route component only. The shared placement owner must combine
    /// this with admitted semantic invocation facts, transfer/presentation
    /// behavior, provider/device identity, and current revisions. This method
    /// does not imply admission, a duration estimate, or an executable GPU
    /// candidate.
    pub fn timing_route_digest(&self) -> Digest {
        let mut encoder = Encoder::new();
        encoder.source(&self.input);
        encoder.count(self.parameters.len());
        for parameter in &self.parameters {
            encoder.source(&parameter.source);
            encoder.count(parameter.view_steps.len());
            for step in &parameter.view_steps {
                encoder.view_step(step);
            }
        }
        encoder.count(self.steps.len());
        for step in &self.steps {
            encoder.operation(&step.operation);
            encoder.parameter_binding(step.parameters);
        }
        encoder.view(&self.output_view);
        Digest::sha256(encoder.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support;

    fn route(elements: usize) -> DeepLearningForwardRoute<'static> {
        DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Host {
                element_type: NumericElementType::F32,
                shape: vec![elements, 1],
            },
            parameters: vec![],
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::Relu { elements },
                ),
                parameters: DeepLearningForwardParameters::None,
            }],
            output_view: GpuTensorView::contiguous(vec![elements, 1]),
        }
    }

    #[test]
    fn physical_route_digest_uses_typed_geometry_and_stage_order() {
        let baseline = route(8).timing_route_digest();
        assert_eq!(baseline, route(8).timing_route_digest());
        assert_ne!(baseline, route(9).timing_route_digest());
        let mut changed = route(8);
        changed.steps.push(DeepLearningForwardStep {
            operation: DeepLearningForwardOperation::View(ProviderViewStep::Reshape(vec![8, 1])),
            parameters: DeepLearningForwardParameters::None,
        });
        assert!(changed.analyze().is_ok());
        assert_ne!(baseline, changed.timing_route_digest());
    }

    fn dense_route<'a>(
        input: &'a GpuTensorHandle,
        weights: &'a GpuTensorHandle,
        bias: &'a GpuTensorHandle,
    ) -> DeepLearningForwardRoute<'a> {
        DeepLearningForwardRoute {
            input: DeepLearningForwardSource::Resident(input),
            parameters: vec![
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Resident(weights),
                    view_steps: vec![],
                },
                DeepLearningForwardParameter {
                    source: DeepLearningForwardSource::Resident(bias),
                    view_steps: vec![],
                },
            ],
            steps: vec![DeepLearningForwardStep {
                operation: DeepLearningForwardOperation::FeedForward(
                    DeepLearningFeedForwardStage::Dense {
                        batch: 2,
                        input_channels: 2,
                        output_channels: 2,
                    },
                ),
                parameters: DeepLearningForwardParameters::Dense {
                    weights: 0,
                    bias: 1,
                },
            }],
            output_view: GpuTensorView::contiguous(vec![2, 2]),
        }
    }

    #[test]
    fn distinct_views_of_one_live_lease_have_stable_alias_equivalence() {
        let first = test_support::handle(
            41,
            vec![2, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let first_view = first
            .with_view(GpuTensorView::contiguous(vec![2, 2]))
            .unwrap();
        assert_ne!(first.view_identity(), first_view.view_identity());
        let bias = test_support::handle(
            41,
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let aliased = dense_route(&first, &first_view, &bias);
        assert!(aliased.analyze().is_ok());
        let digest = aliased.timing_route_digest();

        let replacement = test_support::handle(
            41,
            vec![2, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let replacement_view = replacement
            .with_view(GpuTensorView::contiguous(vec![2, 2]))
            .unwrap();
        let other_bias = test_support::handle(
            41,
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        assert_eq!(
            digest,
            dense_route(&replacement, &replacement_view, &other_bias).timing_route_digest()
        );
        let distinct_weights = test_support::handle(
            41,
            vec![2, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        assert_ne!(
            digest,
            dense_route(&replacement, &distinct_weights, &other_bias).timing_route_digest()
        );
    }

    #[test]
    fn parameter_binding_and_view_changes_have_distinct_physical_signatures() {
        let input = test_support::handle(
            42,
            vec![2, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let weights = test_support::handle(
            42,
            vec![2, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let bias = test_support::handle(
            42,
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let mut route = dense_route(&input, &weights, &bias);
        let baseline = route.timing_route_digest();
        route.parameters[0].view_steps = vec![ProviderViewStep::Transpose2D];
        assert_ne!(baseline, route.timing_route_digest());
        route.parameters[0].view_steps.clear();
        route.steps[0].parameters = DeepLearningForwardParameters::Dense {
            weights: 1,
            bias: 0,
        };
        assert_ne!(baseline, route.timing_route_digest());
    }
}
