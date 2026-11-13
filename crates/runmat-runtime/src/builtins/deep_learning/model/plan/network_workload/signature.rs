//! Stable semantic portion of a whole-network timing signature.
//!
//! The physical route, provider/device revision, preparation transfers, and
//! observed execution phase are separate inputs to the shared placement key.
//! In particular, an allocation identity is used only to encode equivalence
//! among simultaneously consumed values; its process-local number is never
//! persisted in a timing signature.

use runmat_accelerate_api::{
    GpuHandleProvenance, GpuTensorLayout, GpuTensorStorage, NumericElementType,
};
use runmat_builtins::{
    DeepLearningAxisLabel, DeepLearningLayerKind, DeepLearningNormalizationAxes,
    DeepLearningRecurrentGateActivation, DeepLearningRecurrentOutputMode,
    DeepLearningRecurrentStateActivation,
};
use runmat_execution::identity::Digest;
use runmat_value::NumericDType;

use super::{NetworkWorkload, PayloadFacts, PayloadKind, PayloadOrigin};
use crate::builtins::deep_learning::model::execution::host::conv::{Padding, PaddingValue};
use crate::builtins::deep_learning::model::{
    execution::{NetworkCaller, NetworkOutputLayout},
    parameters::ParameterRole,
    plan::{axes::AxisPlan, host::Controls},
    state::StateRole,
};

/// Invocation facts that affect work, output presentation, or retained tape.
/// `requested_outputs` is supplied by the explicit builtin invocation ABI;
/// it must not be inferred from an ambient session or discarded by a wrapper.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) struct NetworkInvocationFacts {
    pub caller: NetworkCaller,
    pub output_layout: NetworkOutputLayout,
    pub requested_outputs: Option<usize>,
    pub record_tape: bool,
}

struct Encoder {
    bytes: Vec<u8>,
    aliases: Vec<runmat_accelerate_api::allocation::AllocationIdentity>,
}

impl Encoder {
    fn new() -> Self {
        Self {
            bytes: b"runmat.network.semantic-timing.v1\0".to_vec(),
            aliases: Vec::new(),
        }
    }

    fn tag(&mut self, tag: u8) {
        self.bytes.push(tag);
    }

    fn number(&mut self, value: usize) {
        self.bytes.extend_from_slice(
            &u64::try_from(value)
                .expect("addressable size fits u64")
                .to_le_bytes(),
        );
    }

    fn count(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn shape(&mut self, shape: &[usize]) {
        self.number(shape.len());
        for extent in shape {
            self.number(*extent);
        }
    }

    fn axis(&mut self, axis: &AxisPlan) {
        self.shape(axis.shape());
        self.number(axis.layout().axes().len());
        for label in axis.layout().axes() {
            self.tag(match label {
                DeepLearningAxisLabel::Channel => 0,
                DeepLearningAxisLabel::Batch => 1,
                DeepLearningAxisLabel::Time => 2,
                DeepLearningAxisLabel::Spatial => 3,
                DeepLearningAxisLabel::Unspecified => 4,
            });
        }
    }

    fn dtype(&mut self, dtype: NumericDType) {
        self.tag(match dtype {
            NumericDType::F64 => 0,
            NumericDType::F32 => 1,
            NumericDType::I8 => 2,
            NumericDType::I16 => 3,
            NumericDType::I32 => 4,
            NumericDType::I64 => 5,
            NumericDType::U8 => 6,
            NumericDType::U16 => 7,
            NumericDType::U32 => 8,
            NumericDType::U64 => 9,
        });
    }

    fn element_type(&mut self, dtype: NumericElementType) {
        self.tag(match dtype {
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

    fn payload(&mut self, payload: &PayloadFacts) {
        match payload.kind {
            PayloadKind::Real(dtype) => {
                self.tag(0);
                self.dtype(dtype);
            }
            PayloadKind::Complex(dtype) => {
                self.tag(1);
                self.dtype(dtype);
            }
            PayloadKind::Logical => self.tag(2),
        }
        self.shape(&payload.shape);
        self.count(payload.logical_bytes);
        match payload.origin {
            PayloadOrigin::Host => self.tag(0),
            PayloadOrigin::Provider {
                allocation,
                provenance,
                layout,
                element_type,
                storage,
                requested_bytes,
            } => {
                self.tag(1);
                // Canonical first-use numbering describes sharing but is
                // independent of process-local allocation numbers.
                let alias = self
                    .aliases
                    .iter()
                    .position(|identity| *identity == allocation)
                    .unwrap_or_else(|| {
                        self.aliases.push(allocation);
                        self.aliases.len() - 1
                    });
                self.number(alias);
                self.tag(match provenance {
                    None => 0,
                    Some(GpuHandleProvenance::Automatic) => 1,
                    Some(GpuHandleProvenance::Explicit) => 2,
                });
                match layout {
                    GpuTensorLayout::Contiguous => self.tag(0),
                    GpuTensorLayout::Transpose(info) => {
                        self.tag(1);
                        self.number(info.base_rows);
                        self.number(info.base_cols);
                    }
                }
                self.element_type(element_type);
                self.tag(match storage {
                    GpuTensorStorage::Real => 0,
                    GpuTensorStorage::ComplexInterleaved => 1,
                });
                self.count(requested_bytes);
            }
        }
    }

    fn layer(&mut self, kind: DeepLearningLayerKind) {
        self.tag(match kind {
            DeepLearningLayerKind::FeatureInput => 0,
            DeepLearningLayerKind::SequenceInput => 1,
            DeepLearningLayerKind::FullyConnected => 2,
            DeepLearningLayerKind::Convolution1D => 3,
            DeepLearningLayerKind::Lstm => 4,
            DeepLearningLayerKind::Bilstm => 5,
            DeepLearningLayerKind::LayerNormalization => 6,
            DeepLearningLayerKind::GlobalAveragePooling1D => 7,
            DeepLearningLayerKind::Relu => 8,
            DeepLearningLayerKind::Elu => 9,
            DeepLearningLayerKind::Softmax => 10,
            DeepLearningLayerKind::ClassificationOutput => 11,
            DeepLearningLayerKind::RegressionOutput => 12,
        });
    }

    fn controls(&mut self, controls: Controls) {
        match controls {
            Controls::Input { min_length } => {
                self.tag(0);
                self.number(min_length);
            }
            Controls::Dense => self.tag(1),
            Controls::Convolution(options) => {
                self.tag(2);
                self.number(options.stride);
                self.number(options.dilation);
                match options.padding {
                    Padding::Valid => self.tag(0),
                    Padding::Same => self.tag(1),
                    Padding::Causal => self.tag(2),
                    Padding::Explicit { left, right } => {
                        self.tag(3);
                        self.number(left);
                        self.number(right);
                    }
                }
                match options.padding_value {
                    PaddingValue::Constant(value) => {
                        self.tag(0);
                        self.count(value.to_bits());
                    }
                    PaddingValue::Replicate => self.tag(1),
                    PaddingValue::SymmetricIncludeEdge => self.tag(2),
                    PaddingValue::SymmetricExcludeEdge => self.tag(3),
                }
            }
            Controls::Recurrent(options) => {
                self.tag(3);
                self.number(options.hidden_size);
                self.tag(u8::from(options.bidirectional));
                self.tag(match options.output_mode {
                    DeepLearningRecurrentOutputMode::Sequence => 0,
                    DeepLearningRecurrentOutputMode::Last => 1,
                });
                self.tag(match options.state_activation {
                    DeepLearningRecurrentStateActivation::Tanh => 0,
                    DeepLearningRecurrentStateActivation::Softsign => 1,
                    DeepLearningRecurrentStateActivation::Relu => 2,
                });
                self.tag(match options.gate_activation {
                    DeepLearningRecurrentGateActivation::Sigmoid => 0,
                    DeepLearningRecurrentGateActivation::HardSigmoid => 1,
                });
            }
            Controls::Normalization(options) => {
                self.tag(4);
                self.count(options.epsilon.to_bits());
                self.tag(match options.operation {
                    DeepLearningNormalizationAxes::Auto => 0,
                    DeepLearningNormalizationAxes::ChannelOnly => 1,
                    DeepLearningNormalizationAxes::SpatialChannel => 2,
                    DeepLearningNormalizationAxes::BatchExcluded => 3,
                });
            }
            Controls::Pooling => self.tag(5),
            Controls::Relu => self.tag(6),
            Controls::Elu { alpha } => {
                self.tag(7);
                self.count(alpha.to_bits());
            }
            Controls::Softmax => self.tag(8),
            Controls::Output => self.tag(9),
        }
    }
}

impl NetworkWorkload<'_> {
    /// Semantic component only. A shared placement consumer must combine this
    /// with the actual physical route, transfer/presentation plan, candidate
    /// identity and revision before recording or looking up a duration.
    pub fn semantic_timing_digest(&self, invocation: NetworkInvocationFacts) -> Digest {
        let mut encoder = Encoder::new();
        encoder.tag(match invocation.caller {
            NetworkCaller::Forward => 0,
            NetworkCaller::Predict => 1,
        });
        encoder.tag(match invocation.output_layout {
            NetworkOutputLayout::Preserve => 0,
            NetworkOutputLayout::ObservationMajor => 1,
            NetworkOutputLayout::MatchInputRank => 2,
        });
        match invocation.requested_outputs {
            Some(count) => {
                encoder.tag(1);
                encoder.number(count);
            }
            None => encoder.tag(0),
        }
        encoder.tag(u8::from(invocation.record_tape));
        encoder.dtype(self.execution_precision);
        encoder.axis(self.input_axes);
        encoder.axis(self.source_output_axes);
        encoder.axis(self.target_output_axes);
        encoder.count(self.output_elements);
        encoder.count(self.output_bytes_if_real);
        encoder.payload(&self.input);
        encoder.number(self.steps.len());
        for step in self.steps {
            encoder.number(step.index);
            encoder.layer(step.kind);
            encoder.controls(step.controls);
            encoder.axis(&step.input_axes);
            encoder.axis(&step.output_axes);
        }
        encoder.number(self.parameters.len());
        for resource in &self.parameters {
            encoder.number(resource.key.layer_index);
            encoder.tag(match resource.key.role {
                ParameterRole::Weights => 0,
                ParameterRole::InputWeights => 1,
                ParameterRole::RecurrentWeights => 2,
                ParameterRole::Bias => 3,
                ParameterRole::Scale => 4,
                ParameterRole::Offset => 5,
            });
            encoder.payload(&resource.payload);
        }
        encoder.number(self.state.len());
        for resource in &self.state {
            encoder.number(resource.key.layer_index);
            encoder.tag(match resource.key.role {
                StateRole::HiddenState => 0,
                StateRole::CellState => 1,
            });
            encoder.number(resource.key.channels);
            encoder.payload(&resource.payload);
        }
        Digest::sha256(encoder.bytes)
    }
}
