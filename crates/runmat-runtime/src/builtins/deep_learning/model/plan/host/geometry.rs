//! Pure whole-network geometry and control admission. No payload is gathered,
//! projected, initialized, or copied while this plan is built.

use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{
    DeepLearningAxisLabel, DeepLearningInputKind, DeepLearningLayerKind as K,
    DeepLearningLayerOption as O,
};
use runmat_value::{NumericDType, ObjectInstance, Value};

use super::controls::{self, Controls};
use crate::builtins::deep_learning::model::plan::{
    axes::AxisPlan,
    options::{LayerOperationalPlan, LayerOptionRequirement},
};
use crate::builtins::deep_learning::model::{
    admission::forward_data,
    execution::host::{conv, dense, normalization, pooling, recurrent, softmax},
    parameters::{self, ParameterRole, ParameterView},
    state::{self, StateRole, StateView},
    topology::{validate_sequential_connections, validate_sequential_topology},
};
use crate::builtins::deep_learning::{deep_learning_error, numeric_vector, DLARRAY_CLASS};
use crate::BuiltinResult;

pub(super) struct StepDescriptor {
    pub index: usize,
    pub kind: K,
    pub controls: Controls,
}

pub(in crate::builtins::deep_learning) struct HostNetworkPlan<'a> {
    pub(super) parameters: Vec<ParameterView<'a>>,
    pub(super) state: Vec<StateView<'a>>,
    pub(super) steps: Vec<StepDescriptor>,
    pub(super) input_axes: AxisPlan,
    pub(super) output_axes: AxisPlan,
    pub(super) precision: NumericDType,
    pub(super) function: &'static str,
}

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

fn input_metadata<'a>(
    input: &'a Value,
    function: &'static str,
) -> BuiltinResult<(&'a [usize], NumericDType, String)> {
    let format = match input {
        Value::Object(object) if object.class_name.is(DLARRAY_CLASS) => {
            match object.properties.get("Format") {
                Some(Value::String(value)) => value.clone(),
                None => String::new(),
                _ => return Err(fail(function, "dlarray Format must be text")),
            }
        }
        _ => String::new(),
    };
    let data = forward_data(input);
    let (shape, dtype) = match data {
        Value::Tensor(tensor) => (
            tensor.shape.as_slice(),
            if tensor.numeric_dtype().integer_class().is_some() {
                // Existing RunMat integer-forward extension computes at a
                // declared floating numerical boundary, never by altering
                // the authoritative source tensor.
                NumericDType::F64
            } else {
                tensor.numeric_dtype()
            },
        ),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|error| fail(function, error))?;
            if crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle).is_none() {
                return Err(fail(function, "input has no live owning provider"));
            }
            let dtype = match runmat_accelerate_api::handle_precision(handle) {
                Some(ProviderPrecision::F32) => NumericDType::F32,
                Some(ProviderPrecision::F64) => NumericDType::F64,
                None => return Err(fail(function, "input has unsupported resident precision")),
            };
            (handle.shape(), dtype)
        }
        Value::LogicalArray(array) => (array.shape.as_slice(), NumericDType::F64),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (&[1usize, 1][..], NumericDType::F64),
        _ => return Err(fail(function, "input must contain real numeric data")),
    };
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(fail(
            function,
            "host network input requires single or double precision",
        ));
    }
    Ok((shape, dtype, format))
}

fn parameter_shape<'a>(
    rows: &[ParameterView<'a>],
    index: usize,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<&'a [usize]> {
    let row = rows
        .iter()
        .find(|row| row.key.layer_index == index && row.key.role == role)
        .ok_or_else(|| {
            fail(
                function,
                format!("layer {} is missing {}", index + 1, role.name()),
            )
        })?;
    let value = row.value.ok_or_else(|| {
        fail(
            function,
            format!("layer {} has uninitialized {}", index + 1, role.name()),
        )
    })?;
    let shape = parameters::numeric_payload_shape(value, role.name(), function)?;
    if shape.contains(&0) {
        return Err(fail(
            function,
            format!("layer {} has uninitialized {}", index + 1, role.name()),
        ));
    }
    Ok(shape)
}

fn state_shape<'a>(
    rows: &[StateView<'a>],
    index: usize,
    role: StateRole,
    function: &'static str,
) -> BuiltinResult<Option<&'a [usize]>> {
    rows.iter()
        .find(|row| row.key.layer_index == index && row.key.role == role)
        .map(|row| {
            parameters::numeric_payload_shape(row.value, role.name(), function)
                .map(|shape| (!shape.contains(&0)).then_some(shape))
        })
        .transpose()
        .map(Option::flatten)
}

fn consumed(requirement: &LayerOptionRequirement<'_>) -> bool {
    use O::*;
    match requirement.option {
        Name | Description => true,
        // These options affect construction or training, not forward execution
        // of an already initialized canonical network.
        WeightsInitializer
        | BiasInitializer
        | InputWeightsInitializer
        | RecurrentWeightsInitializer
        | ScaleInitializer
        | OffsetInitializer
        | WeightLearnRateFactor
        | WeightL2Factor
        | BiasLearnRateFactor
        | BiasL2Factor
        | InputWeightsLearnRateFactor
        | RecurrentWeightsLearnRateFactor
        | InputWeightsL2Factor
        | RecurrentWeightsL2Factor
        | ScaleLearnRateFactor
        | OffsetLearnRateFactor
        | ScaleL2Factor
        | OffsetL2Factor => true,
        Weights
        | Bias
        | InputWeights
        | RecurrentWeights
        | Scale
        | Offset
        | HiddenState
        | CellState
        | InputSize
        | NumChannels
        | Stride
        | DilationFactor
        | Padding
        | PaddingValue
        | MinLength
        | OutputMode
        | StateActivationFunction
        | GateActivationFunction
        | Epsilon => true,
        OperationDimension => matches!(requirement.kind, K::LayerNormalization),
        // Output metadata is not a numerical forward transform.
        Classes | ClassWeights | ResponseNames => requirement.kind.is_output(),
        // Input transforms, callable controls, state I/O, and any future option
        // need an owning consumer before they may be admitted here.
        _ => false,
    }
}

impl<'a> HostNetworkPlan<'a> {
    pub(in crate::builtins::deep_learning) fn admit(
        network: &'a ObjectInstance,
        input: &Value,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        validate_sequential_connections(network, function)?;
        let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
            return Err(fail(function, "network is missing Layers"));
        };
        validate_sequential_topology(&layers.data, function)?;
        let Some(Value::Object(first)) = layers.data.first() else {
            return Err(fail(function, "network must start with an input layer"));
        };
        let first_kind = K::from_identity(&first.class_name)
            .ok_or_else(|| fail(function, "unsupported first layer"))?;
        if !first_kind.is_input() {
            return Err(fail(function, "network must start with an input layer"));
        }
        if network.properties.get("Initialized") != Some(&Value::Bool(true)) {
            return Err(fail(
                function,
                "network parameters and state must be initialized before forward execution",
            ));
        }
        let options = LayerOperationalPlan::inspect(&layers.data, function)?;
        options.require_consumed(function, consumed)?;
        let parameters = parameters::view(network, function)?;
        let state = state::view(network, function)?;
        for row in &parameters {
            let value = row
                .value
                .ok_or_else(|| fail(function, "network has uninitialized parameters"))?;
            if parameters::numeric_payload_shape(value, row.key.role.name(), function)?.contains(&0)
            {
                return Err(fail(function, "network has uninitialized parameters"));
            }
        }
        for row in &state {
            if parameters::numeric_payload_shape(row.value, row.key.role.name(), function)?
                .contains(&0)
            {
                return Err(fail(function, "network has uninitialized state"));
            }
        }
        let (shape, precision, format) = input_metadata(input, function)?;
        let input_kind = match first_kind {
            K::FeatureInput => {
                let value = first
                    .properties
                    .get("InputSize")
                    .ok_or_else(|| fail(function, "featureInputLayer is missing InputSize"))?;
                if numeric_vector(value, function, "InputSize")?.len() == 1 {
                    DeepLearningInputKind::Feature
                } else {
                    DeepLearningInputKind::Spatial1d
                }
            }
            K::SequenceInput => {
                let value = first
                    .properties
                    .get("InputSize")
                    .ok_or_else(|| fail(function, "sequenceInputLayer is missing InputSize"))?;
                if numeric_vector(value, function, "InputSize")?.len() == 1 {
                    DeepLearningInputKind::VectorSequence
                } else {
                    DeepLearningInputKind::SpatialSequence1d
                }
            }
            _ => unreachable!(),
        };
        let input_axes = AxisPlan::for_input(&format, input_kind, shape)
            .map_err(|error| fail(function, format!("invalid input axes: {error:?}")))?;
        let mut axes = input_axes.clone();
        let mut steps = Vec::with_capacity(layers.data.len());
        for (index, value) in layers.data.iter().enumerate() {
            let Value::Object(layer) = value else {
                unreachable!("admitted layer")
            };
            let kind = K::from_identity(&layer.class_name)
                .ok_or_else(|| fail(function, "unsupported layer"))?;
            let controls = controls::decode(layer, kind, function)?;
            let output_axes = match controls {
                Controls::Input { min_length } => {
                    if index != 0 {
                        return Err(fail(function, "input layer must be first"));
                    }
                    let expected = numeric_vector(
                        layer
                            .properties
                            .get("InputSize")
                            .ok_or_else(|| fail(function, "input layer is missing InputSize"))?,
                        function,
                        "InputSize",
                    )?;
                    if expected.len() != 1 && expected.len() != 2 {
                        return Err(fail(function, "input size must have one or two dimensions"));
                    }
                    let channels = axes.channel_size().map_err(|error| {
                        fail(function, format!("invalid channel axis: {error:?}"))
                    })?;
                    if channels != *expected.last().expect("nonempty input size") {
                        return Err(fail(
                            function,
                            "input channel extent differs from InputSize",
                        ));
                    }
                    if let Some(spatial) = axes.axis_size(DeepLearningAxisLabel::Spatial) {
                        if expected.len() != 2 || spatial != expected[0] {
                            return Err(fail(
                                function,
                                "input spatial extent differs from InputSize",
                            ));
                        }
                    }
                    if let Some(time) = axes.axis_size(DeepLearningAxisLabel::Time) {
                        if time < min_length {
                            return Err(fail(function, "sequence is shorter than MinLength"));
                        }
                    }
                    axes.clone()
                }
                Controls::Dense => dense::admit(
                    &axes,
                    parameter_shape(&parameters, index, ParameterRole::Weights, function)?,
                    parameter_shape(&parameters, index, ParameterRole::Bias, function)?,
                )
                .map_err(|error| fail(function, error))?,
                Controls::Convolution(options) => conv::admit(
                    &axes,
                    parameter_shape(&parameters, index, ParameterRole::Weights, function)?,
                    parameter_shape(&parameters, index, ParameterRole::Bias, function)?,
                    precision,
                    options,
                )
                .map_err(|error| fail(function, error))?,
                Controls::Recurrent(options) => recurrent::admit(
                    &axes,
                    recurrent::RecurrentParameterShapes {
                        input_weights: parameter_shape(
                            &parameters,
                            index,
                            ParameterRole::InputWeights,
                            function,
                        )?,
                        recurrent_weights: parameter_shape(
                            &parameters,
                            index,
                            ParameterRole::RecurrentWeights,
                            function,
                        )?,
                        bias: parameter_shape(&parameters, index, ParameterRole::Bias, function)?,
                        initial_hidden: state_shape(
                            &state,
                            index,
                            StateRole::HiddenState,
                            function,
                        )?,
                        initial_cell: state_shape(&state, index, StateRole::CellState, function)?,
                    },
                    precision,
                    options,
                )
                .map_err(|error| fail(function, error))?,
                Controls::Normalization(options) => normalization::admit(
                    &axes,
                    parameter_shape(&parameters, index, ParameterRole::Scale, function)?,
                    parameter_shape(&parameters, index, ParameterRole::Offset, function)?,
                    precision,
                    options,
                )
                .map_err(|error| fail(function, error))?,
                Controls::Pooling => {
                    pooling::admit(&axes).map_err(|error| fail(function, error))?
                }
                Controls::Relu => {
                    super::super::super::execution::host::Activation::Relu
                        .admit(precision)
                        .map_err(|error| fail(function, error))?;
                    axes.clone()
                }
                Controls::Elu { alpha } => {
                    super::super::super::execution::host::Activation::Elu { alpha }
                        .admit(precision)
                        .map_err(|error| fail(function, error))?;
                    axes.clone()
                }
                Controls::Softmax => {
                    softmax::admit(&axes, precision).map_err(|error| fail(function, error))?;
                    axes.clone()
                }
                Controls::Output => axes.clone(),
            };
            steps.push(StepDescriptor {
                index,
                kind,
                controls,
            });
            axes = output_axes;
        }
        Ok(Self {
            parameters,
            state,
            steps,
            input_axes,
            output_axes: axes,
            precision,
            function,
        })
    }
    pub(in crate::builtins::deep_learning) fn input_axes(&self) -> &AxisPlan {
        &self.input_axes
    }

    pub(in crate::builtins::deep_learning) fn has_integer_parameter(&self) -> bool {
        self.parameters
            .iter()
            .filter_map(|row| row.value)
            .any(|value| crate::builtins::common::validation::value_has_native_integer_class(value))
    }
}
