//! Dispatch each admitted layer to its owning pure numerical geometry rule.

use runmat_builtins::DeepLearningAxisLabel;
use runmat_value::{NumericDType, ObjectInstance};

use super::resources;
use crate::builtins::deep_learning::model::{
    execution::host::{conv, dense, normalization, pooling, recurrent, softmax, Activation},
    parameters::{ParameterRole, ParameterView},
    plan::axes::AxisPlan,
    state::{StateRole, StateView},
};
use crate::builtins::deep_learning::{deep_learning_error, numeric_vector};
use crate::BuiltinResult;

use super::super::controls::Controls;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

pub(super) fn admit(
    layer: &ObjectInstance,
    index: usize,
    controls: Controls,
    axes: &AxisPlan,
    parameters: &[ParameterView<'_>],
    state: &[StateView<'_>],
    precision: NumericDType,
    function: &'static str,
) -> BuiltinResult<AxisPlan> {
    match controls {
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
            let channels = axes
                .channel_size()
                .map_err(|error| fail(function, format!("invalid channel axis: {error:?}")))?;
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
            Ok(axes.clone())
        }
        Controls::Dense => dense::admit(
            axes,
            resources::parameter_shape(parameters, index, ParameterRole::Weights, function)?,
            resources::parameter_shape(parameters, index, ParameterRole::Bias, function)?,
        )
        .map_err(|error| fail(function, error)),
        Controls::Convolution(options) => conv::admit(
            axes,
            resources::parameter_shape(parameters, index, ParameterRole::Weights, function)?,
            resources::parameter_shape(parameters, index, ParameterRole::Bias, function)?,
            precision,
            options,
        )
        .map_err(|error| fail(function, error)),
        Controls::Recurrent(options) => recurrent::admit(
            axes,
            recurrent::RecurrentParameterShapes {
                input_weights: resources::parameter_shape(
                    parameters,
                    index,
                    ParameterRole::InputWeights,
                    function,
                )?,
                recurrent_weights: resources::parameter_shape(
                    parameters,
                    index,
                    ParameterRole::RecurrentWeights,
                    function,
                )?,
                bias: resources::parameter_shape(parameters, index, ParameterRole::Bias, function)?,
                initial_hidden: resources::state_shape(
                    state,
                    index,
                    StateRole::HiddenState,
                    function,
                )?,
                initial_cell: resources::state_shape(state, index, StateRole::CellState, function)?,
            },
            precision,
            options,
        )
        .map_err(|error| fail(function, error)),
        Controls::Normalization(options) => normalization::admit(
            axes,
            resources::parameter_shape(parameters, index, ParameterRole::Scale, function)?,
            resources::parameter_shape(parameters, index, ParameterRole::Offset, function)?,
            precision,
            options,
        )
        .map_err(|error| fail(function, error)),
        Controls::Pooling => pooling::admit(axes).map_err(|error| fail(function, error)),
        Controls::Relu => {
            Activation::Relu
                .admit(precision)
                .map_err(|error| fail(function, error))?;
            Ok(axes.clone())
        }
        Controls::Elu { alpha } => {
            Activation::Elu { alpha }
                .admit(precision)
                .map_err(|error| fail(function, error))?;
            Ok(axes.clone())
        }
        Controls::Softmax => {
            softmax::admit(axes, precision).map_err(|error| fail(function, error))?;
            Ok(axes.clone())
        }
        Controls::Output => Ok(axes.clone()),
    }
}
