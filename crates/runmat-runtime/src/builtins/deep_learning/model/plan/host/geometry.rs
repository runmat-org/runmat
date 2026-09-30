//! Pure whole-network geometry and control admission. No payload is gathered,
//! projected, initialized, or copied while this plan is built.

mod input;
mod layers;
mod options;
mod resources;

use runmat_builtins::DeepLearningLayerKind as K;
use runmat_value::{NumericDType, ObjectInstance, Value};

use super::controls::{self, Controls};
use crate::builtins::deep_learning::deep_learning_error;
use crate::builtins::deep_learning::model::plan::{axes::AxisPlan, options::LayerOperationalPlan};
use crate::builtins::deep_learning::model::{
    parameters::ParameterView,
    state::StateView,
    topology::{validate_sequential_connections, validate_sequential_topology},
};
use crate::BuiltinResult;
use input::ChannelPolicy;
pub(super) use input::InputPayloadKind;

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
    pub(super) input_payload_kind: InputPayloadKind,
    pub(super) function: &'static str,
}

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
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
        options.require_consumed(function, options::consumed)?;
        let (parameters, state) = resources::views(network, function)?;
        let (shape, precision, format, input_payload_kind) = input::metadata(input, function)?;
        let input_kind = input::kind(first, first_kind, function)?;
        let channel_policy = ChannelPolicy::for_layers(&layers.data);
        let input_axes = input::axes(&format, input_kind, shape, channel_policy, function)?;
        let mut axes = input_axes.clone();
        let mut steps = Vec::with_capacity(layers.data.len());
        for (index, value) in layers.data.iter().enumerate() {
            let Value::Object(layer) = value else {
                unreachable!("admitted layer")
            };
            let kind = K::from_identity(&layer.class_name)
                .ok_or_else(|| fail(function, "unsupported layer"))?;
            if input_payload_kind == InputPayloadKind::Complex
                && !(kind.is_input() || kind == K::GlobalAveragePooling1D || kind.is_output())
            {
                return Err(fail(
                    function,
                    format!("layer {} does not support complex host input", index + 1),
                ));
            }
            let controls = controls::decode(layer, kind, function)?;
            let output_axes = layers::admit(
                layer,
                index,
                controls,
                &axes,
                &parameters,
                &state,
                precision,
                channel_policy,
                function,
            )?;
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
            input_payload_kind,
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
