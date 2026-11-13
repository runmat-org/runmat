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
    is_deep_learning_network_object, layers_from_network_ref,
    parameters::ParameterView,
    state::StateView,
    topology::{validate_sequential_connections, validate_sequential_topology},
};
use crate::BuiltinResult;
use input::ChannelPolicy;
pub(super) use input::InputPayloadKind;

pub(in crate::builtins::deep_learning) struct StepDescriptor {
    pub index: usize,
    pub kind: K,
    pub controls: Controls,
    /// Geometry immediately before and after this admitted layer. Device
    /// lowering consumes these facts instead of reconstructing axes from a
    /// separate layer-specific shape solver.
    pub input_axes: AxisPlan,
    pub output_axes: AxisPlan,
}

pub(in crate::builtins::deep_learning) struct HostNetworkPlan<'a> {
    network: &'a ObjectInstance,
    pub(super) input: &'a Value,
    pub(super) parameters: Vec<ParameterView<'a>>,
    pub(super) state: Vec<StateView<'a>>,
    pub(super) steps: Vec<StepDescriptor>,
    pub(super) input_axes: AxisPlan,
    pub(super) output_axes: AxisPlan,
    pub(super) precision: NumericDType,
    pub(super) input_payload_kind: InputPayloadKind,
    pub(super) function: &'static str,
}

/// Pure training descriptor. It intentionally owns only shape/control facts;
/// the initialized network is built after target/loss admission and is then
/// checked again by `HostNetworkPlan::admit` against its filled payloads.
pub(in crate::builtins::deep_learning) struct TrainingGeometry {
    pub(in crate::builtins::deep_learning) input_axes: AxisPlan,
    pub(in crate::builtins::deep_learning) output_axes: AxisPlan,
    pub(in crate::builtins::deep_learning) precision: NumericDType,
}

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

impl<'a> HostNetworkPlan<'a> {
    pub(in crate::builtins::deep_learning) fn steps(&self) -> &[StepDescriptor] {
        &self.steps
    }

    pub(in crate::builtins::deep_learning) fn parameter_views(&self) -> &[ParameterView<'a>] {
        &self.parameters
    }

    pub(in crate::builtins::deep_learning) fn state_views(&self) -> &[StateView<'a>] {
        &self.state
    }

    pub(in crate::builtins::deep_learning) fn function(&self) -> &'static str {
        self.function
    }

    pub(in crate::builtins::deep_learning) fn execution_precision(&self) -> NumericDType {
        self.precision
    }

    pub(in crate::builtins::deep_learning) fn admitted_input(&self) -> &'a Value {
        self.input
    }

    pub(in crate::builtins::deep_learning) fn admitted_network(&self) -> &'a ObjectInstance {
        self.network
    }

    /// A bounded first resident route. The same admitted geometry is used by
    /// host execution; future composed device stages consume these steps.
    pub(in crate::builtins::deep_learning) fn standalone_normalization(
        &self,
    ) -> Result<
        Option<crate::builtins::deep_learning::model::plan::normalization::NormalizationStage>,
        crate::builtins::deep_learning::model::plan::normalization::NormalizationError,
    > {
        use crate::builtins::deep_learning::model::plan::normalization::{
            Geometry, NormalizationStage,
        };
        let [input, normalization, remainder @ ..] = self.steps.as_slice() else {
            return Ok(None);
        };
        if !input.kind.is_input()
            || normalization.kind != K::LayerNormalization
            || !(remainder.is_empty() || remainder.len() == 1 && remainder[0].kind.is_output())
        {
            return Ok(None);
        }
        let Controls::Normalization(options) = normalization.controls else {
            return Ok(None);
        };
        debug_assert_eq!(normalization.input_axes, normalization.output_axes);
        Geometry::new(&normalization.input_axes, options).map(|geometry| {
            Some(NormalizationStage {
                layer_index: normalization.index,
                geometry,
            })
        })
    }

    pub(in crate::builtins::deep_learning) fn admit(
        network: &'a ObjectInstance,
        input: &'a Value,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        validate_sequential_connections(network, function)?;
        let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
            return Err(fail(function, "network is missing Layers"));
        };
        validate_sequential_topology(layers.data(), function)?;
        let Some(Value::Object(first)) = layers.data().first() else {
            return Err(fail(function, "network must start with an input layer"));
        };
        let first_kind = K::from_identity(first.class_name())
            .ok_or_else(|| fail(function, "unsupported first layer"))?;
        if !first_kind.is_input() {
            return Err(fail(function, "network must start with an input layer"));
        }
        if network.properties().get("Initialized") != Some(&Value::Bool(true)) {
            return Err(fail(
                function,
                "network parameters and state must be initialized before forward execution",
            ));
        }
        let (parameters, state) = resources::views(network, function)?;
        let shapes = resources::ResourceShapes::filled(&parameters, &state, function)?;
        let (input_axes, axes, precision, input_payload_kind, steps) =
            admit_layers(layers.data(), first, first_kind, input, &shapes, function)?;
        Ok(Self {
            network,
            input,
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

pub(in crate::builtins::deep_learning) fn admit_training_geometry(
    source: &Value,
    input: &Value,
    function: &'static str,
) -> BuiltinResult<TrainingGeometry> {
    let layers = layers_from_network_ref(source, function)?;
    validate_sequential_topology(layers, function)?;
    let Some(Value::Object(first)) = layers.first() else {
        return Err(fail(function, "network must start with an input layer"));
    };
    let first_kind = K::from_identity(first.class_name())
        .ok_or_else(|| fail(function, "unsupported first layer"))?;
    if !first_kind.is_input() {
        return Err(fail(function, "network must start with an input layer"));
    }
    let canonical_network = match source {
        Value::Object(object) if is_deep_learning_network_object(object) => Some(object),
        _ => None,
    };
    let shapes =
        resources::ResourceShapes::before_initialization(layers, canonical_network, function)?;
    let (input_axes, output_axes, precision, _, _) =
        admit_layers(layers, first, first_kind, input, &shapes, function)?;
    Ok(TrainingGeometry {
        input_axes,
        output_axes,
        precision,
    })
}

fn admit_layers(
    layers: &[Value],
    first: &ObjectInstance,
    first_kind: K,
    input: &Value,
    shapes: &resources::ResourceShapes,
    function: &'static str,
) -> BuiltinResult<(
    AxisPlan,
    AxisPlan,
    NumericDType,
    InputPayloadKind,
    Vec<StepDescriptor>,
)> {
    let options = LayerOperationalPlan::inspect(layers, function)?;
    options.require_consumed(function, options::consumed)?;
    let (shape, precision, format, input_payload_kind) = input::metadata(input, function)?;
    let input_kind = input::kind(first, first_kind, function)?;
    let channel_policy = ChannelPolicy::for_layers(layers);
    let input_axes = input::axes(&format, input_kind, shape, channel_policy, function)?;
    let mut axes = input_axes.clone();
    let mut steps = Vec::with_capacity(layers.len());
    for (index, value) in layers.iter().enumerate() {
        let Value::Object(layer) = value else {
            unreachable!("admitted layer")
        };
        let kind = K::from_identity(layer.class_name())
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
            shapes,
            precision,
            channel_policy,
            function,
        )?;
        steps.push(StepDescriptor {
            index,
            kind,
            controls,
            input_axes: axes,
            output_axes: output_axes.clone(),
        });
        axes = output_axes;
    }
    Ok((input_axes, axes, precision, input_payload_kind, steps))
}
