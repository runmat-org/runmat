//! Borrowed sequential topology admission before initialization or execution.
use super::*;

#[cfg(test)]
mod tests;

pub(in crate::builtins) fn validate_sequential_connections(
    network: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<()> {
    match network.properties().get("Connections") {
        None => Ok(()),
        Some(Value::Struct(connections)) if connections.fields.is_empty() => Ok(()),
        Some(_) => Err(unsupported_error(
            function,
            format!("{function}: nonempty or unsupported Connections cannot be executed as a sequential network"),
        )),
    }
}

pub(in crate::builtins) fn validate_forward_layers(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<()> {
    for layer in layers {
        if let Value::Object(object) = layer {
            if ForwardLayerKind::from_identity(object.class_name()).is_none() {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: layer type '{}' is not supported for RunMat forward execution",
                        object.class_name()
                    ),
                ));
            }
        }
    }
    validate_sequential_topology(layers, function)
}

/// Check shared sequence ordering without selecting a consumer's supported
/// layer subset. Each consumer retains its own unsupported-layer diagnostic.
pub(in crate::builtins) fn validate_sequential_topology(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<()> {
    if layers.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: network must contain at least one layer"),
        ));
    }
    for (index, layer) in layers.iter().enumerate() {
        let Value::Object(object) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network layers must be layer objects"),
            ));
        };
        let Some(kind) = ForwardLayerKind::from_identity(object.class_name()) else {
            continue;
        };
        if index != 0 && kind.is_input() {
            return Err(deep_learning_error(
                function,
                format!("{function}: an input layer must be the first and only input layer"),
            ));
        }
        if kind.is_output() && index + 1 != layers.len() {
            return Err(deep_learning_error(
                function,
                format!("{function}: output layers must be terminal layers"),
            ));
        }
        if kind == ForwardLayerKind::Elu {
            elu_alpha(object, function)?;
        }
    }
    Ok(())
}
