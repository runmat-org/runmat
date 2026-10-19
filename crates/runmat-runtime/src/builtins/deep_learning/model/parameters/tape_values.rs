//! Attach admitted tape values to canonical layer parameters without
//! materializing the derived public Learnables table.

use super::*;

/// The keys came from `view` before any tape leaf was created. The network is
/// still owned by the caller and has not changed between admission and commit.
pub(in crate::builtins::deep_learning) fn install_annotated(
    network: &mut ObjectInstance,
    values: Vec<(ParameterKey, Value)>,
) {
    let Some(Value::Cell(layers)) = network.properties.get_mut("Layers") else {
        unreachable!("parameter view admitted the canonical Layers cell")
    };
    for (key, value) in values {
        let Some(Value::Object(layer)) = layers.data.get_mut(key.layer_index) else {
            unreachable!("parameter view admitted this layer index")
        };
        let slot = layer
            .properties
            .get_mut(key.role.name())
            .expect("parameter view admitted this role");
        *slot = value;
    }
}
