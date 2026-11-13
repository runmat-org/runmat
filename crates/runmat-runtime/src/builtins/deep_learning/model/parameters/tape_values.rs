//! Attach admitted tape values to canonical layer parameters without
//! materializing the derived public Learnables table.

use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;

/// The keys came from `view` before any tape leaf was created. The network is
/// still owned by the caller and has not changed between admission and commit.
pub(in crate::builtins::deep_learning) fn install_annotated(
    network: &mut ObjectInstance,
    values: Vec<(ParameterKey, Value)>,
) -> BuiltinResult<()> {
    let Some(Value::Cell(layers)) = network.property("Layers") else {
        unreachable!("parameter view admitted the canonical Layers cell")
    };
    let context = crate::context::legacy::active();
    let prepared_layers = crate::execution::structure::replacement::cell(layers, context.as_ref())?;
    let _roots = prepared_layers.1;
    let mut layers = prepared_layers.0;
    let layer_values = layers
        .try_data_mut_unique()
        .expect("private Layers destination");
    for (key, value) in values {
        let Some(Value::Object(layer)) = layer_values.get_mut(key.layer_index) else {
            unreachable!("parameter view admitted this layer index")
        };
        layer.replace_dl_property(key.role.name(), value)?;
    }
    network.replace_dl_property("Layers", Value::Cell(layers))?;
    Ok(())
}
