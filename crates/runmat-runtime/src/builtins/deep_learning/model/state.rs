//! Network-owned recurrent State table and dependent public property.
//!
//! `_State` is the single serialized state store. Layer HiddenState/CellState
//! properties describe initial conditions, not a mutable second state store.
use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;

mod schema;
mod table;
mod publication;
mod tape_values;

pub(in crate::builtins::deep_learning) use publication::{
    admit_updates, publish_updates, StateUpdate,
};
pub(in crate::builtins::deep_learning) use schema::{StateKey, StateRole};
pub(in crate::builtins::deep_learning) use tape_values::{
    install_annotated, map_result_values, stage_annotated,
};

pub(super) const STORAGE_PROPERTY: &str = "_State";

#[cfg(test)]
pub(super) use table::initial;
pub(super) use table::initial_with_defaults;

/// Admit initial layer state without projecting/copying its payloads.
pub(in crate::builtins::deep_learning::model) fn admit_initial(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<()> {
    let mut batch = None;
    for key in schema::keys(layers, function)? {
        let Value::Object(layer) = &layers[key.layer_index] else {
            unreachable!("state schema admitted layer objects")
        };
        if let Some(value) = layer.properties().get(key.role.name()) {
            schema::validate_value(value, &key, function)?;
            if let Some(current) = schema::batch_size(value, function)? {
                if batch.is_some_and(|prior| prior != current) {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: initial State rows have inconsistent batch dimensions"
                        ),
                    ));
                }
                batch = Some(current);
            }
        }
    }
    Ok(())
}

/// The canonical State schema supplies recurrent initial-state geometry for
/// training preflight, including the documented unfilled H-by-1 default.
/// Payloads are neither allocated nor gathered at this boundary.
pub(in crate::builtins::deep_learning) fn planned_initial_shapes(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<Vec<(usize, StateRole, Vec<usize>)>> {
    admit_initial(layers, function)?;
    schema::keys(layers, function)?
        .into_iter()
        .map(|key| {
            let Value::Object(layer) = &layers[key.layer_index] else {
                unreachable!("canonical state schema admitted layer objects")
            };
            let shape = match layer.properties().get(key.role.name()) {
                Some(value) => {
                    let shape =
                        parameters::numeric_payload_shape(value, key.role.name(), function)?;
                    if shape.contains(&0) {
                        vec![key.channels, 1]
                    } else {
                        shape.to_vec()
                    }
                }
                None => vec![key.channels, 1],
            };
            Ok((key.layer_index, key.role, shape))
        })
        .collect()
}

/// A typed, borrowed row from the single canonical serialized State table.
pub(in crate::builtins::deep_learning) struct StateView<'a> {
    pub key: StateKey,
    pub value: &'a Value,
}

/// Borrow only explicitly supplied layer initial conditions. The State owner
/// decides which recurrent roles exist (including HasStateInputs), so callers
/// need not reproduce that schema while admitting a training route.
pub(in crate::builtins::deep_learning) fn supplied_initial_values<'a>(
    layers: &'a [Value],
    function: &'static str,
) -> BuiltinResult<Vec<StateView<'a>>> {
    Ok(schema::keys(layers, function)?
        .into_iter()
        .filter_map(|key| {
            let Value::Object(layer) = &layers[key.layer_index] else {
                unreachable!("State schema admitted layer objects")
            };
            layer
                .properties()
                .get(key.role.name())
                .map(|value| StateView { key, value })
        })
        .collect())
}

pub(in crate::builtins::deep_learning) fn view<'a>(
    network: &'a ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Vec<StateView<'a>>> {
    if !is_deep_learning_network_object(network) {
        return Err(deep_learning_error(
            function,
            format!("{function}: expected network object"),
        ));
    }
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network is missing Layers"),
        ));
    };
    let stored = network.properties().get(STORAGE_PROPERTY).ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: network is missing canonical State"),
        )
    })?;
    table::validate_canonical(layers.data(), stored, function)?;
    let Value::Object(table) = stored else {
        unreachable!("validated State table")
    };
    let values = super::table_rows::values(table, "State", function)?;
    Ok(schema::keys(layers.data(), function)?
        .into_iter()
        .zip(values.data())
        .map(|(key, value)| StateView { key, value })
        .collect())
}

pub(super) fn all_filled(network: &ObjectInstance, function: &'static str) -> BuiltinResult<bool> {
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network is missing Layers"),
        ));
    };
    let stored = network.properties().get(STORAGE_PROPERTY).ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: network is missing canonical State"),
        )
    })?;
    table::all_filled(layers.data(), stored, function)
}

/// Fill only missing canonical recurrent State rows after layer learnables have
/// been initialized. Existing State payloads remain the authority, including
/// their batch extent and precision. The replacement is staged and validated
/// as a whole before changing the network.
pub(in crate::builtins::deep_learning) fn initialize_missing(
    network: &mut ObjectInstance,
    function: &'static str,
) -> BuiltinResult<()> {
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network is missing Layers"),
        ));
    };
    let keys = schema::keys(layers.data(), function)?;
    let current = view(network, function)?;
    if current.iter().all(|row| {
        parameters::numeric_payload_shape(row.value, row.key.role.name(), function)
            .is_ok_and(|shape| !shape.contains(&0))
    }) {
        let initialized = parameters::all_filled(network, function)?;
        network.replace_dl_property("Initialized", Value::Bool(initialized))?;
        return Ok(());
    }
    let defaults = table::initial_with_defaults(layers.data(), true, function)?;
    let Value::Object(default_table) = &defaults else {
        unreachable!("State owner constructs a table")
    };
    let default_values = super::table_rows::values(default_table, "State", function)?;
    let mut values = Vec::with_capacity(keys.len());
    for (row, default) in current.iter().zip(default_values.data()) {
        let shape = parameters::numeric_payload_shape(row.value, row.key.role.name(), function)?;
        values.push(if shape.contains(&0) {
            default.clone()
        } else {
            parameters::public_value(Some(row.value))?
        });
    }
    let replacement = table::table(&keys, values, function)?;
    table::validate_canonical(layers.data(), &replacement, function)?;
    let initialized = parameters::all_filled(network, function)?
        && table::all_filled(layers.data(), &replacement, function)?;
    network.replace_dl_property(STORAGE_PROPERTY, replacement)?;
    network.replace_dl_property("Initialized", Value::Bool(initialized))?;
    Ok(())
}

pub(in crate::builtins) fn project(
    network: &ObjectInstance,
    function: &'static str,
) -> BuiltinResult<Value> {
    publish_updates(network, Vec::<StateUpdate>::new(), function)
}

pub(in crate::builtins) fn apply(
    mut network: ObjectInstance,
    replacement: Value,
    function: &'static str,
) -> BuiltinResult<Value> {
    if !is_deep_learning_network_object(&network) {
        return Err(deep_learning_error(
            function,
            format!("{function}: expected network object"),
        ));
    }
    let Value::Cell(layers) = network.properties().get("Layers").ok_or_else(|| {
        deep_learning_error(function, format!("{function}: network is missing Layers"))
    })?
    else {
        return Err(deep_learning_error(
            function,
            format!("{function}: Layers must be a cell array"),
        ));
    };
    let admitted = table::validated_replacement(layers.data(), replacement, function)?;
    network.replace_dl_property(STORAGE_PROPERTY, admitted)?;
    let initialized =
        super::parameters::all_filled(&network, function)? && all_filled(&network, function)?;
    network.replace_dl_property("Initialized", Value::Bool(initialized))?;
    Ok(Value::Object(network))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn recurrent_network() -> ObjectInstance {
        let mut layer = ObjectInstance::new(runmat_builtins::LSTM_LAYER_CLASS.owned());
        layer
            .replace_dl_property("Name", Value::String("memory".into()))
            .expect("fixture property");
        layer
            .replace_dl_property("NumHiddenUnits", Value::Num(3.0))
            .expect("fixture property");
        let layers = vec![Value::Object(layer)];
        let state = initial(&layers, "get.State").unwrap();
        let mut network = ObjectInstance::new(DLNETWORK_CLASS.owned());
        network
            .replace_dl_property("Layers", Value::Cell(CellArray::new(layers, 1, 1).unwrap()))
            .expect("fixture property");
        network
            .replace_dl_property(STORAGE_PROPERTY, state)
            .expect("fixture property");
        network
    }

    #[test]
    fn network_state_table_is_canonical_and_validated() {
        let network = recurrent_network();
        let rows = view(&network, "forward").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].key.role, StateRole::HiddenState);
        assert_eq!(rows[1].key.role, StateRole::CellState);
        let Value::Object(stored) = network.properties().get(STORAGE_PROPERTY).unwrap() else {
            panic!("stored State table expected");
        };
        let stored_values = super::super::table_rows::values(stored, "State", "forward").unwrap();
        assert!(std::ptr::eq(rows[0].value, &stored_values.data()[0]));
        assert!(std::ptr::eq(rows[1].value, &stored_values.data()[1]));
        super::super::super::class_registration::ensure_network_class_registered(DLNETWORK_CLASS);
        let public = crate::class_registry::lookup_property(
            network.class_name(),
            &runmat_types::MemberName::from("State"),
        )
        .expect("public State property")
        .0;
        assert!(public.is_dependent);
        let private = crate::class_registry::lookup_property(
            network.class_name(),
            &runmat_types::MemberName::from(STORAGE_PROPERTY),
        )
        .expect("private state backing")
        .0;
        assert_eq!(private.get_access, runmat_types::MemberAccess::Private);
        assert_eq!(private.set_access, runmat_types::MemberAccess::Private);
        let Value::Object(table) = project(&network, "get.State").unwrap() else {
            panic!("expected State table")
        };
        let variables = crate::builtins::table::table_variables(&table).unwrap();
        let Value::StringArray(roles) = variables.fields.get("Parameter").unwrap() else {
            panic!("expected string roles")
        };
        assert_eq!(roles.data(), ["HiddenState", "CellState"]);
        let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
            panic!("expected values")
        };
        assert_eq!(values.shape, [2, 1]);
        assert!(!network.properties().contains_key("State"));

        let mut variables = variables;
        let Value::Cell(values) =
            crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value")
                .unwrap()
        else {
            panic!("expected values")
        };
        crate::builtins::deep_learning::tests::fixture_array_mut(values)[0] =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let updated =
            crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
        let Value::Object(network) = apply(network, updated, "set.State").unwrap() else {
            panic!("expected network")
        };
        let Value::Object(projected) = project(&network, "get.State").unwrap() else {
            panic!("expected table")
        };
        let variables = crate::builtins::table::table_variables(&projected).unwrap();
        let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
            panic!("expected values")
        };
        assert!(
            matches!(&values.data()[0], Value::Object(object) if object.class_name().is(DLARRAY_CLASS))
        );

        let mut variables = variables;
        let Value::Cell(values) =
            crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value")
                .unwrap()
        else {
            panic!("expected values")
        };
        crate::builtins::deep_learning::tests::fixture_array_mut(values)[1] =
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let invalid =
            crate::builtins::table::table_replace_variables_like(&projected, variables).unwrap();
        assert!(apply(network.clone(), invalid, "set.State").is_err());
        assert!(project(&network, "get.State").is_ok());
    }

    #[test]
    fn stateless_network_has_an_empty_state_table() {
        let state = initial(&[], "dlnetwork").unwrap();
        let Value::Object(table) = state else {
            panic!("expected table")
        };
        let variables = crate::builtins::table::table_variables(&table).unwrap();
        let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
            panic!("expected values")
        };
        assert_eq!(values.shape, [0, 1]);
    }

    #[test]
    fn missing_state_initialization_preserves_filled_canonical_rows() {
        let mut network = recurrent_network();
        let Value::Cell(layers) = network.properties().get("Layers").unwrap() else {
            panic!("Layers expected")
        };
        let keys = schema::keys(layers.data(), "trainnet").unwrap();
        let supplied = Value::Tensor(Tensor::from_f32(vec![4., 5., 6.], vec![3, 1]).unwrap());
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let partial = table::table(&keys, vec![supplied.clone(), empty], "trainnet").unwrap();
        network
            .replace_dl_property(STORAGE_PROPERTY, partial)
            .expect("fixture property");

        initialize_missing(&mut network, "trainnet").unwrap();
        let rows = view(&network, "trainnet").unwrap();
        let supplied_data = match rows[0].value {
            Value::Object(wrapper) if wrapper.class_name().is(DLARRAY_CLASS) => {
                wrapper.properties().get("Data").unwrap()
            }
            value => value,
        };
        assert_eq!(supplied_data, &supplied);
        let default_data = match rows[1].value {
            Value::Object(wrapper) if wrapper.class_name().is(DLARRAY_CLASS) => {
                wrapper.properties().get("Data").unwrap()
            }
            value => value,
        };
        let Value::Tensor(default_data) = default_data else {
            panic!("default tensor expected")
        };
        assert_eq!(default_data.shape(), &[3, 1]);
        assert_eq!(default_data.as_f64_slice(), Some(&[0., 0., 0.][..]));
        // Learnables are still unfilled, so filling State must not claim the
        // entire network has been initialized.
        assert_eq!(
            network.properties().get("Initialized"),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn malformed_partial_state_does_not_mutate_network() {
        let mut network = recurrent_network();
        let Value::Object(stored) = network.properties().get(STORAGE_PROPERTY).unwrap() else {
            panic!("State table expected")
        };
        let mut variables = crate::builtins::table::table_variables(stored).unwrap();
        let Value::Cell(values) =
            crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value")
                .unwrap()
        else {
            panic!("State values expected")
        };
        crate::builtins::deep_learning::tests::fixture_array_mut(values)[0] =
            Value::Tensor(Tensor::new(vec![1., 2.], vec![2, 1]).unwrap());
        let invalid =
            crate::builtins::table::table_replace_variables_like(stored, variables).unwrap();
        network
            .replace_dl_property(STORAGE_PROPERTY, invalid)
            .expect("fixture property");
        let original = network.clone();
        assert!(initialize_missing(&mut network, "trainnet").is_err());
        assert_eq!(network, original);
    }
}
