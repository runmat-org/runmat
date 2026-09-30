//! Network-owned recurrent State table and dependent public property.
//!
//! `_State` is the single serialized state store. Layer HiddenState/CellState
//! properties describe initial conditions, not a mutable second state store.
use super::*;

mod schema;
mod table;
mod publication;
mod tape_values;

pub(in crate::builtins::deep_learning) use publication::{publish_updates, StateUpdate};
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
        if let Some(value) = layer.properties.get(key.role.name()) {
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

/// A typed, borrowed row from the single canonical serialized State table.
pub(in crate::builtins::deep_learning) struct StateView<'a> {
    pub key: StateKey,
    pub value: &'a Value,
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
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network is missing Layers"),
        ));
    };
    let stored = network.properties.get(STORAGE_PROPERTY).ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: network is missing canonical State"),
        )
    })?;
    table::validate_canonical(&layers.data, stored, function)?;
    let Value::Object(table) = stored else {
        unreachable!("validated State table")
    };
    let values = super::table_rows::values(table, "State", function)?;
    Ok(schema::keys(&layers.data, function)?
        .into_iter()
        .zip(&values.data)
        .map(|(key, value)| StateView { key, value })
        .collect())
}

pub(super) fn all_filled(network: &ObjectInstance, function: &'static str) -> BuiltinResult<bool> {
    let Some(Value::Cell(layers)) = network.properties.get("Layers") else {
        return Err(deep_learning_error(
            function,
            format!("{function}: network is missing Layers"),
        ));
    };
    let stored = network.properties.get(STORAGE_PROPERTY).ok_or_else(|| {
        deep_learning_error(
            function,
            format!("{function}: network is missing canonical State"),
        )
    })?;
    table::all_filled(&layers.data, stored, function)
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
    let Value::Cell(layers) = network.properties.get("Layers").ok_or_else(|| {
        deep_learning_error(function, format!("{function}: network is missing Layers"))
    })?
    else {
        return Err(deep_learning_error(
            function,
            format!("{function}: Layers must be a cell array"),
        ));
    };
    let admitted = table::validated_replacement(&layers.data, replacement, function)?;
    network.properties.insert(STORAGE_PROPERTY.into(), admitted);
    let initialized =
        super::parameters::all_filled(&network, function)? && all_filled(&network, function)?;
    network
        .properties
        .insert("Initialized".into(), Value::Bool(initialized));
    Ok(Value::Object(network))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn recurrent_network() -> ObjectInstance {
        let mut layer = ObjectInstance::new(runmat_builtins::LSTM_LAYER_CLASS.owned());
        layer
            .properties
            .insert("Name".into(), Value::String("memory".into()));
        layer
            .properties
            .insert("NumHiddenUnits".into(), Value::Num(3.0));
        let layers = vec![Value::Object(layer)];
        let state = initial(&layers, "get.State").unwrap();
        let mut network = ObjectInstance::new(DLNETWORK_CLASS.owned());
        network.properties.insert(
            "Layers".into(),
            Value::Cell(CellArray::new(layers, 1, 1).unwrap()),
        );
        network.properties.insert(STORAGE_PROPERTY.into(), state);
        network
    }

    #[test]
    fn network_state_table_is_canonical_and_validated() {
        let network = recurrent_network();
        let rows = view(&network, "forward").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].key.role, StateRole::HiddenState);
        assert_eq!(rows[1].key.role, StateRole::CellState);
        let Value::Object(stored) = network.properties.get(STORAGE_PROPERTY).unwrap() else {
            panic!("stored State table expected");
        };
        let stored_values = super::super::table_rows::values(stored, "State", "forward").unwrap();
        assert!(std::ptr::eq(rows[0].value, &stored_values.data[0]));
        assert!(std::ptr::eq(rows[1].value, &stored_values.data[1]));
        super::super::super::class_registration::ensure_network_class_registered(DLNETWORK_CLASS);
        let public = crate::class_registry::lookup_property(
            &network.class_name,
            &runmat_types::MemberName::from("State"),
        )
        .expect("public State property")
        .0;
        assert!(public.is_dependent);
        let private = crate::class_registry::lookup_property(
            &network.class_name,
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
        assert_eq!(roles.data, ["HiddenState", "CellState"]);
        let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
            panic!("expected values")
        };
        assert_eq!(values.shape, [2, 1]);
        assert!(!network.properties.contains_key("State"));

        let mut variables = variables;
        let Value::Cell(values) = variables.fields.get_mut("Value").unwrap() else {
            panic!("expected values")
        };
        values.data[0] = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
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
            matches!(&values.data[0], Value::Object(object) if object.class_name.is(DLARRAY_CLASS))
        );

        let mut variables = variables;
        let Value::Cell(values) = variables.fields.get_mut("Value").unwrap() else {
            panic!("expected values")
        };
        values.data[1] = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
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
}
