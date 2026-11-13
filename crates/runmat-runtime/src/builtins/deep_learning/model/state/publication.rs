//! Validated execution State results without mutating the input network.
use super::*;
use std::collections::HashMap;

pub(in crate::builtins::deep_learning) struct StateUpdate {
    pub layer_index: usize,
    pub role: StateRole,
    pub value: Tensor,
}

/// Check every returned State row before a network execution is recorded on
/// the autodiff tape. Publication remains the only owner of the output table.
pub(in crate::builtins::deep_learning) fn admit_updates(
    network: &ObjectInstance,
    updates: &[StateUpdate],
    function: &'static str,
) -> BuiltinResult<()> {
    let rows = view(network, function)?;
    let mut supplied = HashMap::with_capacity(updates.len());
    for update in updates {
        if supplied
            .insert((update.layer_index, update.role), &update.value)
            .is_some()
        {
            return Err(deep_learning_error(
                function,
                format!("{function}: duplicate execution State update"),
            ));
        }
    }
    let mut batch = None;
    for row in rows {
        let update = supplied.remove(&(row.key.layer_index, row.key.role));
        let candidate = update.map(|tensor| Value::Tensor(tensor.clone()));
        let value = candidate.as_ref().unwrap_or(row.value);
        schema::validate_value(value, &row.key, function)?;
        if let Some(current) = schema::batch_size(value, function)? {
            if batch.is_some_and(|prior| prior != current) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: State rows have inconsistent batch dimensions"),
                ));
            }
            batch = Some(current);
        }
    }
    if !supplied.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: execution State update has no canonical row"),
        ));
    }
    Ok(())
}

/// Publish one canonical table only after every supplied row is admitted.
/// Unchanged values share their existing storage; updates move their payloads.
pub(in crate::builtins::deep_learning) fn publish_updates(
    network: &ObjectInstance,
    updates: Vec<StateUpdate>,
    function: &'static str,
) -> BuiltinResult<Value> {
    let rows = view(network, function)?;
    let mut supplied = HashMap::with_capacity(updates.len());
    for update in updates {
        if supplied
            .insert(
                (update.layer_index, update.role),
                Value::Tensor(update.value),
            )
            .is_some()
        {
            return Err(deep_learning_error(
                function,
                format!("{function}: duplicate execution State update"),
            ));
        }
    }
    let mut keys = Vec::with_capacity(rows.len());
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value = supplied.remove(&(row.key.layer_index, row.key.role));
        if let Some(value) = &value {
            schema::validate_value(value, &row.key, function)?;
        }
        values.push(super::super::parameters::public_value(
            value.as_ref().or(Some(row.value)),
        )?);
        keys.push(row.key);
    }
    if !supplied.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: execution State update has no canonical row"),
        ));
    }
    let result = table::table(&keys, values, function)?;
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        unreachable!("view admitted network Layers")
    };
    table::validate_canonical(layers.data(), &result, function)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn update(role: StateRole, shape: Vec<usize>, value: f64) -> StateUpdate {
        let count = shape.iter().product();
        StateUpdate {
            layer_index: 0,
            role,
            value: Tensor::new(vec![value; count], shape).unwrap(),
        }
    }

    #[test]
    fn publishes_canonical_rows_without_changing_network() {
        let network = super::super::tests::recurrent_network();
        let before = project(&network, "forward").unwrap();
        let result = publish_updates(
            &network,
            vec![
                update(StateRole::CellState, vec![3, 2], 4.0),
                update(StateRole::HiddenState, vec![3, 2], 2.0),
            ],
            "forward",
        )
        .unwrap();
        assert_eq!(project(&network, "forward").unwrap(), before);
        let Value::Object(updated) = apply(network, result, "set.State").unwrap() else {
            panic!("expected network");
        };
        let rows = view(&updated, "forward").unwrap();
        assert_eq!(rows[0].key.role, StateRole::HiddenState);
        assert_eq!(rows[1].key.role, StateRole::CellState);
        for (row, expected) in rows.into_iter().zip([2.0, 4.0]) {
            let shape = super::super::super::parameters::numeric_payload_shape(
                row.value, "State", "forward",
            )
            .unwrap();
            assert_eq!(shape, [3, 2]);
            let Value::Object(wrapper) = row.value else {
                panic!("expected public dlarray State");
            };
            let Value::Tensor(tensor) = wrapper.properties().get("Data").unwrap() else {
                panic!("expected native State payload");
            };
            assert_eq!(tensor.as_f64_slice().unwrap(), &[expected; 6]);
        }
    }

    #[test]
    fn publication_preserves_single_storage() {
        let network = super::super::tests::recurrent_network();
        let value = Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F32(vec![0.25; 3]),
            vec![3, 1],
        )
        .unwrap();
        let result = publish_updates(
            &network,
            vec![StateUpdate {
                layer_index: 0,
                role: StateRole::HiddenState,
                value,
            }],
            "forward",
        )
        .unwrap();
        let Value::Object(updated) = apply(network, result, "set.State").unwrap() else {
            panic!("expected network");
        };
        let rows = view(&updated, "forward").unwrap();
        let Value::Object(wrapper) = rows[0].value else {
            panic!("expected dlarray");
        };
        let Value::Tensor(tensor) = wrapper.properties().get("Data").unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.as_f32_slice().unwrap(), &[0.25; 3]);
    }

    #[test]
    fn rejects_invalid_or_duplicate_updates_atomically() {
        let network = super::super::tests::recurrent_network();
        let before = project(&network, "forward").unwrap();
        let mut unknown = update(StateRole::HiddenState, vec![3, 1], 1.0);
        unknown.layer_index = 4;
        for updates in [
            vec![unknown],
            vec![update(StateRole::HiddenState, vec![2, 1], 1.0)],
            vec![
                update(StateRole::HiddenState, vec![3, 1], 1.0),
                update(StateRole::HiddenState, vec![3, 1], 2.0),
            ],
            vec![
                update(StateRole::HiddenState, vec![3, 1], 1.0),
                update(StateRole::CellState, vec![3, 2], 2.0),
            ],
        ] {
            assert!(admit_updates(&network, &updates, "forward").is_err());
            assert!(publish_updates(&network, updates, "forward").is_err());
            assert_eq!(project(&network, "forward").unwrap(), before);
        }
    }

    #[test]
    fn admits_complete_returned_state_without_replacing_input_rows() {
        let network = super::super::tests::recurrent_network();
        let updates = [
            update(StateRole::HiddenState, vec![3, 2], 2.0),
            update(StateRole::CellState, vec![3, 2], 4.0),
        ];
        assert!(admit_updates(&network, &updates, "forward").is_ok());
        assert_eq!(view(&network, "forward").unwrap().len(), 2);
    }
}
