use super::*;

pub(super) fn partition_object(spec: PartitionSpec) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CVPARTITION_CLASS.to_string());
    object
        .properties
        .insert("NumObservations".to_string(), Value::Num(spec.n as f64));
    object.properties.insert(
        "NumTestSets".to_string(),
        Value::Num(spec.test_set_count as f64),
    );
    object.properties.insert(
        "TestSize".to_string(),
        size_vector(&spec.test_sets, spec.n, spec.test_set_count, true)?,
    );
    object.properties.insert(
        "TrainSize".to_string(),
        size_vector(&spec.test_sets, spec.n, spec.test_set_count, false)?,
    );
    object.properties.insert(
        "Type".to_string(),
        Value::string(spec.kind.property_name().to_string()),
    );
    object
        .properties
        .insert("IsCustom".to_string(), Value::Bool(spec.is_custom));
    object
        .properties
        .insert("IsGrouped".to_string(), Value::Bool(false));
    object
        .properties
        .insert("IsStratified".to_string(), Value::Bool(spec.is_stratified));
    object.properties.insert(
        TEST_SETS_PROPERTY.to_string(),
        Value::LogicalArray(
            LogicalArray::new(spec.test_sets, vec![spec.n, spec.test_set_count])
                .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        ),
    );
    object.properties.insert(
        ACTIVE_ROWS_PROPERTY.to_string(),
        Value::LogicalArray(
            LogicalArray::new(spec.active_rows, vec![spec.n, 1])
                .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        ),
    );
    Ok(object)
}

pub(super) fn size_vector(
    test_sets: &[u8],
    rows: usize,
    cols: usize,
    test_size: bool,
) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(cols);
    for col in 0..cols {
        let count = test_sets[col * rows..(col + 1) * rows]
            .iter()
            .filter(|flag| **flag != 0)
            .count();
        let size = if test_size { count } else { rows - count };
        data.push(size as f64);
    }
    if cols == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new(data, vec![1, cols])
            .map(Value::Tensor)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))
    }
}
