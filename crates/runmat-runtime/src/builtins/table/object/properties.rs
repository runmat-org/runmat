use super::*;

pub(in crate::builtins::table) fn default_properties(
    variable_names: Vec<String>,
    row_names: Option<Vec<String>>,
) -> StructValue {
    default_properties_for_class(TABLE_CLASS, variable_names, row_names)
}

pub(in crate::builtins::table) fn default_properties_for_class(
    class_name: &str,
    variable_names: Vec<String>,
    row_names: Option<Vec<String>>,
) -> StructValue {
    let mut props = StructValue::new();
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(variable_names.clone(), vec![1, variable_names.len()])
                .expect("VariableNames shape is valid"),
        ),
    );
    props.insert(
        ROW_NAMES,
        row_names
            .map(|names| {
                Value::StringArray(
                    StringArray::new(names.clone(), vec![names.len(), 1])
                        .expect("RowNames shape is valid"),
                )
            })
            .unwrap_or_else(|| {
                Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap())
            }),
    );
    props.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(
                vec![
                    if class_name == TIMETABLE_CLASS {
                        "Time".to_string()
                    } else {
                        DEFAULT_ROW_DIM_NAME.to_string()
                    },
                    DEFAULT_VARIABLE_DIM_NAME.to_string(),
                ],
                vec![1, 2],
            )
            .expect("DimensionNames shape is valid"),
        ),
    );
    props.insert(
        VARIABLE_UNITS,
        Value::StringArray(
            StringArray::new(
                vec![String::new(); variable_names.len()],
                vec![1, variable_names.len()],
            )
            .expect("VariableUnits shape is valid"),
        ),
    );
    props.insert(
        VARIABLE_DESCRIPTIONS,
        Value::StringArray(
            StringArray::new(
                vec![String::new(); variable_names.len()],
                vec![1, variable_names.len()],
            )
            .expect("VariableDescriptions shape is valid"),
        ),
    );
    props.insert(DESCRIPTION, Value::String(String::new()));
    props.insert(USER_DATA, Value::Tensor(Tensor::zeros(vec![0, 0])));
    props
}

pub(in crate::builtins::table) fn table_public_properties(
    object: &ObjectInstance,
) -> BuiltinResult<StructValue> {
    match object
        .properties
        .get(TABLE_PROPERTIES_FIELD)
        .or_else(|| object.properties.get(PROPERTIES_MEMBER))
    {
        Some(Value::Struct(st)) => Ok(st.clone()),
        Some(other) => Err(invalid_variable(format!(
            "table: invalid Properties storage {other:?}"
        ))),
        None => Ok(default_properties(
            table_variable_names_from_object(object)?,
            None,
        )),
    }
}

pub(in crate::builtins::table) fn sync_table_properties(
    object: &mut ObjectInstance,
    props: StructValue,
) {
    object.properties.insert(
        TABLE_PROPERTIES_FIELD.to_string(),
        Value::Struct(props.clone()),
    );
    object
        .properties
        .insert(PROPERTIES_MEMBER.to_string(), Value::Struct(props));
}

pub(in crate::builtins::table) fn set_table_dimension_names(
    object: &mut ObjectInstance,
    dimension_names: Vec<String>,
    context: &str,
) -> BuiltinResult<()> {
    let mut properties = table_public_properties(object)?;
    properties.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(dimension_names, vec![1, 2])
                .map_err(|error| invalid_variable(format!("{context}: {error}")))?,
        ),
    );
    sync_table_properties(object, properties);
    Ok(())
}

pub(in crate::builtins::table) fn apply_properties(
    object: &mut ObjectInstance,
    mut props: StructValue,
) -> BuiltinResult<()> {
    if let Some(value) = props.fields.get(VARIABLE_NAMES) {
        let names = variable_name_list(value)?;
        rename_table_variables(object, names.clone())?;
        update_variable_metadata_names(&mut props, names)?;
    }
    sync_table_properties(object, props);
    Ok(())
}

pub(in crate::builtins::table) fn rename_table_variables(
    object: &mut ObjectInstance,
    new_names: Vec<String>,
) -> BuiltinResult<()> {
    let old_names = table_variable_names_from_object(object)?;
    if old_names.len() != new_names.len() {
        return Err(invalid_variable(
            "table: VariableNames assignment must preserve variable count",
        ));
    }
    validate_variable_names(&new_names)?;
    let variables = table_variables(object)?;
    let mut renamed = StructValue::new();
    for (old, new) in old_names.iter().zip(new_names.iter()) {
        let value = variables
            .fields
            .get(old)
            .cloned()
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{old}'")))?;
        renamed.insert(new.clone(), value);
    }
    object
        .properties
        .insert(TABLE_VARIABLES_FIELD.to_string(), Value::Struct(renamed));
    Ok(())
}

pub(in crate::builtins::table) fn update_variable_metadata_names(
    props: &mut StructValue,
    names: Vec<String>,
) -> BuiltinResult<()> {
    props.insert(
        VARIABLE_NAMES,
        Value::StringArray(
            StringArray::new(names.clone(), vec![1, names.len()])
                .map_err(|err| invalid_variable(format!("table: {err}")))?,
        ),
    );
    for field in [VARIABLE_UNITS, VARIABLE_DESCRIPTIONS] {
        let existing = props.fields.get(field).cloned();
        let values = match existing {
            Some(Value::StringArray(mut array)) => {
                array.data.resize(names.len(), String::new());
                array.data.truncate(names.len());
                array.data
            }
            _ => vec![String::new(); names.len()],
        };
        props.insert(
            field,
            Value::StringArray(
                StringArray::new(values, vec![1, names.len()])
                    .map_err(|err| invalid_variable(format!("table: {err}")))?,
            ),
        );
    }
    Ok(())
}
