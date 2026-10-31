use super::*;

pub(in crate::builtins::table) fn default_properties(
    variable_names: Vec<String>,
    row_names: Option<Vec<String>>,
) -> StructValue {
    default_properties_for_class(TABLE_CLASS, variable_names, row_names)
}

pub(in crate::builtins::table) fn default_properties_for_class(
    class_name: impl Into<runmat_types::ClassIdentity>,
    variable_names: Vec<String>,
    row_names: Option<Vec<String>>,
) -> StructValue {
    let class_name = class_name.into();
    let mut props = runmat_value::UnboundedStructValueBuilder::new();
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
                    if class_name.is(TIMETABLE_CLASS) {
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
    props.insert(DESCRIPTION, Value::string(String::new()));
    props.insert(USER_DATA, Value::Tensor(Tensor::zeros(vec![0, 0])));
    props.finish()
}

pub(in crate::builtins::table) fn table_public_properties(
    object: &ObjectInstance,
) -> BuiltinResult<StructValue> {
    match object
        .properties()
        .get(TABLE_PROPERTIES_FIELD)
        .or_else(|| object.properties().get(PROPERTIES_MEMBER))
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
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    props: StructValue,
) -> BuiltinResult<()> {
    let updates = [
        (TABLE_PROPERTIES_FIELD, Value::Struct(props.clone())),
        (PROPERTIES_MEMBER, Value::Struct(props)),
    ];
    replace_table_properties(
        runtime,
        object,
        updates.iter().map(|(name, value)| (*name, value)),
    )
}

pub(in crate::builtins::table) fn replace_table_property(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    name: &str,
    value: Value,
) -> BuiltinResult<Option<Value>> {
    crate::execution::object::replacement::replace_property(
        runtime,
        object,
        name,
        value,
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:table:HostAllocation",
                message,
                source,
            )
        },
        invalid_variable,
    )
}

pub(in crate::builtins::table) fn replace_table_properties<'a, I>(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    updates: I,
) -> BuiltinResult<()>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    crate::execution::object::replacement::replace_properties(
        runtime,
        object,
        updates,
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:table:HostAllocation",
                message,
                source,
            )
        },
        invalid_variable,
    )
}

pub(in crate::builtins::table) fn set_table_dimension_names(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    dimension_names: Vec<String>,
    context: &str,
) -> BuiltinResult<()> {
    let properties = table_public_properties(object)?;
    let value = Value::StringArray(
        StringArray::new(dimension_names, vec![1, 2])
            .map_err(|error| invalid_variable(format!("{context}: {error}")))?,
    );
    let (properties, _roots) =
        replace_struct_fields(runtime, &properties, &[(DIMENSION_NAMES, value)])?;
    sync_table_properties(runtime, object, properties)
}

pub(in crate::builtins::table) fn apply_properties(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut ObjectInstance,
    props: StructValue,
) -> BuiltinResult<()> {
    let renamed = if let Some(value) = props.fields.get(VARIABLE_NAMES) {
        let names = variable_name_list(value)?;
        let renamed = renamed_table_variables(object, &names)?;
        Some((renamed, names))
    } else {
        None
    };
    if let Some((renamed, names)) = renamed {
        let (props, _property_roots) = updated_variable_metadata_names(runtime, &props, names)?;
        let updates = [
            (TABLE_VARIABLES_FIELD, Value::Struct(renamed)),
            (TABLE_PROPERTIES_FIELD, Value::Struct(props.clone())),
            (PROPERTIES_MEMBER, Value::Struct(props)),
        ];
        replace_table_properties(
            runtime,
            object,
            updates.iter().map(|(name, value)| (*name, value)),
        )
    } else {
        sync_table_properties(runtime, object, props)
    }
}

fn renamed_table_variables(
    object: &ObjectInstance,
    new_names: &[String],
) -> BuiltinResult<StructValue> {
    let old_names = table_variable_names_from_object(object)?;
    if old_names.len() != new_names.len() {
        return Err(invalid_variable(
            "table: VariableNames assignment must preserve variable count",
        ));
    }
    validate_variable_names(new_names)?;
    let variables = table_variables(object)?;
    let mut renamed = runmat_value::UnboundedStructValueBuilder::new();
    for (old, new) in old_names.iter().zip(new_names.iter()) {
        let value = variables
            .fields
            .get(old)
            .cloned()
            .ok_or_else(|| invalid_variable(format!("table: missing variable '{old}'")))?;
        renamed.insert(new.clone(), value);
    }
    Ok(renamed.finish())
}

pub(in crate::builtins::table) fn updated_variable_metadata_names(
    runtime: Option<&crate::context::RuntimeContext>,
    props: &StructValue,
    names: Vec<String>,
) -> BuiltinResult<(StructValue, Option<crate::execution::RootedValueLease>)> {
    let variable_names = Value::StringArray(
        StringArray::new(names.clone(), vec![1, names.len()])
            .map_err(|err| invalid_variable(format!("table: {err}")))?,
    );
    let mut metadata = [
        (VARIABLE_UNITS, Value::Tensor(Tensor::zeros(vec![0, 0]))),
        (
            VARIABLE_DESCRIPTIONS,
            Value::Tensor(Tensor::zeros(vec![0, 0])),
        ),
    ];
    for (field, output) in &mut metadata {
        let existing = props.fields.get(field).cloned();
        let values = match existing {
            Some(Value::StringArray(array)) => {
                let mut values = array.data().to_vec();
                values.resize(names.len(), runmat_value::StringElement::text(""));
                values.truncate(names.len());
                values
            }
            _ => vec![runmat_value::StringElement::text(""); names.len()],
        };
        *output = Value::StringArray(
            StringArray::from_elements_unbounded(values, vec![1, names.len()])
                .map_err(|err| invalid_variable(format!("table: {err}")))?,
        );
    }
    let updates = [
        (VARIABLE_NAMES, variable_names),
        metadata[0].clone(),
        metadata[1].clone(),
    ];
    replace_struct_fields(runtime, props, &updates)
}

pub(in crate::builtins::table) fn replace_struct_fields(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &StructValue,
    updates: &[(&str, Value)],
) -> BuiltinResult<(StructValue, Option<crate::execution::RootedValueLease>)> {
    if let Some(runtime) = runtime {
        let _source = crate::execution::structure::replacement::lease(
            &Value::Struct(source.clone()),
            Some(runtime),
        )?;
        let fields = source
            .fields
            .iter()
            .filter(|(name, _)| !updates.iter().any(|(updated, _)| name == updated))
            .chain(updates.iter().map(|(name, value)| (*name, value)));
        let output = crate::execution::construct_struct_from_fields_in_context(runtime, fields)?;
        let (output, roots) = output.into_value_and_lease();
        let Value::Struct(output) = output else {
            unreachable!("structure replacement publishes a structure")
        };
        return Ok((output, Some(roots)));
    }
    let mut builder = runmat_value::UnboundedStructValueBuilder::from_structure(source.clone());
    for (name, value) in updates {
        builder.insert(*name, value.clone());
    }
    Ok((builder.finish(), None))
}
