use super::*;
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;

const BUILTIN_NAME: &str = "array2table";

pub(crate) const ARRAY2TABLE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "array2table-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "array2table with an interactive resident GPU input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Array2TableGpuInputExtension"),
    };

pub const ARRAY2TABLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [ARRAY2TABLE_GPU_INPUT_EXTENSION];

const ARRAY2TABLE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented homogeneous array domain includes all eight real integer classes.",
    }];

pub const ARRAY2TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "T = array2table(integer_A, Name,Value...)",
        inputs: &ARRAY2TABLE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each column becomes a table variable with A's exact authoritative integer storage and class. Interactive resident GPU input is a mode-gated RunMat extension that gathers before table construction.",
    }];

#[runtime_builtin(
    name = "array2table",
    category = "table",
    summary = "Convert an array into a table.",
    keywords = "array2table,table,VariableNames,RowNames,DimensionNames",
    accel = "gather",
    descriptor(crate::builtins::table::ARRAY2TABLE_DESCRIPTOR),
    extensions(crate::builtins::table::builtins::conversions::ARRAY2TABLE_EXTENSIONS),
    integer_capabilities(
        crate::builtins::table::builtins::conversions::ARRAY2TABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAY2TABLE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let mut options = parse_array2table_options(&rest)?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    validate_array2table_names(
        &names,
        &mut options.row_names,
        options.dimension_names.as_deref(),
    )?;
    let table = table_from_columns_with_properties(names, columns, options.row_names)?;
    apply_array2table_dimension_names(table, options.dimension_names)
}

fn apply_array2table_dimension_names(
    mut table: Value,
    dimension_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    let Some(dimension_names) = dimension_names else {
        return Ok(table);
    };
    let Value::Object(object) = &mut table else {
        return Err(invalid_variable(
            "array2table: internal table construction failed",
        ));
    };
    let mut properties = table_public_properties(object)?;
    properties.insert(
        DIMENSION_NAMES,
        Value::StringArray(
            StringArray::new(dimension_names, vec![1, 2])
                .map_err(|error| invalid_variable(format!("array2table: {error}")))?,
        ),
    );
    sync_table_properties(object, properties);
    Ok(table)
}

#[runtime_builtin(
    name = "cell2table",
    category = "table",
    summary = "Convert a cell array into a table.",
    keywords = "cell2table,table,cell,VariableNames,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn cell2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "cell2table")?;
    let Value::Cell(cell) = value else {
        return Err(invalid_argument("cell2table: expected cell array input"));
    };
    let mut columns = Vec::with_capacity(cell.cols);
    for col in 0..cell.cols {
        let mut data = Vec::with_capacity(cell.rows);
        for row in 0..cell.rows {
            data.push(cell.get(row, col).map_err(invalid_index)?);
        }
        columns
            .push(Value::Cell(CellArray::new(data, cell.rows, 1).map_err(
                |err| invalid_variable(format!("cell2table: {err}")),
            )?));
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

#[runtime_builtin(
    name = "struct2table",
    category = "table",
    summary = "Convert a scalar struct into a table.",
    keywords = "struct2table,table,struct,AsArray,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn struct2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_struct2table_options(&rest)?;
    match value {
        Value::Struct(st) => {
            let mut names = Vec::with_capacity(st.fields.len());
            let mut columns = Vec::with_capacity(st.fields.len());
            for (name, value) in st.fields {
                names.push(name);
                if options.as_array && value_row_count(&value)? != 1 {
                    columns.push(Value::Cell(
                        CellArray::new(vec![value], 1, 1).map_err(invalid_variable)?,
                    ));
                } else {
                    columns.push(value);
                }
            }
            let names = options.table.variable_names.unwrap_or(names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        Value::Cell(cell)
            if cell
                .data
                .iter()
                .all(|value| matches!(value, Value::Struct(_))) =>
        {
            let rows = cell.data.len();
            let first = cell.data.iter().find_map(|value| match value {
                Value::Struct(st) => Some(st),
                _ => None,
            });
            let field_names = first
                .map(|st| st.fields.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let mut columns = Vec::with_capacity(field_names.len());
            for name in &field_names {
                let mut values = Vec::with_capacity(rows);
                for value in &cell.data {
                    let Value::Struct(st) = value else {
                        unreachable!("checked above")
                    };
                    values.push(st.fields.get(name).cloned().unwrap_or(Value::Num(f64::NAN)));
                }
                columns.push(Value::Cell(
                    CellArray::new(values, rows, 1).map_err(invalid_variable)?,
                ));
            }
            let names = options.table.variable_names.unwrap_or(field_names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        other => Err(invalid_argument(format!(
            "struct2table: expected struct or struct array, got {other:?}"
        ))),
    }
}

#[runtime_builtin(
    name = "table2struct",
    category = "table",
    summary = "Convert a table into row structs or a scalar struct of variables.",
    keywords = "table2struct,table,struct,ToScalar",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2struct_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let to_scalar = parse_table2struct_to_scalar(&rest)?;
    let object = into_table_object(host, "table2struct")?;
    if to_scalar {
        return Ok(Value::Struct(table_variables(&object)?));
    }
    let height = table_height(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut rows = Vec::with_capacity(height);
    for row in 0..height {
        let mut st = StructValue::new();
        for name in &names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("table2struct: missing variable '{name}'"))
            })?;
            st.insert(name.clone(), row_value(value, row)?);
        }
        rows.push(Value::Struct(st));
    }
    CellArray::new(rows, height, 1)
        .map(Value::Cell)
        .map_err(invalid_variable)
}

#[runtime_builtin(
    name = "table2array",
    category = "table",
    summary = "Convert table variables into a homogeneous array when possible.",
    keywords = "table2array,table,array",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2array_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2array")?;
    table_brace_get(&object, &colon_colon_payload())
}

#[runtime_builtin(
    name = "table2cell",
    category = "table",
    summary = "Convert table variables into a cell array.",
    keywords = "table2cell,table,cell",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2cell_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2cell")?;
    table_to_cell_array(&object)
}
