use super::*;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, NumericScalar, Tensor,
};
use runmat_macros::runtime_builtin;

const BUILTIN_NAME: &str = "array2table";

const CELL2TABLE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Table result.",
}];
const CELL2TABLE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-dimensional cell array whose columns become table variables.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "VariableNames and RowNames options.",
    },
];
const CELL2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = cell2table(C, Name, Value, ...)",
    inputs: &CELL2TABLE_INPUTS,
    outputs: &CELL2TABLE_OUTPUTS,
}];
const CELL2TABLE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_INDEX,
    TABLE_ERROR_INVALID_VARIABLE,
];
pub const CELL2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELL2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CELL2TABLE_ERRORS,
};

const CELL2TABLE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "C contents",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "All eight integer classes may occur in cell contents. Compatible one-row values in a column are vertically concatenated into an integer table variable; scalar doubles may coexist except with int64 or uint64.",
    }];
pub const CELL2TABLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "T = cell2table(C, Name,Value...) with integer cell contents",
        inputs: &CELL2TABLE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Homogeneous same-class integer columns preserve exact authoritative storage. Mixed integer columns use the leftmost integer class and saturating assignment; scalar doubles use that conversion except with int64/uint64. Incompatible columns remain cells. Nested resident objects are preserved without eager gather; resident numeric arrays are currently incompatible rather than concatenated into a resident variable.",
    }];

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
    set_table_dimension_names(object, dimension_names, "array2table")?;
    Ok(table)
}

#[runtime_builtin(
    name = "cell2table",
    category = "table",
    summary = "Convert a cell array into a table.",
    keywords = "cell2table,table,cell,VariableNames,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::builtins::conversions::CELL2TABLE_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::table::builtins::conversions::CELL2TABLE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn cell2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "cell2table")?;
    let Value::Cell(cell) = value else {
        return Err(invalid_argument("cell2table: expected cell array input"));
    };
    if cell.shape.len() > 2 {
        return Err(invalid_argument(
            "cell2table: C must be a two-dimensional cell array",
        ));
    }
    let mut columns = Vec::with_capacity(cell.cols);
    for col in 0..cell.cols {
        let mut data = Vec::with_capacity(cell.rows);
        for row in 0..cell.rows {
            data.push(cell.get(row, col).map_err(invalid_index)?);
        }
        if let Some(integer_column) = compatible_integer_column(&data, cell.rows)? {
            columns.push(integer_column);
        } else {
            columns
                .push(Value::Cell(CellArray::new(data, cell.rows, 1).map_err(
                    |err| invalid_variable(format!("cell2table: {err}")),
                )?));
        }
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

fn compatible_integer_column(values: &[Value], rows: usize) -> BuiltinResult<Option<Value>> {
    let Some(prototype) = values
        .iter()
        .find_map(|value| integer_row(value).map(|row| row.0))
    else {
        return Ok(None);
    };
    let Some(width) = values.first().and_then(integer_compatible_row_width) else {
        return Ok(None);
    };
    if values.iter().skip(1).any(|value| {
        integer_compatible_row_width(value).is_none_or(|next_width| next_width != width)
    }) {
        return Ok(None);
    }
    let has_scalar_double = values.iter().any(|value| matches!(value, Value::Num(_)));
    if has_scalar_double
        && matches!(
            prototype,
            runmat_builtins::IntegerStorage::I64(_) | runmat_builtins::IntegerStorage::U64(_)
        )
    {
        return Ok(None);
    }
    let total = rows
        .checked_mul(width)
        .ok_or_else(|| invalid_variable("cell2table: integer variable shape exceeds limits"))?;
    let mut output = Tensor::new_integer(prototype.zeros_like(total), vec![rows, width])
        .map_err(|err| invalid_variable(format!("cell2table: {err}")))?;
    for (row, value) in values.iter().enumerate() {
        let actual_width = integer_compatible_row_width(value).expect("validated integer row");
        for col in 0..actual_width {
            let scalar = integer_row_value(value, col).expect("validated integer row value");
            output
                .set_numeric_assignment_at(row + col * rows, scalar)
                .map_err(|err| invalid_variable(format!("cell2table: {err}")))?;
        }
    }
    Ok(Some(Value::Tensor(output)))
}

fn integer_compatible_row_width(value: &Value) -> Option<usize> {
    match value {
        Value::Num(_) | Value::Int(_) => Some(1),
        _ => integer_row(value).map(|(_, width)| width),
    }
}

fn integer_row(value: &Value) -> Option<(&runmat_builtins::IntegerStorage, usize)> {
    match value {
        Value::Int(value) => Some((integer_scalar_prototype(value), 1)),
        Value::Tensor(tensor)
            if tensor.shape.len() <= 2
                && tensor.shape.first().copied().unwrap_or(1) == 1
                && tensor.integer_storage().is_some() =>
        {
            Some((tensor.integer_storage().expect("checked"), tensor.len()))
        }
        _ => None,
    }
}

fn integer_scalar_prototype(value: &IntValue) -> &'static runmat_builtins::IntegerStorage {
    use runmat_builtins::IntegerStorage;
    static I8: IntegerStorage = IntegerStorage::I8(Vec::new());
    static I16: IntegerStorage = IntegerStorage::I16(Vec::new());
    static I32: IntegerStorage = IntegerStorage::I32(Vec::new());
    static I64: IntegerStorage = IntegerStorage::I64(Vec::new());
    static U8: IntegerStorage = IntegerStorage::U8(Vec::new());
    static U16: IntegerStorage = IntegerStorage::U16(Vec::new());
    static U32: IntegerStorage = IntegerStorage::U32(Vec::new());
    static U64: IntegerStorage = IntegerStorage::U64(Vec::new());
    match value {
        IntValue::I8(_) => &I8,
        IntValue::I16(_) => &I16,
        IntValue::I32(_) => &I32,
        IntValue::I64(_) => &I64,
        IntValue::U8(_) => &U8,
        IntValue::U16(_) => &U16,
        IntValue::U32(_) => &U32,
        IntValue::U64(_) => &U64,
    }
}

fn integer_row_value(value: &Value, index: usize) -> Option<NumericScalar> {
    match value {
        Value::Num(value) if index == 0 => Some(NumericScalar::F64(*value)),
        Value::Int(value) if index == 0 => Some(NumericScalar::from(value.clone())),
        Value::Tensor(tensor) => tensor
            .integer_storage()?
            .value_at(index)
            .map(NumericScalar::from),
        _ => None,
    }
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

#[cfg(test)]
mod cell2table_tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

    #[test]
    fn dedicated_descriptor_is_fixed_and_public() {
        assert_eq!(CELL2TABLE_DESCRIPTOR.signatures.len(), 1);
        assert!(CELL2TABLE_DESCRIPTOR.signatures[0]
            .label
            .starts_with("T = cell2table"));
        assert!(matches!(
            CELL2TABLE_DESCRIPTOR.output_mode,
            BuiltinOutputMode::Fixed
        ));
        assert!(matches!(
            CELL2TABLE_DESCRIPTOR.completion_policy,
            BuiltinCompletionPolicy::Public
        ));
    }

    #[test]
    fn compatible_integer_columns_preserve_and_saturate_leftmost_class() {
        let exact = vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ),
            Value::Int(IntValue::U64(1_u64 << 63)),
        ];
        let Value::Tensor(exact) = compatible_integer_column(&exact, 2).unwrap().unwrap() else {
            panic!("expected integer variable");
        };
        assert_eq!(
            exact.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        );

        let mixed = vec![
            Value::Int(IntValue::U8(9)),
            Value::Int(IntValue::I64(-1)),
            Value::Int(IntValue::U64(u64::MAX)),
        ];
        let Value::Tensor(mixed) = compatible_integer_column(&mixed, 3).unwrap().unwrap() else {
            panic!("expected integer variable");
        };
        assert_eq!(
            mixed.integer_storage(),
            Some(&IntegerStorage::U8(vec![9, 0, u8::MAX]))
        );

        let scalar_double = vec![Value::Num(300.2), Value::Int(IntValue::U8(7))];
        let Value::Tensor(scalar_double) = compatible_integer_column(&scalar_double, 2)
            .unwrap()
            .unwrap()
        else {
            panic!("expected integer variable");
        };
        assert_eq!(
            scalar_double.integer_storage(),
            Some(&IntegerStorage::U8(vec![u8::MAX, 7]))
        );

        assert!(
            compatible_integer_column(&[Value::Int(IntValue::U64(1)), Value::Num(2.0)], 2)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn compatible_integer_columns_preserve_every_native_class() {
        for value in [
            IntValue::I8(7),
            IntValue::I16(7),
            IntValue::I32(7),
            IntValue::I64(7),
            IntValue::U8(7),
            IntValue::U16(7),
            IntValue::U32(7),
            IntValue::U64(7),
        ] {
            let expected_dtype = integer_scalar_prototype(&value).numeric_dtype();
            let Value::Tensor(column) =
                compatible_integer_column(&[Value::Int(value.clone()), Value::Int(value)], 2)
                    .unwrap()
                    .unwrap()
            else {
                panic!("expected integer variable");
            };
            assert_eq!(column.numeric_dtype(), expected_dtype);
            assert_eq!(column.materialize_f64(), vec![7.0, 7.0]);
        }
    }

    #[test]
    fn incompatible_columns_remain_cells_and_rank_over_two_rejects() {
        assert!(
            compatible_integer_column(&[Value::Int(IntValue::U8(1)), Value::from("text")], 2)
                .unwrap()
                .is_none()
        );

        let cell =
            CellArray::new_with_shape(vec![Value::Int(IntValue::U8(1))], vec![1, 1, 1]).unwrap();
        let err = block_on(cell2table_builtin(Value::Cell(cell), Vec::new())).unwrap_err();
        assert!(err.message().contains("two-dimensional"));
    }
}
