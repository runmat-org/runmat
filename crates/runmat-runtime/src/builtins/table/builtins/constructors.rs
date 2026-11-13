use super::*;
mod file_datastore;
pub(super) mod allocation;
pub(super) mod basic;
#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) use basic::*;
use runmat_builtins::BuiltinExtensionDescriptor;
#[cfg(test)]
pub(crate) use runmat_builtins::{
    ARRAY_DATASTORE_EXTENSIONS, ARRAY_DATASTORE_INTEGER_CAPABILITIES,
    PARQUET_DATASTORE_INTEGER_CAPABILITIES, ROWFILTER_INTEGER_AUDIT, TIMERANGE_DESCRIPTOR,
    TIMERANGE_EXTENSIONS, TIMERANGE_INTEGER_CAPABILITIES, UITABLE_EXTENSIONS,
    UITABLE_INTEGER_CAPABILITIES, VARTYPE_INTEGER_AUDIT,
};
pub(crate) use runmat_builtins::{
    ARRAY_DATASTORE_GPU_INPUT_EXTENSION, TIMERANGE_EXPLICIT_GPU_EXTENSION,
    TIMERANGE_NUMERIC_BOUNDS_EXTENSION, UITABLE_EXPLICIT_GPU_EXTENSION,
};
#[cfg(test)]
pub(crate) use runmat_builtins::{
    CATEGORICAL_GPU_INPUT_EXTENSION, DICTIONARY_EXTENSIONS, DICTIONARY_GPU_INPUT_EXTENSION,
    DICTIONARY_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericDType;

const ARRAY_DATASTORE_BUILTIN_NAME: &str = "arrayDatastore";

fn preallocated_table(
    variables: Vec<Value>,
    options: TableConstructorOptions,
) -> BuiltinResult<Value> {
    let (names, columns, row_names, _) = preallocated_table_columns(variables, options, "table")?;
    table_from_columns_with_properties(names, columns, row_names)
}

pub(in crate::builtins::table) type PreallocatedTableColumns =
    (Vec<String>, Vec<Value>, Option<Vec<String>>, usize);

pub(in crate::builtins::table) fn preallocated_table_columns(
    variables: Vec<Value>,
    options: TableConstructorOptions,
    context: &str,
) -> BuiltinResult<PreallocatedTableColumns> {
    if !variables.is_empty() {
        return Err(invalid_argument(format!(
            "{context}: Size/VariableTypes preallocation cannot be combined with data variables"
        )));
    }
    let size = options
        .size
        .as_ref()
        .ok_or_else(|| invalid_argument(format!("{context}: preallocation requires Size")))?;
    let variable_types = options.variable_types.as_ref().ok_or_else(|| {
        invalid_argument(format!("{context}: preallocation requires VariableTypes"))
    })?;
    let [rows, width] = table_preallocation_size(size)?;
    if variable_types.len() != width {
        return Err(invalid_argument(format!(
            "{context}: VariableTypes has {} entries but Size requests {width} variables",
            variable_types.len()
        )));
    }
    let mut columns = Vec::with_capacity(width);
    for type_name in variable_types {
        let dtype = table_preallocation_numeric_dtype(type_name).ok_or_else(|| {
            invalid_argument(format!(
                "{context}: preallocation type '{type_name}' is not yet supported; use double, single, or a built-in integer type"
            ))
        })?;
        columns.push(Value::Tensor(
            crate::builtins::common::tensor::zeros_with_dtype(&[rows, 1], dtype)
                .map_err(invalid_variable)?,
        ));
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(width));
    Ok((names, columns, options.row_names, rows))
}

fn table_preallocation_size(value: &Value) -> BuiltinResult<[usize; 2]> {
    let Value::Tensor(tensor) = value else {
        return Err(invalid_argument(
            "table: Size must be a two-element numeric vector",
        ));
    };
    if tensor.len() != 2 || (tensor.rows() != 1 && tensor.cols() != 1) {
        return Err(invalid_argument(
            "table: Size must be a two-element numeric vector",
        ));
    }
    let mut size = [0usize; 2];
    for (index, output) in size.iter_mut().enumerate() {
        let scalar = tensor
            .numeric_value_at(index)
            .ok_or_else(|| invalid_argument("table: Size contains a missing value"))?;
        *output = match scalar {
            runmat_value::NumericScalar::F64(value) => table_size_float(value)?,
            runmat_value::NumericScalar::F32(value) => table_size_float(f64::from(value))?,
            integer => integer
                .into_int_value()
                .and_then(|value| value.try_to_usize())
                .ok_or_else(|| {
                    invalid_argument("table: Size values must be nonnegative platform integers")
                })?,
        };
    }
    Ok(size)
}

fn table_size_float(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return Err(invalid_argument(
            "table: Size values must be nonnegative platform integers",
        ));
    }
    Ok(value as usize)
}

fn table_preallocation_numeric_dtype(type_name: &str) -> Option<NumericDType> {
    runmat_types::NumericClass::from_class_name(type_name.trim()).map(NumericDType::from)
}

#[runtime_builtin(
    name = "timerange",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timerange_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if !(2..=3).contains(&args.len()) {
        return Err(invalid_argument(
            "timerange: expected start, end, and optional inclusivity",
        ));
    }
    if args.iter().take(2).any(timerange_argument_is_numeric) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMERANGE_NUMERIC_BOUNDS_EXTENSION,
            "timerange",
        )?;
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TIMERANGE_EXPLICIT_GPU_EXTENSION,
            "timerange",
        )?;
    }
    let gathered = gather_retained_values(&args, runtime.as_ref()).await?;
    let inclusivity = gathered
        .get(2)
        .map(|value| scalar_text(value, "timerange inclusivity"))
        .transpose()?
        .unwrap_or_else(|| "openright".to_string());
    if !["openright", "openleft", "open", "closed"]
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(inclusivity.trim()))
    {
        return Err(invalid_argument(format!(
            "timerange: unsupported inclusivity '{inclusivity}'"
        )));
    }
    let fields = [
        ("Start", gathered[0].clone()),
        ("End", gathered[1].clone()),
        (
            "Inclusivity",
            Value::string(inclusivity.trim().to_ascii_lowercase()),
        ),
    ];
    construct_descriptor_object(runtime.as_ref(), TIMERANGE_CLASS, &fields)
}

fn timerange_argument_is_numeric(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

#[runtime_builtin(
    name = "vartype",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn vartype_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if crate::builtins::common::validation::value_has_native_integer_class(&value)
        || matches!(value, Value::GpuTensor(_))
    {
        return Err(invalid_argument(
            "vartype: type must be a host string scalar or character vector",
        ));
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let kind = scalar_text(&value, "vartype type")?;
    let fields = [("Type", Value::string(kind))];
    construct_descriptor_object(runtime.as_ref(), VARTYPE_CLASS, &fields)
}

#[runtime_builtin(
    name = "rowfilter",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn rowfilter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.len() > 2 {
        return Err(invalid_argument(
            "rowfilter: expected at most a source and optional predicate",
        ));
    }
    if args.iter().any(|value| {
        crate::builtins::common::validation::value_has_native_integer_class(value)
            || matches!(value, Value::GpuTensor(_))
    }) {
        return Err(invalid_argument(
            "rowfilter: typed numeric arrays are not valid rowfilter constructor inputs",
        ));
    }
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    let fields = [
        (
            "Variables",
            args.first()
                .cloned()
                .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())),
        ),
        (
            "Predicate",
            args.get(1)
                .cloned()
                .unwrap_or_else(|| Value::string(String::new())),
        ),
    ];
    construct_descriptor_object(runtime.as_ref(), ROWFILTER_CLASS, &fields)
}

#[runtime_builtin(
    name = "arrayDatastore",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array_datastore_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAY_DATASTORE_GPU_INPUT_EXTENSION,
            ARRAY_DATASTORE_BUILTIN_NAME,
        )?;
    }
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    let Some(data) = args.first().cloned() else {
        return Err(invalid_argument(
            "arrayDatastore: input array A is required",
        ));
    };
    if (args.len() - 1) % 2 != 0 {
        return Err(invalid_argument(
            "arrayDatastore: name-value options must be provided in pairs",
        ));
    }
    let mut read_size = 1usize;
    let mut iteration_dimension = 1usize;
    let mut output_type = "cell".to_string();
    let mut index = 1usize;
    while index < args.len() {
        let name = scalar_text(&args[index], "arrayDatastore option")?;
        let value = &args[index + 1];
        if name.eq_ignore_ascii_case("ReadSize") {
            read_size = array_datastore_positive_double_integer(value, "ReadSize")?;
        } else if name.eq_ignore_ascii_case("IterationDimension") {
            iteration_dimension =
                array_datastore_positive_double_integer(value, "IterationDimension")?;
        } else if name.eq_ignore_ascii_case("OutputType") {
            output_type = scalar_text(value, "arrayDatastore OutputType")?.to_ascii_lowercase();
            if output_type != "cell" && output_type != "same" {
                return Err(invalid_argument(
                    "arrayDatastore: OutputType must be 'cell' or 'same'",
                ));
            }
        } else {
            return Err(invalid_argument(format!(
                "arrayDatastore: unsupported option '{name}'"
            )));
        }
        index += 2;
    }
    if output_type == "same" && iteration_dimension != 1 {
        return Err(invalid_argument(
            "arrayDatastore: IterationDimension must be 1 when OutputType is 'same'",
        ));
    }
    let fields = [
        ("Data", data),
        ("ReadSize", Value::Num(read_size as f64)),
        ("IterationDimension", Value::Num(iteration_dimension as f64)),
        ("OutputType", Value::string(output_type)),
    ];
    construct_descriptor_object(runtime.as_ref(), ARRAY_DATASTORE_CLASS, &fields)
}

fn array_datastore_positive_double_integer(value: &Value, property: &str) -> BuiltinResult<usize> {
    let value = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor)
            if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 =>
        {
            crate::builtins::common::tensor::tensor_value_f64(tensor, 0)
        }
        _ => {
            return Err(invalid_argument(format!(
                "arrayDatastore: {property} must be a positive double integer"
            )))
        }
    };
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return Err(invalid_argument(format!(
            "arrayDatastore: {property} must be a positive double integer"
        )));
    }
    Ok(value as usize)
}

#[runtime_builtin(
    name = "fileDatastore",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn file_datastore_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::FILE_DATASTORE_DESCRIPTOR,
        &runmat_builtins::TABLE_ERROR_INVALID_ARGUMENT,
        "fileDatastore",
    )?;
    ensure_table_class_registered();
    file_datastore::construct(runtime.as_ref(), args).await
}

#[runtime_builtin(
    name = "parquetDatastore",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquet_datastore_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_native_integer_class)
    {
        return Err(invalid_argument(
            "parquetDatastore: typed-integer constructor arguments are not supported",
        ));
    }
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    let Some(files) = args.first().cloned() else {
        return Err(invalid_argument(
            "parquetDatastore: files location is required",
        ));
    };
    if (args.len() - 1) % 2 != 0 {
        return Err(invalid_argument(
            "parquetDatastore: name-value options must be provided in pairs",
        ));
    }
    let mut read_size = Value::from("rowgroup");
    let mut block_size = Value::Num(128_000_000.0);
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "parquetDatastore option")?;
        let value = args[idx + 1].clone();
        if name.eq_ignore_ascii_case("ReadSize") {
            read_size = match &value {
                Value::String(runmat_value::StringElement::Text(text))
                    if text.eq_ignore_ascii_case("file")
                        || text.eq_ignore_ascii_case("rowgroup") =>
                {
                    value
                }
                Value::CharArray(_) | Value::StringArray(_) => {
                    let text = scalar_text(&value, "ReadSize")?;
                    if !text.eq_ignore_ascii_case("file") && !text.eq_ignore_ascii_case("rowgroup")
                    {
                        return Err(invalid_argument("parquetDatastore: ReadSize must be 'file', 'rowgroup', or a positive double integer"));
                    }
                    Value::string(text)
                }
                _ => {
                    Value::Num(array_datastore_positive_double_integer(&value, "ReadSize")? as f64)
                }
            };
        } else if name.eq_ignore_ascii_case("BlockSize") {
            block_size =
                Value::Num(array_datastore_positive_double_integer(&value, "BlockSize")? as f64);
        } else {
            return Err(invalid_argument(format!(
                "parquetDatastore: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    let fields = [
        ("Files", files),
        ("ReadSize", read_size),
        ("BlockSize", block_size),
    ];
    construct_descriptor_object(runtime.as_ref(), PARQUET_DATASTORE_CLASS, &fields)
}

fn construct_descriptor_object(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: runmat_types::StaticClassIdentity,
    fields: &[(&str, Value)],
) -> BuiltinResult<Value> {
    let output = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class_name.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(output.into_value_and_lease().0)
}

#[runtime_builtin(
    name = "uitable",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn uitable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &UITABLE_EXPLICIT_GPU_EXTENSION,
            "uitable",
        )?;
    }
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    let data = parse_named_option(args.view(), "Data")
        .cloned()
        .or_else(|| args.first().cloned())
        .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()));
    let fields = [
        ("Data", data),
        (
            "ColumnName",
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
        ),
        (
            "RowName",
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
        ),
    ];
    construct_descriptor_object(runtime.as_ref(), UITABLE_CLASS, &fields)
}
