use super::*;
use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_macros::runtime_builtin;

pub use runmat_builtins::DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION;

#[cfg(test)]
pub use runmat_builtins::DETECT_IMPORT_OPTIONS_EXTENSIONS;

#[cfg(test)]
pub use runmat_builtins::DETECT_IMPORT_OPTIONS_INTEGER_CAPABILITIES;

pub use runmat_builtins::READCELL_TYPED_INTEGER_CONTROL_EXTENSION;
pub use runmat_builtins::READTABLE_TYPED_INTEGER_CONTROL_EXTENSION;
#[cfg(test)]
pub use runmat_builtins::SPREADSHEET_IMPORT_OPTIONS_EXTENSIONS;
pub use runmat_builtins::SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION;

mod writers;
#[cfg(test)]
use runmat_builtins::{
    WRITETABLE_BYTES_OUTPUT_EXTENSION, WRITETABLE_INTEGER_LOGICAL_EXTENSION,
    WRITETIMETABLE_BYTES_OUTPUT_EXTENSION,
};
#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) use writers::*;

#[cfg(test)]
pub use runmat_builtins::SPREADSHEET_IMPORT_OPTIONS_INTEGER_CAPABILITIES;

#[runtime_builtin(
    name = "readtable",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtable_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    enforce_table_import_integer_control_gate(
        rest.view(),
        &READTABLE_TYPED_INTEGER_CONTROL_EXTENSION,
        "readtable",
        &["VariableNamesLine"],
    )?;
    let path_value = gather_retained_value(&path, runtime.as_ref()).await?;
    let args = gather_retained_values(&rest, runtime.as_ref()).await?;
    let options = ReadTableOptions::parse(runtime.as_ref(), args.view())?;
    let resolved = resolve_path(&path_value)?;
    read_table_from_file(runtime.as_ref(), &resolved, &options).await
}

#[runtime_builtin(
    name = "parquetread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquetread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PARQUETREAD_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "parquetread",
    )?;
    ensure_table_class_registered();
    let resolved = resolve_path(&path)?;
    let options = ParquetReadOptions::parse(rest.view()).await?;
    read_parquet_table(runtime.as_ref(), &resolved, &options).await
}

#[runtime_builtin(
    name = "parquetinfo",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquetinfo_builtin(path: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::PARQUETINFO_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "parquetinfo",
    )?;
    let resolved = resolve_path(&path)?;
    parquet_file_info(&resolved).await
}

#[runtime_builtin(
    name = "spreadsheetImportOptions",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn spreadsheet_import_options_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    enforce_table_import_integer_control_gate(
        args.view(),
        &SPREADSHEET_OPTIONS_TYPED_INTEGER_CONTROL_EXTENSION,
        "spreadsheetImportOptions",
        &["NumVariables"],
    )?;
    let gathered = gather_retained_values(&args, runtime.as_ref()).await?;
    let Value::Struct(options) = spreadsheet_import_options(gathered.view())? else {
        unreachable!("spreadsheet import constructor returns options fields")
    };
    crate::builtins::io::tabular::import_options::ImportOptions::from_legacy(options)?
        .to_value(runtime.as_ref())
}

#[runtime_builtin(
    name = "detectImportOptions",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn detect_import_options_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::value_contains_gpu(&path) || rest.iter().any(crate::value_contains_gpu) {
        return Err(invalid_argument(
            "detectImportOptions: resident arguments are not supported",
        ));
    }
    if detect_option_has_typed_integer(rest.view(), "NumHeaderLines") {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DETECT_IMPORT_OPTIONS_INTEGER_NUM_HEADER_LINES_EXTENSION,
            "detectImportOptions",
        )?;
    }
    let path_value = gather_retained_value(&path, runtime.as_ref()).await?;
    let args = gather_retained_values(&rest, runtime.as_ref()).await?;
    let options = ReadTableOptions::parse(runtime.as_ref(), args.view())?;
    let resolved = resolve_path(&path_value)?;
    let Value::Struct(detected) = detect_import_options_from_file(&resolved, &options).await?
    else {
        unreachable!("detected import constructor returns options fields")
    };
    let origin = options.detected_variable_source_origin();
    crate::builtins::io::tabular::import_options::ImportOptions::from_detected(detected, origin)?
        .to_value(runtime.as_ref())
}

#[runtime_builtin(
    name = "setvaropts",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn setvaropts_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    use crate::builtins::io::tabular::import_options::ImportOptions;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SETVAROPTS_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "setvaropts",
    )?;
    if args.iter().any(crate::value_contains_gpu) {
        return Err(invalid_argument(
            "setvaropts: resident arguments are not supported",
        ));
    }
    if args.len() < 3 {
        return Err(invalid_argument(
            "setvaropts: expected options and name-value pairs",
        ));
    }
    let source = &args[0];
    if !matches!(source, Value::Object(_)) {
        return Err(invalid_argument(
            "setvaropts: expected a typed import-options object",
        ));
    }
    let mut options = ImportOptions::decode(runtime.as_ref(), source)?
        .ok_or_else(|| invalid_argument("setvaropts: expected an import-options object"))?;
    let remaining = args.view().range(1..)?;
    let (selection, pairs) = if remaining.len().is_multiple_of(2) {
        (None, remaining)
    } else {
        (remaining.first(), remaining.range(1..)?)
    };
    if pairs.is_empty() || !pairs.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "setvaropts: name-value options must be paired",
        ));
    }
    let pairs = pairs
        .chunks_exact(2)
        .map(|pair| Ok((scalar_text(&pair[0], "setvaropts option")?, pair[1].clone())))
        .collect::<BuiltinResult<Vec<_>>>()?;
    options.set_options(selection, &pairs)?;
    options.to_value(runtime.as_ref())
}

#[runtime_builtin(
    name = "setvartype",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn setvartype_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    use crate::builtins::io::tabular::import_options::ImportOptions;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SETVARTYPE_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "setvartype",
    )?;
    if args.iter().any(crate::value_contains_gpu) {
        return Err(invalid_argument(
            "setvartype: resident arguments are not supported",
        ));
    }
    let (3, Some(source), Some(selection), Some(types)) =
        (args.len(), args.first(), args.get(1), args.get(2))
    else {
        return Err(invalid_argument(
            "setvartype: expected options, selection, and type",
        ));
    };
    if !matches!(source, Value::Object(_)) {
        return Err(invalid_argument(
            "setvartype: expected a typed import-options object",
        ));
    }
    let mut options = ImportOptions::decode(runtime.as_ref(), source)?
        .ok_or_else(|| invalid_argument("setvartype: expected an import-options object"))?;
    options.set_types(runtime.as_ref(), selection, types)?;
    for variable in options.variables() {
        ImportVariableType::parse(&variable.type_name)?;
    }
    options.to_value(runtime.as_ref())
}

#[runtime_builtin(
    name = "getvaropts",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn getvaropts_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    use crate::builtins::io::tabular::import_options::ImportOptions;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::GETVAROPTS_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "getvaropts",
    )?;
    if args.iter().any(crate::value_contains_gpu) {
        return Err(invalid_argument(
            "getvaropts: resident arguments are not supported",
        ));
    }
    let (2, Some(source), Some(selection)) = (args.len(), args.first(), args.get(1)) else {
        return Err(invalid_argument(
            "getvaropts: expected options and selection",
        ));
    };
    if !matches!(source, Value::Object(_)) {
        return Err(invalid_argument(
            "getvaropts: expected a typed import-options object",
        ));
    }
    let options = ImportOptions::decode(runtime.as_ref(), source)?
        .ok_or_else(|| invalid_argument("getvaropts: expected an import-options object"))?;
    let selected = options.selected(selection)?;
    if selected.len() == 1 {
        return ImportOptions::variable_value(runtime.as_ref(), selected[0]);
    }
    let count = selected.len();
    let mut output = crate::execution::object_array::ObjectArrayConstruction::new(
        runtime.as_ref(),
        crate::execution::object_array::Class::Name(
            crate::builtins::io::tabular::import_options::VARIABLE_IMPORT_CLASS.display_name(),
        ),
        count,
        &[1, count],
        std::iter::empty::<&std::rc::Rc<runmat_value::SelectedObjectClassFacts>>(),
        true,
    )?;
    for variable in selected {
        output.push(ImportOptions::variable_value(runtime.as_ref(), variable)?)?;
    }
    output.finish_value()
}

fn detect_option_has_typed_integer(
    args: crate::call::FunctionArgumentView<'_>,
    sought: &str,
) -> bool {
    if let Some(Value::Struct(options)) = args.first() {
        if options
            .fields
            .iter()
            .any(|(name, value)| name.eq_ignore_ascii_case(sought) && is_typed_integer(value))
        {
            return true;
        }
    }
    (0..args.len().saturating_sub(1)).any(|index| {
        scalar_text(&args[index], "detectImportOptions option")
            .is_ok_and(|name| name.eq_ignore_ascii_case(sought))
            && is_typed_integer(&args[index + 1])
    })
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

#[runtime_builtin(
    name = "readcell",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readcell_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    enforce_table_import_integer_control_gate(
        rest.view(),
        &READCELL_TYPED_INTEGER_CONTROL_EXTENSION,
        "readcell",
        &["NumHeaderLines"],
    )?;
    let path = gather_retained_value(&path, runtime.as_ref()).await?;
    let rest = gather_retained_values(&rest, runtime.as_ref()).await?;
    let path = resolve_path(&path)?;
    let options = ReadTableOptions::parse(runtime.as_ref(), rest.view())?;
    if options.has_variable_value_controls() {
        return Err(invalid_argument(
            "readcell: per-variable import options are not supported by the current cell reader",
        ));
    }
    read_cell_from_file(&path, &options).await
}

pub(crate) fn enforce_table_import_integer_control_gate(
    args: crate::call::FunctionArgumentView<'_>,
    extension: &'static BuiltinExtensionDescriptor,
    builtin: &'static str,
    documented_integer_options: &[&str],
) -> BuiltinResult<()> {
    let is_extension = |name: &str| {
        !documented_integer_options
            .iter()
            .any(|allowed| name.eq_ignore_ascii_case(allowed))
    };
    let struct_extension = matches!(args.first(), Some(Value::Struct(options)) if
        options.fields.iter().any(|(name, value)|
            value_contains_typed_integer(value) && is_extension(name)));
    let named_extension = (0..args.len().saturating_sub(1)).any(|index| {
        value_contains_typed_integer(&args[index + 1])
            && scalar_text(&args[index], "import option").is_ok_and(|name| is_extension(&name))
    });
    if struct_extension || named_extension {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

fn value_contains_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
        || matches!(value, Value::Cell(cell) if cell.data().iter().any(value_contains_typed_integer))
        || matches!(value, Value::Struct(structure) if structure.fields.values().any(value_contains_typed_integer))
}

#[cfg(test)]
mod write_extension_tests {
    use super::*;

    #[test]
    fn writetable_gates_integer_logical_option_before_file_access() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = futures::executor::block_on(writetable_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("definitely/missing/out.csv"),
                Value::from("WriteVariableNames"),
                Value::Int(runmat_value::IntValue::U8(1)),
            ]),
        ))
        .expect_err("strict mode rejects typed integer logical aliases");
        assert_eq!(
            error.identifier(),
            WRITETABLE_INTEGER_LOGICAL_EXTENSION.error_identifier
        );
        drop(strict);
    }

    #[test]
    fn table_writers_gate_bytes_outputs_before_file_access() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let table_error = futures::executor::block_on(writetable_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "definitely/missing/table.csv",
            )]),
        ))
        .expect_err("strict mode rejects writetable bytes output");
        assert_eq!(
            table_error.identifier(),
            WRITETABLE_BYTES_OUTPUT_EXTENSION.error_identifier
        );
        let timetable_error = futures::executor::block_on(writetimetable_builtin(
            None,
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "definitely/missing/timetable.csv",
            )]),
        ))
        .expect_err("strict mode rejects writetimetable bytes output");
        assert_eq!(
            timetable_error.identifier(),
            WRITETIMETABLE_BYTES_OUTPUT_EXTENSION.error_identifier
        );
        drop(outputs);
        drop(strict);
    }
}
