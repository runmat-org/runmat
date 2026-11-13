use super::*;

#[cfg(test)]
#[path = "args/retained_tests.rs"]
mod retained_tests;

pub(in crate::builtins::table) async fn gather_retained_values(
    arguments: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let project = |index| async move {
        gather_if_needed_async(&arguments[index])
            .await
            .map_err(map_control_flow)
    };
    let gather = arguments.try_map_preserving_roots_async(arguments.len(), runtime, project);
    match runtime {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
}

pub(in crate::builtins::table) async fn gather_retained_value(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let gather = gather_if_needed_async(value);
    match runtime {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
    .map_err(map_control_flow)
}

#[derive(Default)]
pub(in crate::builtins::table) struct TableConstructorOptions {
    pub(in crate::builtins::table) variable_names: Option<Vec<String>>,
    pub(in crate::builtins::table) row_names: Option<Vec<String>>,
    pub(in crate::builtins::table) size: Option<Value>,
    pub(in crate::builtins::table) variable_types: Option<Vec<String>>,
}

#[derive(Default)]
pub(in crate::builtins::table) struct Array2TableOptions {
    pub(in crate::builtins::table) variable_names: Option<Vec<String>>,
    pub(in crate::builtins::table) row_names: Option<Vec<String>>,
    pub(in crate::builtins::table) dimension_names: Option<Vec<String>>,
}

pub(in crate::builtins::table) struct Struct2TableOptions {
    pub(in crate::builtins::table) table: TableConstructorOptions,
    pub(in crate::builtins::table) as_array: bool,
}

#[derive(Clone)]
pub(in crate::builtins::table) struct TableVariables<'a> {
    source: crate::call::FunctionArgumentView<'a>,
    position: usize,
}
impl<'a> Iterator for TableVariables<'a> {
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.source.len() {
            let index = self.position;
            if self.source.get(index + 1).is_some()
                && scalar_text(&self.source[index], "table option").is_ok_and(|name| {
                    runmat_builtins::TableConstructorOption::parse(&name).is_some()
                })
            {
                self.position += 2;
                continue;
            }
            self.position += 1;
            return self.source.get(index);
        }
        None
    }
}

pub(in crate::builtins::table) fn split_table_constructor_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(TableVariables<'_>, TableConstructorOptions)> {
    use runmat_builtins::TableConstructorOption;
    let mut options = TableConstructorOptions::default();
    let mut index = 0;
    while index < args.len() {
        if let Some(value) = args.get(index + 1) {
            if let Ok(name) = scalar_text(&args[index], "table option") {
                if let Some(key) = TableConstructorOption::parse(&name) {
                    match key {
                        TableConstructorOption::VariableNames => {
                            options.variable_names = Some(variable_name_list(value)?)
                        }
                        TableConstructorOption::RowNames => {
                            options.row_names = Some(string_list(value)?)
                        }
                        TableConstructorOption::Size => options.size = Some(value.clone()),
                        TableConstructorOption::VariableTypes => {
                            options.variable_types = Some(string_list(value)?)
                        }
                    }
                    index += 2;
                    continue;
                }
            }
        }
        index += 1;
    }
    Ok((
        TableVariables {
            source: args,
            position: 0,
        },
        options,
    ))
}

pub(in crate::builtins::table) fn parse_table_options(
    args: crate::call::FunctionArgumentView<'_>,
    context: &str,
) -> BuiltinResult<TableConstructorOptions> {
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let name = scalar_text(&args[idx], "table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            options.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            options.row_names = Some(string_list(&args[idx + 1])?);
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(options)
}

pub(in crate::builtins::table) fn parse_array2table_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Array2TableOptions> {
    let mut options = Array2TableOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "array2table: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "array2table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            options.variable_names = Some(raw_variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            options.row_names = Some(string_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("DimensionNames") {
            options.dimension_names = Some(string_list(&args[idx + 1])?);
        } else {
            return Err(invalid_argument(format!(
                "array2table: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(options)
}

pub(in crate::builtins::table) fn parse_struct2table_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Struct2TableOptions> {
    let mut table = TableConstructorOptions::default();
    let mut as_array = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "struct2table: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "struct2table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            table.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            table.row_names = Some(string_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("AsArray") {
            as_array = zero_one_bool_scalar(&args[idx + 1], "AsArray")?;
        } else {
            return Err(invalid_argument(format!(
                "struct2table: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(Struct2TableOptions { table, as_array })
}

pub(in crate::builtins::table) fn parse_table2struct_to_scalar(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<bool> {
    let mut to_scalar = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "table2struct: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "table2struct option")?;
        if name.eq_ignore_ascii_case("ToScalar") {
            to_scalar = bool_scalar(&args[idx + 1], "ToScalar")?;
        } else {
            return Err(invalid_argument(format!(
                "table2struct: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(to_scalar)
}

pub(in crate::builtins::table) async fn split_readtimetable_options(
    args: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(crate::call::BuiltinArguments, crate::call::BuiltinArguments)> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "readtimetable: name-value options must be provided in pairs",
        ));
    }
    let mut readtable_count = 0usize;
    for pair in args.chunks_exact(2) {
        let name = scalar_text(&pair[0], "readtimetable option")?;
        if !name.eq_ignore_ascii_case(ROW_TIMES) {
            readtable_count += 2;
        }
    }
    // Partition only the final headers. Each phase scans the immutable source
    // once, preserving pair order without an index scratch or per-pair owner.
    let mut cursor = 0usize;
    let mut value_slot = false;
    let mut selecting_table = true;
    let partitioned = args
        .try_map_preserving_roots_async(args.len(), runtime, |output| {
            if output == readtable_count {
                cursor = 0;
                value_slot = false;
                selecting_table = false;
            }
            let result = if value_slot {
                let value = args[cursor + 1].clone();
                cursor += 2;
                value_slot = false;
                Ok(value)
            } else {
                loop {
                    let name = match scalar_text(&args[cursor], "readtimetable option") {
                        Ok(name) => name,
                        Err(error) => break Err(error),
                    };
                    if name.eq_ignore_ascii_case(ROW_TIMES) != selecting_table {
                        value_slot = true;
                        break Ok(args[cursor].clone());
                    }
                    cursor += 2;
                }
            };
            std::future::ready(result)
        })
        .await?;
    Ok((
        partitioned.range(..readtable_count)?,
        partitioned.into_range(readtable_count..)?,
    ))
}

pub(in crate::builtins::table) fn parse_named_option<'a>(
    args: crate::call::FunctionArgumentView<'a>,
    name: &str,
) -> Option<&'a Value> {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if scalar_text(&args[idx], "option name")
            .map(|text| text.eq_ignore_ascii_case(name))
            .unwrap_or(false)
        {
            return args.get(idx + 1);
        }
        idx += 2;
    }
    None
}

pub(in crate::builtins::table) fn parse_named_text_option(
    args: crate::call::FunctionArgumentView<'_>,
    name: &str,
    default: &str,
    context: &str,
) -> BuiltinResult<String> {
    let mut result = default.to_string();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if option_name.eq_ignore_ascii_case(name) {
            result = scalar_text(&args[idx + 1], name)?;
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(result)
}
