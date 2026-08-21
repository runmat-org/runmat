use super::*;

pub(in crate::builtins::table) async fn gather_values(
    values: &[Value],
) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
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

pub(in crate::builtins::table) fn split_table_constructor_args(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, TableConstructorOptions)> {
    let mut variables = Vec::new();
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if let Ok(name) = scalar_text(&args[idx], "table option") {
            if idx + 1 < args.len() && is_table_constructor_option(&name) {
                let value = &args[idx + 1];
                if name.eq_ignore_ascii_case("VariableNames") {
                    options.variable_names = Some(variable_name_list(value)?);
                } else if name.eq_ignore_ascii_case("RowNames") {
                    options.row_names = Some(string_list(value)?);
                } else if name.eq_ignore_ascii_case("Size") {
                    options.size = Some(value.clone());
                } else if name.eq_ignore_ascii_case("VariableTypes") {
                    options.variable_types = Some(string_list(value)?);
                }
                idx += 2;
                continue;
            }
        }
        variables.push(args[idx].clone());
        idx += 1;
    }
    Ok((variables, options))
}

pub(in crate::builtins::table) fn is_table_constructor_option(name: &str) -> bool {
    name.eq_ignore_ascii_case("VariableNames")
        || name.eq_ignore_ascii_case("RowNames")
        || name.eq_ignore_ascii_case("Size")
        || name.eq_ignore_ascii_case("VariableTypes")
}

pub(in crate::builtins::table) fn parse_table_options(
    args: &[Value],
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
    args: &[Value],
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
    args: &[Value],
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
    args: &[Value],
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

pub(in crate::builtins::table) fn split_readtimetable_options(
    args: &[Value],
) -> BuiltinResult<(Vec<Value>, Vec<Value>)> {
    let mut readtable_args = Vec::new();
    let mut timetable_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "readtimetable: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "readtimetable option")?;
        if name.eq_ignore_ascii_case(ROW_TIMES) {
            timetable_args.push(args[idx].clone());
            timetable_args.push(args[idx + 1].clone());
        } else {
            readtable_args.push(args[idx].clone());
            readtable_args.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok((readtable_args, timetable_args))
}

pub(in crate::builtins::table) fn parse_named_option<'a>(
    args: &'a [Value],
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

pub(in crate::builtins::table) fn parse_bool_option(
    args: &[Value],
    name: &str,
    default: bool,
    context: &str,
) -> BuiltinResult<bool> {
    let mut result = default;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if option_name.eq_ignore_ascii_case(name) {
            result = bool_scalar(&args[idx + 1], name)?;
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(result)
}

pub(in crate::builtins::table) fn parse_named_text_option(
    args: &[Value],
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
