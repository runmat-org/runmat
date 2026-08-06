use super::*;

pub(super) fn generated_variable_names(count: usize) -> Vec<String> {
    (1..=count).map(|idx| format!("Var{idx}")).collect()
}

pub(super) fn make_unique_variable_names(names: Vec<String>) -> Vec<String> {
    make_unique_names(
        names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| make_valid_variable_name(&name, idx + 1))
            .collect(),
    )
}

pub(super) fn validate_variable_names(names: &[String]) -> BuiltinResult<()> {
    if names.is_empty() {
        return Err(invalid_variable("table: variable names must not be empty"));
    }
    let mut used = HashSet::new();
    for name in names {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(invalid_variable("table: variable names must not be blank"));
        }
        let valid = trimmed == name && make_valid_variable_name(name, 1) == name.as_str();
        if !valid {
            return Err(invalid_variable(format!(
                "table: invalid variable name '{name}'"
            )));
        }
        let key = name.to_ascii_lowercase();
        if !used.insert(key) {
            return Err(invalid_variable(format!(
                "table: duplicate variable name '{name}'"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_array2table_names(
    variable_names: &[String],
    row_names: &mut Option<Vec<String>>,
    dimension_names: Option<&[String]>,
) -> BuiltinResult<()> {
    let default_dimension_names = [
        DEFAULT_ROW_DIM_NAME.to_string(),
        DEFAULT_VARIABLE_DIM_NAME.to_string(),
    ];
    let effective_dimension_names = dimension_names.unwrap_or(&default_dimension_names);
    if effective_dimension_names.len() != 2 {
        return Err(invalid_variable(
            "array2table: DimensionNames must contain exactly two names",
        ));
    }
    validate_nonempty_distinct_names(effective_dimension_names, "dimension")?;
    for name in effective_dimension_names {
        if is_reserved_table_name(name) {
            return Err(invalid_variable(format!(
                "array2table: reserved dimension name '{name}'"
            )));
        }
    }

    validate_nonempty_distinct_names(variable_names, "variable")?;
    for name in variable_names {
        if is_reserved_table_name(name)
            || effective_dimension_names
                .iter()
                .any(|dimension_name| dimension_name == name)
        {
            return Err(invalid_variable(format!(
                "array2table: reserved variable name '{name}'"
            )));
        }
    }

    if let Some(names) = row_names {
        for name in names.iter_mut() {
            *name = name.trim().to_string();
        }
        validate_nonempty_distinct_names(names, "row")?;
        if let Some(name) = names.iter().find(|name| name.contains(':')) {
            return Err(invalid_variable(format!(
                "array2table: row name '{name}' must not contain a colon"
            )));
        }
    }
    Ok(())
}

fn validate_nonempty_distinct_names(names: &[String], kind: &str) -> BuiltinResult<()> {
    let mut used = HashSet::new();
    for name in names {
        if name.trim().is_empty() {
            return Err(invalid_variable(format!(
                "array2table: {kind} names must not be empty"
            )));
        }
        if !used.insert(name.as_str()) {
            return Err(invalid_variable(format!(
                "array2table: duplicate {kind} name '{name}'"
            )));
        }
    }
    Ok(())
}

fn is_reserved_table_name(name: &str) -> bool {
    matches!(
        name,
        PROPERTIES_MEMBER | VARIABLE_NAMES | ROW_NAMES | DIMENSION_NAMES | ":"
    )
}

pub(super) fn validate_unique_names(names: &[String]) -> BuiltinResult<()> {
    let mut used = HashSet::new();
    for name in names {
        if !used.insert(name.as_str()) {
            return Err(invalid_variable(format!(
                "table: duplicate variable name '{name}'"
            )));
        }
    }
    Ok(())
}

pub(super) fn make_unique_names(names: Vec<String>) -> Vec<String> {
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(names.len());
    for (idx, name) in names.into_iter().enumerate() {
        let base = if name.trim().is_empty() {
            format!("Var{}", idx + 1)
        } else {
            name.trim().to_string()
        };
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while used.contains(&candidate.to_ascii_lowercase()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        used.insert(candidate.to_ascii_lowercase());
        out.push(candidate);
    }
    out
}

pub(super) fn make_valid_variable_name(raw: &str, fallback_index: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in raw.trim().chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() || !out.chars().next().unwrap().is_ascii_alphabetic() {
        format!("Var{fallback_index}")
    } else {
        out
    }
}
