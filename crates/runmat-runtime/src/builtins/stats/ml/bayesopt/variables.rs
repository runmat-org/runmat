use super::*;
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub(super) struct Variable {
    pub(super) name: String,
    pub(super) kind: VariableKind,
    pub(super) transform: Transform,
    pub(super) optimize: bool,
}

#[derive(Clone, Debug)]
pub(super) enum VariableKind {
    Real { lower: f64, upper: f64 },
    Integer { lower: i64, upper: i64 },
    Categorical { categories: Vec<String> },
}

pub(super) fn parse_variables(value: Value) -> BuiltinResult<Vec<Variable>> {
    let raw = match &value {
        Value::Object(object) if object.class_name().is(OPTIMIZABLE_VARIABLE_CLASS) => {
            std::slice::from_ref(&value)
        }
        Value::Cell(cell) => cell.data(),
        Value::OutputList(values) => values.as_slice(),
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: variables must be optimizableVariable objects, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if raw.is_empty() {
        return Err(bayesopt_error(
            "bayesopt: variables must not be empty",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut names = HashSet::with_capacity(raw.len());
    let mut vars = Vec::with_capacity(raw.len());
    for value in raw.iter() {
        let Value::Object(object) = value else {
            return Err(bayesopt_error(
                "bayesopt: variables must be optimizableVariable objects",
                &ERROR_INVALID_ARGUMENT,
            ));
        };
        if !object.class_name().is(OPTIMIZABLE_VARIABLE_CLASS) {
            return Err(bayesopt_error(
                format!(
                    "bayesopt: expected optimizableVariable, got {}",
                    object.class_name()
                ),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let variable = variable_from_object(object)?;
        if !names.insert(variable.name.clone()) {
            return Err(bayesopt_error(
                format!("bayesopt: duplicate variable name '{}'", variable.name),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        vars.push(variable);
    }
    if vars.iter().all(|var| !var.optimize) {
        return Err(bayesopt_error(
            "bayesopt: at least one variable must have Optimize=true",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(vars)
}

fn variable_from_object(object: &ObjectInstance) -> BuiltinResult<Variable> {
    let name = string_property(object, "Name")?;
    let public_type = string_property(object, "Type")?;
    let var_type = VariableType::from_public_name(&public_type).ok_or_else(|| {
        bayesopt_error(
            format!("bayesopt: unsupported variable Type '{public_type}'"),
            &ERROR_INVALID_ARGUMENT,
        )
    })?;
    let public_transform = string_property(object, "Transform")?;
    let transform = Transform::from_public_name(&public_transform).ok_or_else(|| {
        bayesopt_error(
            format!("bayesopt: unsupported Transform '{public_transform}'"),
            &ERROR_INVALID_ARGUMENT,
        )
    })?;
    let optimize = match object.properties().get("Optimize") {
        Some(Value::Bool(value)) => *value,
        Some(other) => {
            return Err(bayesopt_error(
                format!("bayesopt: Optimize must be logical scalar, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
        None => true,
    };
    let range = object.properties().get("Range").ok_or_else(|| {
        bayesopt_error(
            format!("bayesopt: variable '{name}' is missing Range"),
            &ERROR_INVALID_ARGUMENT,
        )
    })?;
    let kind = match var_type {
        VariableType::Real => {
            let (lower, upper) = numeric_range(range, &name)?;
            if transform == Transform::Log && lower <= 0.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: log transform for '{name}' requires positive bounds"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            VariableKind::Real { lower, upper }
        }
        VariableType::Integer => {
            let (lower, upper) = numeric_range(range, &name)?;
            if lower.fract() != 0.0 || upper.fract() != 0.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: integer variable '{name}' must have whole-number bounds"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            if transform == Transform::Log && lower < 0.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: log transform for integer variable '{name}' requires nonnegative bounds"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            if lower < i64::MIN as f64 || upper >= 9_223_372_036_854_775_808.0 {
                return Err(bayesopt_error(
                    format!("bayesopt: integer variable '{name}' exceeds the supported signed integer range"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            VariableKind::Integer {
                lower: lower as i64,
                upper: upper as i64,
            }
        }
        VariableType::Categorical => {
            if transform != Transform::None {
                return Err(bayesopt_error(
                    format!(
                        "bayesopt: categorical variable '{name}' only supports Transform='none'"
                    ),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            VariableKind::Categorical {
                categories: categorical_range(range, &name)?,
            }
        }
    };
    Ok(Variable {
        name: name.into_owned(),
        kind,
        transform,
        optimize,
    })
}

fn string_property<'a>(object: &'a ObjectInstance, name: &str) -> BuiltinResult<Cow<'a, str>> {
    match object.properties().get(name) {
        Some(value) => value_to_text(value, name),
        None => Err(bayesopt_error(
            format!("bayesopt: optimizableVariable is missing {name}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn numeric_range(value: &Value, name: &str) -> BuiltinResult<(f64, f64)> {
    if !native_integer_value_is_exact_f64(value) {
        return Err(bayesopt_error(
            format!("bayesopt: integer Range for '{name}' must be exactly representable as double"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let values = match value {
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor),
        Value::Num(n) => vec![*n],
        Value::Int(i) => vec![i.to_f64()],
        other => {
            return Err(bayesopt_error(
                format!("bayesopt: Range for '{name}' must be numeric, got {other:?}"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if values.len() != 2
        || !values[0].is_finite()
        || !values[1].is_finite()
        || values[0] >= values[1]
    {
        return Err(bayesopt_error(
            format!("bayesopt: Range for '{name}' must contain increasing finite bounds"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok((values[0], values[1]))
}

fn categorical_range(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    let categories = match value {
        Value::String(runmat_value::StringElement::Text(text)) => vec![text.as_str().to_owned()],
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    bayesopt_error(
                        format!("bayesopt: categorical Range for '{name}' must be nonmissing text"),
                        &ERROR_INVALID_ARGUMENT,
                    )
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?,
        Value::CharArray(chars) if chars.rows == 1 => {
            vec![crate::builtins::stats::decode_utf16_with_error(
                chars.data(),
                || {
                    bayesopt_error(
                        format!(
                            "bayesopt: categorical Range for '{name}' must be valid UTF-16 text"
                        ),
                        &ERROR_INVALID_ARGUMENT,
                    )
                },
            )?]
        }
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data().len());
            for value in cell.data() {
                out.push(value_to_text(value, "Range")?.into_owned());
            }
            out
        }
        other => {
            return Err(bayesopt_error(
                format!(
                "bayesopt: categorical Range for '{name}' must contain text values, got {other:?}"
            ),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    if categories.is_empty() || categories.iter().any(|text| text.trim().is_empty()) {
        return Err(bayesopt_error(
            format!("bayesopt: categorical Range for '{name}' must contain nonempty text"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut seen = HashSet::with_capacity(categories.len());
    for category in &categories {
        if !seen.insert(category.as_str()) {
            return Err(bayesopt_error(
                format!("bayesopt: categorical Range for '{name}' contains duplicate '{category}'"),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
    }
    Ok(categories)
}
