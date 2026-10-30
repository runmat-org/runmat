use runmat_meshing_core::{ElementOrder, MeshingRequestSettings};
use runmat_value::Value;

use super::{error, GENERATE_NAME, SAVE_NAME};
use crate::BuiltinResult;

pub(super) fn parse_settings(args: &[Value]) -> BuiltinResult<MeshingRequestSettings> {
    if !args.len().is_multiple_of(2) {
        return Err(error(GENERATE_NAME, "options must be Name, Value pairs"));
    }
    let mut settings = MeshingRequestSettings::default();
    for pair in args.chunks(2) {
        let name = text(GENERATE_NAME, &pair[0])?
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        match name.as_str() {
            "targetsize" => {
                settings.target_edge_length_m =
                    runmat_types::physical::Length::new(finite_positive(GENERATE_NAME, &pair[1])?)
            }
            "maximumdeviation" => {
                settings.maximum_chordal_deviation_m =
                    runmat_types::physical::Length::new(finite_positive(GENERATE_NAME, &pair[1])?)
            }
            "maximumgradingratio" => {
                settings.maximum_grading_ratio = finite_positive(GENERATE_NAME, &pair[1])?
            }
            "elementorder" => {
                settings.element_order =
                    match text(GENERATE_NAME, &pair[1])?.to_ascii_lowercase().as_str() {
                        "tet4" => ElementOrder::Tet4,
                        "tet10" => ElementOrder::Tet10,
                        other => {
                            return Err(error(
                                GENERATE_NAME,
                                format!("unsupported element order `{other}`"),
                            ))
                        }
                    }
            }
            "maximumelements" => {
                settings.limits.maximum_elements = exact_positive_u64(GENERATE_NAME, &pair[1])?
            }
            "seed" => settings.deterministic_seed = exact_u64(GENERATE_NAME, &pair[1])?,
            other => {
                return Err(error(
                    GENERATE_NAME,
                    format!("unsupported option `{other}`"),
                ))
            }
        }
    }
    Ok(settings)
}

#[cfg(feature = "plot-core")]
pub(super) fn parse_display_options(
    args: &[Value],
) -> BuiltinResult<crate::meshing::SolverMeshFigureOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(error("mesh", "display options must be Name, Value pairs"));
    }
    let mut options = crate::meshing::SolverMeshFigureOptions {
        show_edges: true,
        ..Default::default()
    };
    for pair in args.chunks(2) {
        match text("mesh", &pair[0])?.to_ascii_lowercase().as_str() {
            "edges" => {
                options.show_edges = match pair[1] {
                    Value::Bool(value) => value,
                    _ => return Err(error("mesh", "Edges must be logical")),
                }
            }
            "regions" => options.regions = Some(text_set("mesh", &pair[1])?),
            "section" => options.section_plane = Some(section_plane(&pair[1])?),
            other => {
                return Err(error(
                    "mesh",
                    format!("unsupported solver-mesh display option `{other}`"),
                ))
            }
        }
    }
    Ok(options)
}

#[cfg(feature = "plot-core")]
fn text_set(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<std::collections::BTreeSet<String>> {
    let values = match value {
        Value::String(_) | Value::CharArray(_) => vec![text(builtin, value)?],
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|value| text(builtin, &Value::String(value.clone())))
            .collect::<BuiltinResult<Vec<_>>>()?,
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|value| text(builtin, value))
            .collect::<BuiltinResult<Vec<_>>>()?,
        _ => return Err(error(builtin, "Regions must be text or a text array")),
    };
    if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) {
        return Err(error(
            builtin,
            "Regions must contain at least one nonempty id",
        ));
    }
    Ok(values.into_iter().collect())
}

#[cfg(feature = "plot-core")]
fn section_plane(value: &Value) -> BuiltinResult<[f64; 4]> {
    let Value::Tensor(tensor) = value else {
        return Err(error(
            "mesh",
            "Section must be a four-element numeric vector [a b c d]",
        ));
    };
    let plane: [f64; 4] = tensor.materialize_f64().try_into().map_err(|_| {
        error(
            "mesh",
            "Section must be a four-element numeric vector [a b c d]",
        )
    })?;
    if plane.iter().any(|value| !value.is_finite()) || plane[..3].iter().all(|value| *value == 0.0)
    {
        return Err(error("mesh", "Section must have a finite nonzero normal"));
    }
    Ok(plane)
}

pub(super) fn parse_force_option(args: &[Value]) -> BuiltinResult<bool> {
    if !args.len().is_multiple_of(2) {
        return Err(error(SAVE_NAME, "save options must be Name, Value pairs"));
    }
    let mut force = false;
    let mut seen = false;
    for pair in args.chunks(2) {
        if !text(SAVE_NAME, &pair[0])?.eq_ignore_ascii_case("force") || seen {
            return Err(error(SAVE_NAME, "expected at most one Force option"));
        }
        let Value::Bool(value) = pair[1] else {
            return Err(error(SAVE_NAME, "Force must be logical"));
        };
        force = value;
        seen = true;
    }
    Ok(force)
}

pub(super) fn text(builtin: &'static str, value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => Ok(value.to_string()),
        Value::CharArray(value) if value.rows == 1 => Ok(runmat_value::decode_utf16_code_units(
            value.data(),
        )
        .map_err(|source| {
            let mut error = error(builtin, source.to_string());
            error.source = Some(Box::new(source));
            error
        })?),
        _ => Err(error(builtin, "expected a text scalar")),
    }
}

fn finite_positive(builtin: &'static str, value: &Value) -> BuiltinResult<f64> {
    let number = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        _ => return Err(error(builtin, "expected a numeric scalar")),
    };
    if !number.is_finite() || number <= 0.0 {
        return Err(error(builtin, "expected a finite positive scalar"));
    }
    Ok(number)
}

fn exact_u64(builtin: &'static str, value: &Value) -> BuiltinResult<u64> {
    match value {
        Value::Int(value) => value
            .try_to_u64()
            .ok_or_else(|| error(builtin, "expected a nonnegative integer scalar")),
        Value::Num(value)
            if value.is_finite()
                && *value >= 0.0
                && value.fract() == 0.0
                && *value < u64::MAX as f64 =>
        {
            Ok(*value as u64)
        }
        _ => Err(error(builtin, "expected a nonnegative integer scalar")),
    }
}

fn exact_positive_u64(builtin: &'static str, value: &Value) -> BuiltinResult<u64> {
    let value = exact_u64(builtin, value)?;
    if value == 0 {
        return Err(error(builtin, "expected a positive integer scalar"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_controls_preserve_wide_integers() {
        let settings = parse_settings(&[
            Value::String("MaximumElements".to_string().into()),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            Value::String("Seed".to_string().into()),
            Value::Int(runmat_value::IntValue::U64(42)),
        ])
        .expect("valid generation controls");
        assert_eq!(settings.limits.maximum_elements, u64::MAX);
        assert_eq!(settings.deterministic_seed, 42);
    }

    #[test]
    fn generation_controls_reject_negative_integers() {
        let error = parse_settings(&[
            Value::String("Seed".to_string().into()),
            Value::Int(runmat_value::IntValue::I64(-1)),
        ])
        .expect_err("negative seed");
        assert_eq!(error.identifier(), Some("RunMat:meshing:InvalidInput"));
    }

    #[test]
    fn generation_controls_reject_double_above_the_u64_domain() {
        let error = parse_settings(&[
            Value::String("MaximumElements".to_string().into()),
            Value::Num(u64::MAX as f64),
        ])
        .expect_err("2^64 is outside the u64 domain");
        assert_eq!(error.identifier(), Some("RunMat:meshing:InvalidInput"));
    }
}
