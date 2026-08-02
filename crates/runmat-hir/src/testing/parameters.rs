use runmat_test::descriptor::ParameterDescriptor;
use std::collections::BTreeMap;

use runmat_test::discovery::{
    MaterializationKind, MaterializationLimits, MaterializationRequest, MaterializationResponse,
};
use runmat_test::identity::ParameterId;
use runmat_test::plan::ProgramRevision;
use serde_json::{Number, Value};

use crate::{ClassProperty, HirExpr, HirExprKind, OperatorKind};

use super::{source::source_descriptor, SemanticTestSource};

pub(super) fn class_parameter_sets(
    source: &SemanticTestSource<'_>,
    class_name: &str,
    properties: &[ClassProperty],
    revision: &ProgramRevision,
    materialized: &BTreeMap<&str, &MaterializationResponse>,
) -> (Vec<Vec<ParameterDescriptor>>, Vec<MaterializationRequest>) {
    let mut sets = vec![Vec::new()];
    let mut pending = Vec::new();
    for property in properties.iter().filter(|property| {
        super::attributes::has(&property.declared_attributes, "TestParameter")
            || super::attributes::has(&property.declared_attributes, "MethodSetupParameter")
            || super::attributes::has(&property.declared_attributes, "ClassSetupParameter")
    }) {
        let is_test_parameter =
            super::attributes::has(&property.declared_attributes, "TestParameter");
        let path_kind = if is_test_parameter {
            "parameter"
        } else {
            "fixture-parameter"
        };
        let semantic_path = format!("{class_name}/{path_kind}/{}", property.name.0);
        let static_values = property.default.as_ref().and_then(static_parameter_values);
        let values = if let Some(values) = static_values {
            values
        } else {
            let expression = &source.source_text[property.span.start.min(source.source_text.len())
                ..property.span.end.min(source.source_text.len())];
            let id = ParameterId::derive(&semantic_path, expression)
                .as_str()
                .to_owned();
            if let Some(response) = materialized.get(id.as_str()) {
                response
                    .values
                    .iter()
                    .map(|value| value.value.clone())
                    .collect()
            } else {
                pending.push(MaterializationRequest {
                    id,
                    program_revision: revision.clone(),
                    kind: if is_test_parameter {
                        MaterializationKind::ParameterExpression
                    } else {
                        MaterializationKind::FixtureExpression
                    },
                    semantic_path: semantic_path.clone(),
                    source: source_descriptor(source, semantic_path, property.span),
                    expression: expression.to_owned(),
                    limits: MaterializationLimits::default(),
                });
                continue;
            }
        };
        let parameters = values
            .into_iter()
            .map(|value| {
                let normalized_identity =
                    serde_json::to_string(&value).expect("JSON value is serializable");
                ParameterDescriptor {
                    id: ParameterId::derive(&property.name.0, &normalized_identity),
                    name: property.name.0.clone(),
                    normalized_identity,
                    value,
                }
            })
            .collect::<Vec<_>>();
        let mut expanded = Vec::new();
        for existing in &sets {
            for parameter in &parameters {
                let mut next = existing.clone();
                next.push(parameter.clone());
                expanded.push(next);
            }
        }
        sets = expanded;
    }
    (sets, pending)
}

fn static_parameter_values(expression: &HirExpr) -> Option<Vec<Value>> {
    if let HirExprKind::Cell(rows) = &expression.kind {
        return rows
            .iter()
            .flat_map(|row| row.iter())
            .map(static_value)
            .collect();
    }
    static_value(expression).map(|value| vec![value])
}

fn static_value(expression: &HirExpr) -> Option<Value> {
    match &expression.kind {
        HirExprKind::Number(value) => value
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map(Value::Number),
        HirExprKind::String(value) => Some(Value::String(unquote(&value.0))),
        HirExprKind::Constant(value) if value.0.eq_ignore_ascii_case("true") => {
            Some(Value::Bool(true))
        }
        HirExprKind::Constant(value) if value.0.eq_ignore_ascii_case("false") => {
            Some(Value::Bool(false))
        }
        HirExprKind::Unary(OperatorKind::UnaryMinus, inner) => {
            let Value::Number(number) = static_value(inner)? else {
                return None;
            };
            Number::from_f64(-number.as_f64()?).map(Value::Number)
        }
        HirExprKind::Tensor(rows) => rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(static_value)
                    .collect::<Option<Vec<_>>>()
                    .map(Value::Array)
            })
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        _ => None,
    }
}

fn unquote(value: &str) -> String {
    if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
        value[1..value.len() - 1].replace("''", "'")
    } else if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
        value[1..value.len() - 1].replace("\"\"", "\"")
    } else {
        value.to_owned()
    }
}
