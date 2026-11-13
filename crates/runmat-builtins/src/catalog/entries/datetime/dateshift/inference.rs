use super::policy::*;
use crate::{
    catalog::inference::{argument_error, finish_fixed, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-DATESHIFT-ARITY",
            "dateshift requires datetime, boundary, unit/weekday, and at most one rule",
            0,
        ));
    }
    let mut shape = match request.arguments.first() {
        Some(ValueFact {
            kind: ValueKindFact::Object(object),
            shape,
            ..
        }) => match object.runtime_class.as_ref() {
            Some(class) if class == &standard::DATETIME.owned() => shape.clone(),
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-DATESHIFT-CLASS",
                    "dateshift requires a datetime object",
                    0,
                ));
                ShapeFact::Unknown
            }
            None => ShapeFact::Unknown,
        },
        Some(value) if value.kind == ValueKindFact::Unknown => ShapeFact::Unknown,
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-DATESHIFT-INPUT",
                "dateshift does not convert relative text or numeric serials",
                0,
            ));
            ShapeFact::Unknown
        }
        None => ShapeFact::Unknown,
    };
    let boundary_text = request.literals.literal_string_at(1);
    let boundary = boundary_text
        .as_deref()
        .and_then(DateShiftBoundary::parse_text);
    if request.arguments.get(1).is_some_and(|v| !scalar_text(v))
        || boundary_text.is_some() && boundary.is_none()
    {
        diagnostics.push(argument_error(
            "RM-DATESHIFT-BOUNDARY",
            "boundary must be start, end, or dayofweek scalar text",
            1,
        ));
    }
    if let Some(unit) = request.arguments.get(2) {
        let numeric = real_numeric(unit);
        let text = request.literals.literal_string_at(2);
        match boundary {
            Some(DateShiftBoundary::Start | DateShiftBoundary::End) => {
                if !scalar_text(unit)
                    || text
                        .as_deref()
                        .is_some_and(|s| DateShiftUnit::parse_text(s).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-DATESHIFT-UNIT",
                        "unit must be a supported singular calendar/clock unit",
                        2,
                    ));
                }
            }
            Some(DateShiftBoundary::DayOfWeek) => {
                if !numeric
                    && (!scalar_text(unit)
                        || text
                            .as_deref()
                            .is_some_and(|s| DateShiftDayTarget::parse_text(s).is_none()))
                {
                    diagnostics.push(argument_error(
                        "RM-DATESHIFT-DAY-TARGET",
                        "weekday must be a scalar day name/group or real numeric indices 1..7",
                        2,
                    ));
                }
                if numeric {
                    if request
                        .literals
                        .literal_args
                        .get(2)
                        .is_some_and(|v| invalid_integer_literal(v, ControlDomain::Weekday))
                    {
                        diagnostics.push(argument_error(
                            "RM-DATESHIFT-WEEKDAY-RANGE",
                            "numeric weekdays must be exact indices in 1..7",
                            2,
                        ));
                    }
                    merge_shape(&mut shape, &unit.shape, 2, &mut diagnostics);
                } else if unit.kind == ValueKindFact::Unknown {
                    merge_shape(&mut shape, &unit.shape, 2, &mut diagnostics);
                }
            }
            None => {
                if !(numeric || scalar_text(unit)) {
                    diagnostics.push(argument_error(
                        "RM-DATESHIFT-UNIT",
                        "unit/weekday requires scalar text or real numeric controls",
                        2,
                    ));
                }
                if numeric || unit.kind == ValueKindFact::Unknown {
                    merge_shape(&mut shape, &unit.shape, 2, &mut diagnostics);
                }
            }
        }
    }
    if let Some(rule) = request.arguments.get(3) {
        if real_numeric(rule) {
            let domain = if boundary == Some(DateShiftBoundary::DayOfWeek) {
                ControlDomain::DayOccurrence
            } else {
                ControlDomain::Occurrence
            };
            if request
                .literals
                .literal_args
                .get(3)
                .is_some_and(|v| invalid_integer_literal(v, domain))
            {
                diagnostics.push(argument_error(
                    "RM-DATESHIFT-RULE-RANGE",
                    "occurrence controls must be exact supported integers",
                    3,
                ));
            }
            merge_shape(&mut shape, &rule.shape, 3, &mut diagnostics);
        } else {
            if rule.kind == ValueKindFact::Unknown {
                merge_shape(&mut shape, &rule.shape, 3, &mut diagnostics);
            }
            let text = request.literals.literal_string_at(3);
            if !scalar_text(rule)
                || text
                    .as_deref()
                    .is_some_and(|s| DateShiftRuleKeyword::parse_text(s).is_none())
            {
                diagnostics.push(argument_error(
                    "RM-DATESHIFT-RULE",
                    "rule must be current, next, previous, nearest, or a real integer scalar/array",
                    3,
                ));
            }
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DATETIME.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}
fn scalar_text(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) && !known_invalid_text_scalar(value)
}
fn real_numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
    )
}
fn merge_shape(
    shape: &mut ShapeFact,
    control: &ShapeFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if control.element_count() == Some(1) {
        return;
    }
    if shape.element_count() == Some(1) || *shape == ShapeFact::Unknown {
        *shape = control.clone();
        return;
    }
    if matches!((shape.known_dims(),control.known_dims()),(Some(a),Some(b)) if a.len()!=b.len()||a.iter().zip(&b).any(|(a,b)|a.is_some()&&b.is_some()&&a!=b))
    {
        diagnostics.push(argument_error(
            "RM-DATESHIFT-SHAPE",
            "non-scalar datetime, weekday, and rule arrays must have exactly matching sizes",
            index,
        ));
    }
}
#[derive(Clone, Copy)]
enum ControlDomain {
    Weekday,
    Occurrence,
    DayOccurrence,
}
fn invalid_integer_literal(value: &LiteralValue, domain: ControlDomain) -> bool {
    let parsed = match value {
        LiteralValue::Vector(values) => {
            return values.iter().any(|v| invalid_integer_literal(v, domain))
        }
        LiteralValue::Matrix(rows) => {
            return rows
                .iter()
                .flatten()
                .any(|v| invalid_integer_literal(v, domain))
        }
        LiteralValue::Unknown | LiteralValue::Empty => return false,
        LiteralValue::Integer { .. } => value
            .integer_value_i128()
            .and_then(|v| i64::try_from(v).ok()),
        LiteralValue::Number(v) => dateshift_exact_f64_to_i64(*v),
        LiteralValue::Real {
            text,
            class: NumericClass::Single,
        } => text
            .parse::<f32>()
            .ok()
            .and_then(|v| dateshift_exact_f64_to_i64(f64::from(v))),
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .ok()
            .and_then(dateshift_exact_f64_to_i64),
        _ => return true,
    };
    parsed.is_none_or(|v| match domain {
        ControlDomain::Weekday => !(1..=7).contains(&v),
        ControlDomain::Occurrence => false,
        ControlDomain::DayOccurrence => v.unsigned_abs() > MAX_DATESHIFT_DAY_OCCURRENCE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn datetime(shape: ShapeFact) -> ValueFact {
        ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(standard::DATETIME.owned()),
                properties: BTreeMap::new(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            shape,
            StorageFact::Opaque,
        )
    }

    #[test]
    fn exact_control_domain_and_empty_shape_without_clock() {
        let entry = &super::super::DATESHIFT_CATALOG_ENTRY;
        let empty = ShapeFact::from(vec![Some(0), Some(2), Some(1)]);
        let mut request = CallRequest {
            arguments: vec![
                datetime(empty.clone()),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("dayofweek".into()),
                LiteralValue::Integer {
                    text: "7".into(),
                    class: NumericClass::UInt64,
                },
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, empty);
        assert!(
            matches!(&result.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class == Some(standard::DATETIME.owned()))
        );
        assert!(!entry.contract.effects.contains(&EffectKind::Clock));
        request.literals.literal_args[2] = LiteralValue::Integer {
            text: "18446744073709551615".into(),
            class: NumericClass::UInt64,
        };
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.literals.literal_args[2] = LiteralValue::Number(1.5);
        assert!(!infer(&request, entry).diagnostics.is_empty());
    }

    #[test]
    fn scalar_expansion_requires_matching_non_scalar_geometry() {
        let entry = &super::super::DATESHIFT_CATALOG_ENTRY;
        let mut request = CallRequest {
            arguments: vec![
                datetime(ShapeFact::from(vec![Some(1), Some(1)])),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Int64,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![Some(0), Some(3)]),
                    StorageFact::Dense,
                ),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("start".into()),
                LiteralValue::String("month".into()),
                LiteralValue::Empty,
            ]),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, entry).outputs[0].shape,
            request.arguments[3].shape
        );
        request.arguments[3] = ValueFact::unknown(DynamicReason::RuntimeValue);
        request.literals.literal_args[3] = LiteralValue::Unknown;
        assert_eq!(infer(&request, entry).outputs[0].shape, ShapeFact::Unknown);
        request.arguments[3] = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(0), Some(3)]),
            StorageFact::Dense,
        );
        request.arguments[0] = datetime(ShapeFact::from(vec![Some(0), Some(2)]));
        assert!(!infer(&request, entry).diagnostics.is_empty());
        request.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
        request.arguments[3] = ValueFact::unknown(DynamicReason::RuntimeValue);
        request.literals.literal_args[3] = LiteralValue::Unknown;
        assert_eq!(infer(&request, entry).outputs[0].shape, ShapeFact::Unknown);
    }
}
