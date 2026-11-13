use super::WebOptionField;
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

const FIELDS: [WebOptionField; 11] = [
    WebOptionField::ContentType,
    WebOptionField::Timeout,
    WebOptionField::KeyName,
    WebOptionField::KeyValue,
    WebOptionField::HeaderFields,
    WebOptionField::UserAgent,
    WebOptionField::Username,
    WebOptionField::Password,
    WebOptionField::RequestMethod,
    WebOptionField::MediaType,
    WebOptionField::QueryParameters,
];

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut start = 0;
    if let Some(first) = request.arguments.first() {
        match &first.kind {
            ValueKindFact::Struct(_) if first.shape.element_count() == Some(1) => start = 1,
            ValueKindFact::Struct(_) if first.shape.element_count().is_some() => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-WEBOPTIONS-STRUCT",
                    "options input must be a scalar struct",
                    0,
                ));
                return finish_fixed(entry, request, output_fact(), diagnostics);
            }
            ValueKindFact::Struct(_) => {
                return finish_fixed(entry, request, output_fact(), diagnostics);
            }
            ValueKindFact::Unknown if request.literals.literal_string_at(0).is_none() => {
                return finish_fixed(entry, request, output_fact(), diagnostics);
            }
            _ => {}
        }
    }
    if !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-WEBOPTIONS-PAIR",
            "weboptions requires complete name-value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let mut index = start;
    while index < request.arguments.len() {
        let name = &request.arguments[index];
        if !possible_text(name) {
            diagnostics.push(argument_error(
                "RM-CATALOG-WEBOPTIONS-NAME",
                "option name must be a character row or string scalar",
                index,
            ));
        } else if let Some(literal) = request.literals.literal_string_at(index) {
            match WebOptionField::decode(&literal) {
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-WEBOPTIONS-UNKNOWN",
                    "weboptions does not recognize this option name",
                    index,
                )),
                Some(field)
                    if request
                        .arguments
                        .get(index + 1)
                        .is_some_and(|value| invalid_value_class(field, value)) =>
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WEBOPTIONS-VALUE",
                        "weboptions option value has a class or shape rejected by this field",
                        index + 1,
                    ))
                }
                _ => {}
            }
        }
        index += 2;
    }
    finish_fixed(entry, request, output_fact(), diagnostics)
}

fn invalid_value_class(field: WebOptionField, value: &ValueFact) -> bool {
    use WebOptionField as Field;
    if matches!(value.kind, ValueKindFact::Unknown) {
        return false;
    }
    match field {
        Field::ContentType
        | Field::KeyName
        | Field::UserAgent
        | Field::Username
        | Field::Password
        | Field::RequestMethod
        | Field::MediaType => !possible_text(value),
        Field::Timeout => {
            !matches!(value.kind, ValueKindFact::Numeric(_))
                || value.shape.element_count().is_some_and(|count| count != 1)
        }
        Field::KeyValue => {
            if possible_text(value) {
                return false;
            }
            !matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) || value.shape.element_count().is_some_and(|count| count != 1)
        }
        Field::HeaderFields | Field::QueryParameters => !matches!(
            value.kind,
            ValueKindFact::Struct(_) | ValueKindFact::Cell(_)
        ),
    }
}

fn output_fact() -> ValueFact {
    let fields = FIELDS
        .into_iter()
        .map(|field| {
            let value = match field {
                WebOptionField::Timeout => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                WebOptionField::HeaderFields | WebOptionField::QueryParameters => {
                    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                        BTreeMap::new(),
                        false,
                    )))
                }
                WebOptionField::KeyValue => ValueFact::unknown(DynamicReason::RuntimeValue),
                _ => ValueFact::scalar(ValueKindFact::String),
            };
            (field.canonical_name().to_string(), value)
        })
        .collect();
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

fn possible_text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || super::super::super::repl_fs::text_input::scalar(value)
}
