use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry, OptimParameterLookup, OptimParameterName,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPTIMGET-ARITY",
            "expected options, parameter and optional default",
            0,
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    let empty = || {
        super::super::program_facts::double(
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        )
    };
    let fallback = || request.arguments.get(2).cloned().unwrap_or_else(empty);
    let mut parameter = None;
    if let Some(name) = request.literals.literal_string_at(1) {
        match OptimParameterName::resolve(&name) {
            OptimParameterLookup::Matched(field) => parameter = Some(field),
            OptimParameterLookup::Ambiguous | OptimParameterLookup::Unknown => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIMGET-NAME",
                    "parameter name is unknown or ambiguous",
                    1,
                ))
            }
        }
    } else if request.arguments.get(1).is_some_and(|name| {
        !matches!(
            name.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        )
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPTIMGET-TEXT",
            "parameter must be scalar text",
            1,
        ));
    }
    if let Some(options) = request.arguments.first() {
        match &options.kind {
            ValueKindFact::Struct(fields) => {
                if options.shape.element_count().is_some_and(|n| n != 1) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIMGET-STRUCT",
                        "options must be a scalar struct",
                        0,
                    ));
                }
                if let Some(parameter) = parameter {
                    let mut matches=fields.fields.iter().filter_map(|(name,value)|
                        matches!(OptimParameterName::resolve(name),OptimParameterLookup::Matched(field)if field.name()==parameter.name()).then_some(value));
                    if let Some(field) = matches.next() {
                        if matches.next().is_some() {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-OPTIMGET-DUPLICATE",
                                "multiple fields identify the same parameter",
                                0,
                            ));
                        }
                        if numeric_empty(field) {
                            output = fallback();
                        } else if field.shape.element_count().is_some_and(|n| n > 0)
                            || matches!(
                                field.residency,
                                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                            )
                            || field.storage == StorageFact::Sparse
                            || !matches!(
                                field.kind,
                                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
                            )
                        {
                            output = field.clone();
                        }
                    } else if fields.fields_complete {
                        output = fallback();
                    }
                }
            }
            ValueKindFact::Numeric(numeric) => {
                if numeric_empty(options) {
                    output = fallback();
                } else if numeric.domain != NumericDomain::Real
                    || options.shape.element_count().is_some_and(|count| count > 0)
                    || options.storage == StorageFact::Sparse
                    || matches!(
                        options.residency,
                        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                    )
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIMGET-OPTIONS",
                        "options must be a scalar struct or numeric []",
                        0,
                    ));
                }
            }
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMGET-OPTIONS",
                "options must be a scalar struct or numeric []",
                0,
            )),
        }
    }
    // A new view of the same carrier is not a new uniqueness certificate.
    output.alias = AliasFact::Unknown;
    finish_fixed(entry, request, output, diagnostics)
}
fn numeric_empty(value: &ValueFact) -> bool {
    value.shape.element_count() == Some(0)
        && value.residency == ResidencyFact::Host
        && value.storage == StorageFact::Dense
        && matches!(
            value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
        )
}
