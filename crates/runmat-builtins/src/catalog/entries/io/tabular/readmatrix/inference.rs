use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut matrix = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    matrix.layout = LayoutFact::ColumnMajor;
    matrix.contiguity = ContiguityFact::Contiguous;
    matrix.view = ViewFact::Materialized;
    matrix.residency = ResidencyFact::Host;
    let len = request.arguments.len();
    let second_is_struct = request
        .arguments
        .get(1)
        .is_some_and(|fact| matches!(fact.kind, ValueKindFact::Struct(_)));
    let second_is_unknown = request
        .arguments
        .get(1)
        .is_some_and(|fact| matches!(fact.kind, ValueKindFact::Unknown));
    if second_is_struct || second_is_unknown {
        // An options struct (or unknown second argument) can select OutputType
        // or Like at runtime; its shape alone does not prove default double.
        matrix.kind = ValueKindFact::Unknown;
        matrix.residency = ResidencyFact::Unknown;
    }
    let mut diagnostics = Vec::new();
    if len == 0 || (len > 1 && !second_is_unknown && (len.is_multiple_of(2) != second_is_struct)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-READMATRIX-ARITY",
            "readmatrix requires a filename, an optional options struct, and complete name-value pairs",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !super::super::super::repl_fs::text_input::scalar(value))
        || request
            .literals
            .literal_string_at(0)
            .is_some_and(|text| text.is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-READMATRIX-FILENAME",
            "filename must be nonempty scalar text",
            0,
        ));
    }
    let first_name = if second_is_struct { 2 } else { 1 };
    if !second_is_unknown {
        for index in (first_name..len).step_by(2) {
            if !super::super::super::repl_fs::text_input::scalar(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-READMATRIX-OPTION",
                    "option names must be scalar text",
                    index,
                ));
                continue;
            }
            let Some(name) = request.literals.literal_string_at(index) else {
                continue;
            };
            let Some(value_index) = index.checked_add(1).filter(|next| *next < len) else {
                continue;
            };
            let control = super::ReadMatrixControl::parse(&name);
            if control == super::ReadMatrixControl::OutputType {
                if let Some(spec) = request.literals.literal_string_at(value_index) {
                    if let Some(kind) = output_type(&spec) {
                        matrix.kind = kind;
                    } else {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-READMATRIX-OUTPUT-TYPE",
                            "OutputType must name a supported numeric or logical class",
                            value_index,
                        ));
                    }
                } else {
                    matrix.kind = ValueKindFact::Unknown;
                }
            } else if control == super::ReadMatrixControl::Like {
                let prototype = &request.arguments[value_index];
                if matches!(
                    prototype.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical
                ) {
                    matrix.kind = prototype.kind.clone();
                    matrix.residency = prototype.residency.clone();
                } else {
                    matrix.kind = ValueKindFact::Unknown;
                    matrix.residency = ResidencyFact::Unknown;
                }
            }
        }
    }
    let mut contract = CallContract::fixed(vec![matrix]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn output_type(spec: &str) -> Option<ValueKindFact> {
    Some(match super::ReadMatrixOutputClass::parse(spec)? {
        super::ReadMatrixOutputClass::Logical => ValueKindFact::Logical,
        super::ReadMatrixOutputClass::Numeric(class) => ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
    })
}
