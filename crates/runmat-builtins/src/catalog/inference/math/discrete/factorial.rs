use crate::{
    catalog::inference::{argument_error, finish_fixed, literal_text},
    BuiltinCatalogEntry,
};
use runmat_types::{
    AliasFact, CallInference, CallRequest, ContiguityFact, DynamicReason, LayoutFact, MutationFact,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
    ViewFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-FACTORIAL-ARITY",
            "factorial requires an input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let mut output = infer_input(input, &mut diagnostics);
    match request.arguments.as_slice() {
        [_] => {}
        [_, keyword, prototype] => {
            infer_like(keyword, prototype, request, &mut output, &mut diagnostics)
        }
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-FACTORIAL-ARITY",
            "factorial accepts one input or an input followed by \"like\" and a prototype",
            request.arguments.len().saturating_sub(1),
        )),
    }

    finish_fixed(entry, request, output, diagnostics)
}

fn infer_input(
    input: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    let output_kind = match input.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Unknown => {
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FACTORIAL-INPUT",
                "factorial requires a dense real numeric input",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    };
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FACTORIAL-STORAGE",
            "factorial does not support sparse input",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = input.clone();
    output.kind = output_kind;
    output.storage = if input.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.residency = match input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { .. } => input.residency.clone(),
        _ => ResidencyFact::Unknown,
    };
    output.alias = AliasFact::Unique;
    output.view = ViewFact::Materialized;
    output.mutation = MutationFact::ValueSemantics;
    output
}

fn infer_like(
    keyword: &ValueFact,
    prototype: &ValueFact,
    request: &CallRequest,
    output: &mut ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match request.literals.literal_args.get(1).and_then(literal_text) {
        Some(value) if value.eq_ignore_ascii_case("like") => {
            output.residency = prototype_residency(prototype, diagnostics);
        }
        Some(_) => diagnostics.push(argument_error(
            "RM-CATALOG-FACTORIAL-LIKE",
            "factorial accepts only the \"like\" option",
            1,
        )),
        None if matches!(
            keyword.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) =>
        {
            output.residency = ResidencyFact::Unknown;
        }
        None => diagnostics.push(argument_error(
            "RM-CATALOG-FACTORIAL-LIKE",
            "factorial requires \"like\" as its second input",
            1,
        )),
    }
}

fn prototype_residency(
    prototype: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ResidencyFact {
    match prototype.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => match &prototype.residency {
            ResidencyFact::Host => ResidencyFact::Host,
            ResidencyFact::Device { provider } => ResidencyFact::Device {
                provider: provider.clone(),
            },
            ResidencyFact::Unknown => ResidencyFact::Unknown,
            ResidencyFact::Remote { .. } => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FACTORIAL-PROTOTYPE",
                    "factorial cannot use a remote value as a residency prototype",
                    2,
                ));
                ResidencyFact::Unknown
            }
        },
        ValueKindFact::Unknown => ResidencyFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FACTORIAL-PROTOTYPE",
                "factorial requires a real numeric or logical prototype",
                2,
            ));
            ResidencyFact::Unknown
        }
    }
}

#[cfg(test)]
#[path = "factorial_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "distributed_tests.rs"]
mod distributed_tests;
