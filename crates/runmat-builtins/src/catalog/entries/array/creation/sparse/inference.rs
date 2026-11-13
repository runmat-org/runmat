use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::Ranked { rank: 2 };
    output.storage = StorageFact::Sparse;
    output.residency = ResidencyFact::Host;
    let count = request.arguments.len();
    let third_text = request.literals.literal_string_at(2);
    let form = super::SparseConstructionForm::for_arguments(count, third_text.is_some());
    if form == Some(super::SparseConstructionForm::Conversion) {
        let input = &request.arguments[0];
        output.kind = input.kind.clone();
        if input.shape.rank() == Some(2) {
            output.shape = input.shape.clone();
        }
        if !matches!(
            input.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-SPARSE-CONVERSION",
                "sparse requires numeric or logical input",
                0,
            ));
            output.kind = ValueKindFact::Unknown;
        }
        if input.shape.rank().is_some_and(|rank| rank != 2) {
            diagnostics.push(argument_error(
                "RM-SPARSE-RANK",
                "sparse conversion requires a 2-D matrix",
                0,
            ));
        }
        if input.numeric().is_some_and(|numeric| {
            numeric.domain == NumericDomain::Complex && numeric.class.integer_class().is_some()
        }) {
            diagnostics.push(argument_error(
                "RM-SPARSE-COMPLEX-INTEGER",
                "complex integer sparse storage is unsupported",
                0,
            ));
        }
    } else if form == Some(super::SparseConstructionForm::Dimensions) {
        output.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        if let Some(label) = third_text {
            output.kind = match super::SparseStorageType::parse(&label) {
                Some(super::SparseStorageType::Double) => ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                Some(super::SparseStorageType::Single) => ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    domain: NumericDomain::Real,
                }),
                Some(super::SparseStorageType::Logical) => ValueKindFact::Logical,
                None => {
                    diagnostics.push(argument_error(
                        "RM-SPARSE-TYPENAME",
                        "typename must be double, single, or logical",
                        2,
                    ));
                    ValueKindFact::Unknown
                }
            };
        }
        output.shape = dimensions_shape(request, 0, &mut diagnostics);
    } else if form == Some(super::SparseConstructionForm::Triplets) {
        let values = &request.arguments[2];
        let dynamic_form = count == 3
            && matches!(
                values.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            );
        if !dynamic_form {
            output.kind = values.kind.clone();
            if !matches!(
                values.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            ) || values
                .numeric()
                .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
            {
                diagnostics.push(argument_error(
                    "RM-SPARSE-VALUES",
                    "triplet values must be real numeric or logical data",
                    2,
                ));
                output.kind = ValueKindFact::Unknown;
            }
            let integers = request.arguments[..2]
                .iter()
                .map(|fact| {
                    fact.numeric()
                        .and_then(|numeric| numeric.class.integer_class())
                })
                .collect::<Vec<_>>();
            if integers[0] != integers[1]
                && request.arguments[..2]
                    .iter()
                    .all(|fact| fact.kind != ValueKindFact::Unknown)
            {
                diagnostics.push(argument_error(
                    "RM-SPARSE-INDEX-CLASS",
                    "integer row and column subscripts require one shared datatype",
                    0,
                ));
            }
        }
        if count >= 5 {
            output.shape = dimensions_shape(request, 3, &mut diagnostics);
        }
        if count == 6 {
            dimension(request, 5, &mut diagnostics);
        }
    } else {
        diagnostics.push(argument_error(
            "RM-SPARSE-ARITY",
            "expected conversion, dimensions, or three/five/six triplet arguments",
            0,
        ));
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn dimensions_shape(
    request: &CallRequest,
    start: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    ShapeFact::from(vec![
        dimension(request, start, diagnostics),
        dimension(request, start + 1, diagnostics),
    ])
}
fn dimension(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let fact = &request.arguments[index];
    let literal = request.literals.literal_args.get(index);
    let result = match literal {
        Some(LiteralValue::Bool(value)) => Some(usize::from(*value)),
        Some(
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. },
        ) => request
            .literals
            .numeric_dims_from(index)
            .first()
            .copied()
            .flatten(),
        _ => None,
    };
    let known_invalid_literal = literal
        .is_some_and(|literal| !matches!(literal, LiteralValue::Unknown) && result.is_none());
    if known_invalid_literal
        || fact.shape.element_count().is_some_and(|count| count != 1)
        || !matches!(
            fact.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        )
        || fact
            .numeric()
            .is_some_and(|numeric| numeric.domain != NumericDomain::Real)
    {
        diagnostics.push(argument_error(
            "RM-SPARSE-SIZE",
            "dimensions and nzmax must be nonnegative scalar integers",
            index,
        ));
    }
    result
}
