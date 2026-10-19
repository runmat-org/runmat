use crate::catalog::inference::argument_error;
use runmat_types::{
    broadcast_shape, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn fact(
    source: &ValueFact,
    operands: &[(usize, &ValueFact)],
    grammar_known: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let mut shape = if grammar_known {
        source.shape.clone()
    } else {
        ShapeFact::Unknown
    };
    if grammar_known {
        for (index, operand) in operands {
            shape = match broadcast_shape(&shape, &operand.shape) {
                Ok(shape) => shape,
                Err(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-RESCALE-SIZE",
                        "rescale bounds must be implicitly expandable with the input",
                        *index,
                    ));
                    ShapeFact::Unknown
                }
            };
        }
    }

    let kind = match source.kind {
        ValueKindFact::Numeric(numeric) => ValueKindFact::Numeric(NumericFact {
            class: if numeric.class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            },
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => ValueKindFact::Unknown,
    };
    let storage = if shape.element_count() == Some(1)
        && matches!(
            kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                ..
            })
        ) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let mut output = ValueFact::proven(kind, shape, storage);
    output.residency = residency(source, operands, diagnostics);
    if matches!(output.kind, ValueKindFact::Unknown) {
        output.certainty = runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}

fn residency(
    source: &ValueFact,
    operands: &[(usize, &ValueFact)],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ResidencyFact {
    let mut provider: Option<Option<String>> = None;
    let mut uncertain = false;
    for (index, value) in std::iter::once((0, source)).chain(operands.iter().copied()) {
        match &value.residency {
            ResidencyFact::Host => {}
            ResidencyFact::Device { provider: next } => match &provider {
                None => provider = Some(next.clone()),
                Some(current) if current == next => {}
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-RESCALE-PROVIDER",
                        "provider-resident rescale operands must share one owner",
                        index,
                    ));
                    uncertain = true;
                }
            },
            ResidencyFact::Unknown | ResidencyFact::Remote { .. } => uncertain = true,
        }
    }
    if uncertain {
        ResidencyFact::Unknown
    } else if let Some(provider) = provider {
        ResidencyFact::Device { provider }
    } else {
        ResidencyFact::Host
    }
}
