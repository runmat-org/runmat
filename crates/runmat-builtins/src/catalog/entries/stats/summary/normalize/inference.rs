//! Conservative static facts for normalization's method-dependent outputs.
use super::NormalizeKeyword;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    if input.is_none() {
        diagnostics.push(error("ARITY", "normalize requires input data"));
    }
    let class_domain = input.and_then(|fact| match fact.kind {
        ValueKindFact::Numeric(NumericFact { class, domain }) => Some((
            match class {
                NumericClass::Single => NumericClass::Single,
                NumericClass::Double
                | NumericClass::Int8
                | NumericClass::UInt8
                | NumericClass::Int16
                | NumericClass::UInt16
                | NumericClass::Int32
                | NumericClass::UInt32
                | NumericClass::Int64
                | NumericClass::UInt64 => NumericClass::Double,
            },
            domain,
        )),
        ValueKindFact::Logical => Some((NumericClass::Double, NumericDomain::Real)),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(error(
                "INPUT",
                "normalize requires numeric or logical array data",
            ));
            None
        }
    });
    // Strings are decoded only at the public literal boundary. Unknown
    // options remain dynamic rather than acquiring a fictitious method.
    for literal in request.literals.literal_args.iter().skip(1) {
        let text = match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => text,
            _ => continue,
        };
        let method = NormalizeKeyword::from_public_text(text);
        if matches!(
            method,
            NormalizeKeyword::DataVariables | NormalizeKeyword::ReplaceValues
        ) {
            diagnostics.push(error(
                "METHOD",
                "tabular normalize options are not supported for array input",
            ));
        }
    }
    let shape = input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone());
    let mut normalized = numeric_output(class_domain, shape.clone());
    normalized.storage = match (class_domain, shape.element_count()) {
        (Some((NumericClass::Single, _)), Some(1)) => StorageFact::Dense,
        (_, Some(1)) => StorageFact::Scalar,
        (_, Some(_)) => StorageFact::Dense,
        _ => StorageFact::Unknown,
    };
    if let Some(input) = input {
        normalized.residency = match input.residency {
            ResidencyFact::Host => ResidencyFact::Host,
            ResidencyFact::Device { .. }
            | ResidencyFact::Remote { .. }
            | ResidencyFact::Unknown => ResidencyFact::Unknown,
        };
    }
    // C and S may be explicit broadcastable arrays, reduced parameter
    // arrays, or scalar parameters depending on the selected method.
    let center = numeric_output(class_domain, ShapeFact::Unknown);
    let scale = numeric_output(
        class_domain.map(|(class, _)| (class, NumericDomain::Real)),
        ShapeFact::Unknown,
    );
    let mut contract = CallContract::fixed(vec![normalized, center, scale]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn numeric_output(
    class_domain: Option<(NumericClass, NumericDomain)>,
    shape: ShapeFact,
) -> ValueFact {
    match class_domain {
        Some((class, domain)) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact { class, domain }),
            shape,
            StorageFact::Unknown,
        ),
        None => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.shape = shape;
            fact
        }
    }
}

fn error(code: &'static str, message: &'static str) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount};

    fn request(kind: ValueKindFact, shape: ShapeFact) -> CallRequest {
        CallRequest {
            arguments: vec![ValueFact::proven(kind, shape, StorageFact::Dense)],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        }
    }

    #[test]
    fn one_element_single_remains_dense_and_integer_enters_double_boundary() {
        let single = request(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Scalar,
        );
        let result = infer(&single, &super::super::NORMALIZE_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            })
        );

        let integer = request(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Scalar,
        );
        let result = infer(&integer, &super::super::NORMALIZE_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
    }

    #[test]
    fn unknown_huge_rank_remains_symbolic() {
        let shape = ShapeFact::Ranked { rank: usize::MAX };
        let request = request(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape.clone(),
        );
        let result = infer(&request, &super::super::NORMALIZE_CATALOG_ENTRY);
        assert_eq!(result.outputs[0].shape, shape);
    }
}
