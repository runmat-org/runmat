use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericDomain, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [input] => infer_input(input, &mut diagnostics),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISPRIME-ARITY",
                "isprime requires exactly one input",
                request.arguments.len().min(1),
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_input(
    input: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    match input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            ValueFact::proven(
                ValueKindFact::Logical,
                input.shape.clone(),
                storage_for(input),
            )
        }
        ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISPRIME-INPUT",
                "isprime requires real numeric input",
                0,
            ));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    }
}

fn storage_for(input: &ValueFact) -> StorageFact {
    if input.is_scalar() {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_catalog_entry_by_name;
    use runmat_types::{
        LiteralContext, NumericClass, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(Vec::new()),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn returns_shape_preserving_logical_output() {
        let shape = ShapeFact::from(vec![Some(2), Some(3)]);
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int32,
                domain: NumericDomain::Real,
            }),
            shape.clone(),
            StorageFact::Dense,
        );
        let entry = builtin_catalog_entry_by_name("isprime").expect("isprime entry");
        let inferred = infer(&request(vec![input]), entry);
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(inferred.outputs[0].shape, shape);
    }
}
