use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericDomain, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [input] => infer_input(input, &mut diagnostics),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FACTOR-ARITY",
                "factor requires exactly one input",
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
    if !input.is_scalar() && input.shape.element_count().is_some() {
        diagnostics.push(argument_error(
            "RM-CATALOG-FACTOR-SCALAR",
            "factor requires a scalar input",
            0,
        ));
    }
    let kind = match input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            input.kind.clone()
        }
        ValueKindFact::Unknown => {
            return ValueFact::unknown(DynamicReason::RuntimeValue);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FACTOR-INPUT",
                "factor requires a real numeric scalar",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    };
    ValueFact::proven(
        kind,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_catalog_entry_by_name;
    use runmat_types::{
        LiteralContext, NumericClass, NumericFact, OutputSelection, RequestedOutputCount,
        ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(Vec::new()),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn preserves_numeric_class_and_returns_dynamic_row() {
        let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        let entry = builtin_catalog_entry_by_name("factor").expect("factor entry");
        let inferred = infer(&request(vec![input]), entry);
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
}
