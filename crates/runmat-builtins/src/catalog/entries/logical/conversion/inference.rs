use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-ARITY",
            "logical requires exactly one input",
            0,
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    let invalid = match &input.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Complex,
        ValueKindFact::Unknown
        | ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Symbolic => false,
        _ => true,
    };
    if invalid {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-CONVERSION",
            "logical requires real numeric, logical, character, or an enabled extension input",
            0,
        ));
    }
    // Logical values are returned unchanged, including sparse storage and aliasing.
    if input.kind == ValueKindFact::Logical {
        return finish_fixed(entry, request, input.clone(), diagnostics);
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Logical;
    output.shape = input.shape.clone();
    output.storage = match input.storage {
        StorageFact::Sparse => StorageFact::Sparse,
        StorageFact::Unknown => StorageFact::Unknown,
        _ if input.shape.element_count() == Some(1) => StorageFact::Scalar,
        _ => StorageFact::Dense,
    };
    // Residency facts do not distinguish explicit gpuArray intent from automatic
    // residency. A provider fallback may legitimately return an automatic input
    // to host, so device residency is not a provable output fact.
    output.residency = match input.residency {
        ResidencyFact::Host => ResidencyFact::Host,
        _ => ResidencyFact::Unknown,
    };
    if output.storage != StorageFact::Unknown {
        output.layout = LayoutFact::ColumnMajor;
        output.contiguity = ContiguityFact::Contiguous;
        output.view = ViewFact::Materialized;
    }
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn infer_input(input: ValueFact) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::LOGICAL_CATALOG_ENTRY,
        )
    }
    #[test]
    fn unknown_preserves_uncertainty() {
        let result = infer_input(ValueFact::unknown(DynamicReason::RuntimeValue));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    }
    #[test]
    fn sparse_and_empty_nd_shapes_are_preserved() {
        for (shape, storage) in [
            (vec![Some(3), Some(4)], StorageFact::Sparse),
            (vec![Some(0), Some(3), Some(2)], StorageFact::Dense),
        ] {
            let input = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(shape),
                storage,
            );
            let result = infer_input(input.clone());
            assert_eq!(result.outputs[0].shape, input.shape);
            assert_eq!(result.outputs[0].storage, storage);
        }
    }
    #[test]
    fn logical_identity_does_not_invent_unique_storage() {
        let mut input = ValueFact::scalar(ValueKindFact::Logical);
        input.alias = AliasFact::Shared;
        assert_eq!(infer_input(input.clone()).outputs[0], input);
    }
    #[test]
    fn invalid_complex_and_requested_outputs_diagnose() {
        let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }));
        assert!(!infer_input(input).diagnostics.is_empty());
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        assert!(!infer(&request, &super::super::LOGICAL_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
    #[test]
    fn device_input_does_not_prove_device_output_after_fallback() {
        let mut input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        input.residency = ResidencyFact::Device {
            provider: Some("test".into()),
        };
        assert_eq!(
            infer_input(input).outputs[0].residency,
            ResidencyFact::Unknown
        );
    }
}
