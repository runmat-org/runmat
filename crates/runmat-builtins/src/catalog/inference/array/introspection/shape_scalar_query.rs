use super::super::super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, ShapeScalarQuery};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    _query: ShapeScalarQuery,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHAPE-SCALAR-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            request.arguments.len().min(1),
        ));
    }
    if request.arguments.is_empty() {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    }
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{OutputSelection, RequestedOutputCount};

    #[test]
    fn shape_scalar_queries_return_double_scalars() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        for (entry, query) in [
            (&crate::LENGTH_CATALOG_ENTRY, ShapeScalarQuery::Length),
            (&crate::NDIMS_CATALOG_ENTRY, ShapeScalarQuery::Rank),
        ] {
            let result = super::infer(&request, entry, query);
            assert!(result.diagnostics.is_empty());
            assert!(result.outputs[0].is_scalar());
            assert_eq!(
                result.outputs[0].kind,
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            );
        }
    }

    #[test]
    fn missing_input_is_diagnostic_and_dynamic() {
        let result = super::infer(
            &CallRequest {
                arguments: Vec::new(),
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &crate::LENGTH_CATALOG_ENTRY,
            ShapeScalarQuery::Length,
        );
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(
            result.outputs[0].certainty,
            runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        );
    }
}
