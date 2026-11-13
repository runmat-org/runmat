use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-SYMS-ARITY",
            "syms requires at least one declaration",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        let invalid = match input.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::String => input.shape.element_count().is_some_and(|n| n != 1),
            ValueKindFact::Character => input.shape.known_dims().is_some_and(|dims| {
                dims.first().is_some_and(|d| d.is_some_and(|d| d != 1))
                    || dims.iter().skip(2).any(|d| d.is_some_and(|d| d != 1))
            }),
            _ => true,
        };
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-SYMS-INPUT",
                "syms declarations must be scalar text",
                index,
            ));
        }
    }
    // Workspace assignment remains an effect even when no return value is selected.
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_outputs_do_not_erase_workspace_write() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        let result = infer(&request, &super::super::SYMS_CATALOG_ENTRY);
        assert!(result.outputs.is_empty());
        assert!(result.effects.0.contains(&EffectKind::WorkspaceWrite));
    }
    #[test]
    fn placeholder_shape_and_invalid_input() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::SYMS_CATALOG_ENTRY);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape.element_count(), Some(0));
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
}
