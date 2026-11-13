use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXIST-ARITY",
            "exist expects a name and optional query selector",
            0,
        ));
    }
    let selector = request.literals.literal_string_at(1);
    // A dynamic selector may be the RunMat handle-query extension. Its nontext
    // name is not a proven error until the selector is resolved at runtime.
    let requires_text_name = request.arguments.len() == 1
        || selector
            .as_deref()
            .is_some_and(|value| value.trim() != "handle");
    if requires_text_name
        && request
            .arguments
            .first()
            .is_some_and(|value| !possible_text(value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXIST-NAME",
            "exist expects a character-row or string-scalar name",
            0,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(|value| !possible_text(value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXIST-SELECTOR",
            "exist expects a character-row or string-scalar query selector",
            1,
        ));
    }
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

fn possible_text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown) || super::super::text_input::scalar(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    fn request(name: ValueFact, selector: Option<LiteralValue>) -> CallRequest {
        let mut arguments = vec![name];
        let mut literals = vec![LiteralValue::Unknown];
        if let Some(selector) = selector {
            arguments.push(ValueFact::scalar(ValueKindFact::String));
            literals.push(selector);
        }
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }
    #[test]
    fn result_is_host_double_scalar_with_lookup_effects() {
        let result = infer(
            &request(ValueFact::scalar(ValueKindFact::String), None),
            &super::super::EXIST_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0],
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        );
        assert!(result.effects.0.contains(&EffectKind::WorkspaceRead));
        assert!(result.effects.0.contains(&EffectKind::FilesystemRead));
    }
    #[test]
    fn selector_admission_preserves_handle_extension_and_unknowns() {
        let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        for selector in [LiteralValue::String("handle".into()), LiteralValue::Unknown] {
            assert!(infer(
                &request(numeric.clone(), Some(selector)),
                &super::super::EXIST_CATALOG_ENTRY
            )
            .diagnostics
            .is_empty());
        }
        assert!(
            !infer(&request(numeric, None), &super::super::EXIST_CATALOG_ENTRY)
                .diagnostics
                .is_empty()
        );
    }
    #[test]
    fn true_character_pages_reject_but_trailing_singletons_remain_text() {
        for (shape, valid) in [
            (vec![Some(1), Some(3), Some(2)], false),
            (vec![Some(1), Some(3), Some(1)], true),
            (vec![Some(1), Some(3), None], true),
        ] {
            let name = ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(shape),
                StorageFact::Dense,
            );
            assert_eq!(
                infer(&request(name, None), &super::super::EXIST_CATALOG_ENTRY)
                    .diagnostics
                    .is_empty(),
                valid
            );
        }
    }
}
