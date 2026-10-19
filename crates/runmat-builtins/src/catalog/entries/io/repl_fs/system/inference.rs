use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SYSTEM-ARITY",
            "system requires a command and at most one echo option",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !super::super::text_input::scalar(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SYSTEM-TEXT",
                "system controls must be host string scalars or character rows",
                index,
            ));
        }
    }
    if let Some(option) = request.literals.literal_string_at(1) {
        if !option.eq_ignore_ascii_case("-echo") {
            diagnostics.push(argument_error(
                "RM-CATALOG-SYSTEM-ECHO",
                "system optional second argument must be '-echo'",
                1,
            ));
        }
    }
    let status = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    finish_fixed_outputs(entry, request, vec![status, output], diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount};

    #[test]
    fn status_and_output_facts_match_runtime() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::String("echo ready".into()),
                LiteralValue::String("-echo".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        let valid = infer(&request, &super::super::SYSTEM_CATALOG_ENTRY);
        assert!(valid.diagnostics.is_empty());
        assert_eq!(valid.outputs.len(), 2);
        assert!(matches!(
            valid.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
        assert_eq!(valid.outputs[1].kind, ValueKindFact::Character);

        let invalid = CallRequest {
            literals: LiteralContext::new(vec![
                LiteralValue::String("echo ready".into()),
                LiteralValue::String("-bad".into()),
            ]),
            ..request
        };
        assert!(infer(&invalid, &super::super::SYSTEM_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-SYSTEM-ECHO"));
    }
}
