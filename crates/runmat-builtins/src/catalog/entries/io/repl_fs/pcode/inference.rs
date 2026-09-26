use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, ValueKindFact};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut possible_source = false;
    for (index, value) in request.arguments.iter().enumerate() {
        if !matches!(value.kind, ValueKindFact::Unknown) && !super::super::text_input::scalar(value)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PCODE-TEXT",
                "pcode expects scalar text",
                index,
            ));
        }
        match request.literals.literal_string_at(index) {
            Some(text) if text.is_empty() => diagnostics.push(argument_error(
                "RM-CATALOG-PCODE-EMPTY",
                "pcode source items must not be empty",
                index,
            )),
            Some(text) if text.starts_with('-') => {
                if !matches!(text.as_str(), "-inplace" | "-r2007b" | "-r2022a") {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PCODE-OPTION",
                        "pcode option is unsupported",
                        index,
                    ));
                }
            }
            _ => possible_source = true,
        }
    }
    if !possible_source {
        diagnostics.push(argument_error(
            "RM-CATALOG-PCODE-SOURCE",
            "pcode requires at least one source item",
            0,
        ));
    }
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    #[test]
    fn rejects_options_without_source_and_requested_outputs() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String("-inplace".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        assert!(
            infer(&request, &super::super::PCODE_CATALOG_ENTRY)
                .diagnostics
                .len()
                >= 2
        );
    }
    #[test]
    fn unknown_text_source_retains_filesystem_effects_and_zero_outputs() {
        let request = CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        let result = infer(&request, &super::super::PCODE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert!(result.outputs.is_empty());
        assert!(result.effects.0.contains(&EffectKind::FilesystemWrite));
    }
}
