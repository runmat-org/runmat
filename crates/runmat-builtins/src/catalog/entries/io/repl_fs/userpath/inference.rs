use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-USERPATH-ARITY",
            "userpath accepts at most one input",
            1,
        ));
    }
    if let Some(argument) = request.arguments.first() {
        if !super::super::text_input::scalar(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-USERPATH-TEXT",
                "userpath requires host scalar text",
                0,
            ));
        }
        if let Some(text) = request.literals.literal_string_at(0) {
            if let Err(error) = super::UserPathControl::parse(&text) {
                let (code, message) = match error {
                    super::UserPathControlError::Empty => (
                        "RM-CATALOG-USERPATH-EMPTY",
                        "an empty custom path is not admitted; use clear",
                    ),
                    super::UserPathControlError::NotOnePath => (
                        "RM-CATALOG-USERPATH-PATH",
                        "userpath requires one path entry without a NUL byte",
                    ),
                };
                diagnostics.push(argument_error(code, message, 0));
            }
        }
    }
    finish_fixed(
        entry,
        request,
        super::super::search_path::result::character_row(),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact,
        ValueKindFact,
    };

    #[test]
    fn text_forms_and_known_invalid_values_match_runtime_admission() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String("clear".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let entry = &super::super::USERPATH_CATALOG_ENTRY;
        let result = infer(&request, entry);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
        let bad = CallRequest {
            literals: LiteralContext::new(vec![LiteralValue::String(String::new())]),
            ..request.clone()
        };
        assert!(infer(&bad, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-USERPATH-EMPTY"));
        let numeric = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Logical)],
            ..request
        };
        assert!(infer(&numeric, entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-USERPATH-TEXT"));
    }

    #[test]
    fn partial_path_release_is_declared_without_misclassifying_absolute_or_commands() {
        use crate::{BuiltinExtensionMode, MatlabRelease, MatlabReleaseHalf};

        assert_eq!(
            super::super::USERPATH_PARTIAL_PATH_EXTENSION.mode,
            BuiltinExtensionMode::AvailableSince(MatlabRelease {
                year: 2026,
                half: MatlabReleaseHalf::B,
            })
        );
        let entry = &super::super::USERPATH_CATALOG_ENTRY;
        assert_eq!(
            entry.contract.compatibility,
            crate::BuiltinCompatibility::Matlab
        );
        assert_eq!(
            entry.extensions,
            &[super::super::USERPATH_PARTIAL_PATH_EXTENSION]
        );
        let request = |text: &str| CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![LiteralValue::String(text.into())]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        for text in ["reset", "clear", "relative/toolbox"] {
            assert!(
                infer(&request(text), entry).diagnostics.is_empty(),
                "{text}"
            );
        }
        #[cfg(windows)]
        let absolute = "C:\\installed\\toolbox";
        #[cfg(not(windows))]
        let absolute = "/installed/toolbox";
        assert!(infer(&request(absolute), entry).diagnostics.is_empty());
        assert!(infer(&request("first\0second"), entry)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-USERPATH-PATH"));
    }
}
