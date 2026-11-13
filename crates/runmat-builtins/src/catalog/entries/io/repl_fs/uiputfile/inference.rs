use super::super::file_dialog::{double_scalar, filter, text, text_or_cancel};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-UIPUTFILE-ARITY",
            "uiputfile accepts filter, title, and default name",
            3,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !filter(value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-UIPUTFILE-FILTER",
            "uiputfile expects scalar text or a one/two-column text filter cell",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate().skip(1).take(2) {
        if !text(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-UIPUTFILE-TEXT",
                "uiputfile expects scalar text",
                index,
            ));
        }
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![text_or_cancel(), text_or_cancel(), double_scalar()],
        diagnostics,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    #[test]
    fn selected_and_cancelled_paths_join_without_erasing_index_class() {
        let request = CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let result = infer(&request, &super::super::UIPUTFILE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs.len(), 3);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert_eq!(result.outputs[2], double_scalar());
    }
    #[test]
    fn invalid_title_and_output_count_report_before_host_selection() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String), double_scalar()],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
        };
        assert!(
            infer(&request, &super::super::UIPUTFILE_CATALOG_ENTRY)
                .diagnostics
                .len()
                >= 2
        );
    }
}
