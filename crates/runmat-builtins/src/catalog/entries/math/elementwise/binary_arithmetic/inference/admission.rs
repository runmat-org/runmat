use crate::catalog::inference::argument_error;
use runmat_types::{CallRequest, ValueFact, ValueKindFact};

use super::DiagnosticCollector;

pub(super) struct AdmittedCall<'a> {
    pub left: Option<&'a ValueFact>,
    pub right: Option<&'a ValueFact>,
    pub prototype: Option<&'a ValueFact>,
    pub diagnostics: DiagnosticCollector,
}

pub(super) fn inspect(request: &CallRequest, render_diagnostics: bool) -> AdmittedCall<'_> {
    let mut diagnostics = DiagnosticCollector::new(render_diagnostics);
    let valid_arity = matches!(request.arguments.len(), 2 | 4);
    if !valid_arity {
        diagnostics.push(|| argument_error(
            "RM-CATALOG-BINARY-ARITHMETIC-ARITY",
            "element-wise binary arithmetic requires two inputs or two inputs followed by 'like' and a prototype",
            request.arguments.len().min(2),
        ));
    }

    let prototype = if request.arguments.len() == 4 {
        let keyword = request.literals.literal_string_at(2);
        let possible_text = matches!(
            request.arguments[2].kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        );
        if possible_text && keyword.as_deref().is_none_or(|value| value == "like") {
            request.arguments.get(3)
        } else {
            diagnostics.push(|| {
                argument_error(
                    "RM-CATALOG-BINARY-ARITHMETIC-OPTION",
                    "the third input must be 'like'",
                    2,
                )
            });
            None
        }
    } else {
        None
    };

    AdmittedCall {
        left: request.arguments.first(),
        right: request.arguments.get(1),
        prototype,
        diagnostics,
    }
}
