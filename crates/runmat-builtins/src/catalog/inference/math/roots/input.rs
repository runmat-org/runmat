use runmat_types::{CallRequest, InferenceDiagnostic, ValueFact};

use super::literal::{self, RootLiteralDomain};

pub(super) struct UnaryRootInput<'a> {
    pub(super) fact: &'a ValueFact,
    pub(super) literal: Option<RootLiteralDomain>,
    pub(super) diagnostics: Vec<InferenceDiagnostic>,
}

pub(super) fn prepare<'a>(
    request: &'a CallRequest,
    name: &'static str,
) -> Result<UnaryRootInput<'a>, Vec<InferenceDiagnostic>> {
    let Some(fact) = request.arguments.first() else {
        return Err(vec![super::super::super::argument_error(
            "RM-CATALOG-ROOT-ARITY",
            format!("{name} requires exactly one input"),
            0,
        )]);
    };
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(super::super::super::argument_error(
            "RM-CATALOG-ROOT-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }
    let literal = request
        .literals
        .literal_args
        .first()
        .and_then(literal::classify);
    Ok(UnaryRootInput {
        fact,
        literal,
        diagnostics,
    })
}
