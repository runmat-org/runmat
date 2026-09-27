use crate::catalog::inference::{argument_error, literal_text};
use runmat_types::{CallRequest, InferenceDiagnostic, LiteralContext, LiteralValue};

pub(super) struct Options {
    pub(super) start: usize,
    pub(super) uniform_output: Option<bool>,
    pub(super) diagnostics: Vec<InferenceDiagnostic>,
}

impl Options {
    pub(super) fn parse(request: &CallRequest) -> Self {
        let start = trailing_pair_start(request);
        let mut diagnostics = Vec::new();
        let mut uniform_output = Some(true);
        // Runtime consumes trailing pairs right-to-left, so the first duplicate wins.
        for index in (start..request.arguments.len().saturating_sub(1))
            .step_by(2)
            .rev()
        {
            let Some(name) = request
                .literals
                .literal_args
                .get(index)
                .and_then(literal_text)
            else {
                uniform_output = None;
                continue;
            };
            if name.trim().eq_ignore_ascii_case("UniformOutput") {
                uniform_output = parse_uniform_output(request, index + 1, &mut diagnostics);
            } else if name.trim().eq_ignore_ascii_case("ErrorHandler") {
                validate_error_handler(request, index + 1, &mut diagnostics);
            } else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CELLFUN-OPTION",
                    "cellfun supports only UniformOutput and ErrorHandler options",
                    index,
                ));
            }
        }
        Self {
            start,
            uniform_output,
            diagnostics,
        }
    }
}

fn trailing_pair_start(request: &CallRequest) -> usize {
    let mut end = request.arguments.len();
    while end >= 3 {
        let candidate = end - 2;
        let literal_name = request
            .literals
            .literal_args
            .get(candidate)
            .and_then(literal_text)
            .is_some();
        let possible_name = candidate >= 2
            && request.arguments.get(candidate).is_some_and(|value| {
                matches!(
                    value.kind,
                    runmat_types::ValueKindFact::String
                        | runmat_types::ValueKindFact::Character
                        | runmat_types::ValueKindFact::Unknown
                )
            });
        if !literal_name && !possible_name {
            break;
        }
        end = candidate;
    }
    end
}

fn parse_uniform_output(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<bool> {
    let literal = request.literals.literal_args.get(index)?;
    if matches!(literal, LiteralValue::Unknown) {
        return None;
    }
    let value = match literal {
        LiteralValue::Bool(value) => Some(*value),
        literal => LiteralContext::numeric_from_literal(literal).and_then(|value| match value {
            0.0 => Some(false),
            1.0 => Some(true),
            _ => None,
        }),
    };
    if value.is_none() {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELLFUN-UNIFORM-OUTPUT",
            "cellfun UniformOutput must be logical true or false, or double one or zero",
            index,
        ));
    }
    value
}

fn validate_error_handler(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = request.arguments.get(index) else {
        return;
    };
    if !matches!(
        value.kind,
        runmat_types::ValueKindFact::Callable(_)
            | runmat_types::ValueKindFact::String
            | runmat_types::ValueKindFact::Character
            | runmat_types::ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CELLFUN-ERROR-HANDLER",
            "cellfun ErrorHandler must be callable",
            index,
        ));
    }
}
