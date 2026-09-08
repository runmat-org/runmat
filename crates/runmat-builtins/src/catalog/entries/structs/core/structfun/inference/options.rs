use crate::catalog::inference::{argument_error, literal_text};
use runmat_types::{CallRequest, InferenceDiagnostic, LiteralValue, ValueKindFact};

pub(super) struct Options {
    pub(super) uniform: Option<bool>,
    pub(super) diagnostics: Vec<InferenceDiagnostic>,
}

impl Options {
    pub(super) fn parse(request: &CallRequest) -> Self {
        let mut diagnostics = Vec::new();
        let mut uniform = Some(true);
        if request.arguments.len() < 2 {
            return Self {
                uniform,
                diagnostics,
            };
        }
        let options = &request.arguments[2..];
        if !options.len().is_multiple_of(2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRUCTFUN-OPTIONS",
                "structfun options must be complete name-value pairs",
                request.arguments.len() - 1,
            ));
            return Self {
                uniform,
                diagnostics,
            };
        }
        for index in (2..request.arguments.len()).step_by(2) {
            let literal = request
                .literals
                .literal_args
                .get(index)
                .and_then(literal_text);
            match literal.as_deref().map(str::to_ascii_lowercase).as_deref() {
                Some("uniformoutput") => uniform = literal_bool(request, index + 1),
                Some("errorhandler") => validate_handler(request, index + 1, &mut diagnostics),
                Some(name) => diagnostics.push(argument_error(
                    "RM-CATALOG-STRUCTFUN-OPTION",
                    format!("unknown structfun option '{name}'"),
                    index,
                )),
                None => match request.arguments.get(index).map(|value| &value.kind) {
                    Some(
                        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown,
                    ) => {
                        uniform = None;
                    }
                    Some(_) => diagnostics.push(argument_error(
                        "RM-CATALOG-STRUCTFUN-OPTION",
                        "structfun option names must be scalar text",
                        index,
                    )),
                    None => {}
                },
            }
        }
        Self {
            uniform,
            diagnostics,
        }
    }
}

fn literal_bool(request: &CallRequest, index: usize) -> Option<bool> {
    match request.literals.literal_args.get(index) {
        Some(LiteralValue::Bool(value)) => Some(*value),
        Some(LiteralValue::Number(value)) => Some(*value != 0.0),
        Some(LiteralValue::String(value))
            if value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("on") =>
        {
            Some(true)
        }
        Some(LiteralValue::String(value))
            if value.eq_ignore_ascii_case("false") || value.eq_ignore_ascii_case("off") =>
        {
            Some(false)
        }
        _ => None,
    }
}

fn validate_handler(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = request.arguments.get(index) else {
        return;
    };
    if !matches!(
        value.kind,
        ValueKindFact::Callable(_)
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRUCTFUN-HANDLER",
            "structfun ErrorHandler must be callable",
            index,
        ));
    }
}
