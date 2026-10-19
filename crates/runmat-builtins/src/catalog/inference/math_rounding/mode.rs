use super::super::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, LiteralValue, ValueKindFact};

pub(super) fn validate_round_mode(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(mode) = request.arguments.get(2) else {
        return;
    };
    if !matches!(
        mode.kind,
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-MODE",
            "round requires mode to be \"decimals\" or \"significant\"",
            2,
        ));
    }
    let mode_literal = request
        .literals
        .literal_args
        .get(2)
        .and_then(|literal| match literal {
            LiteralValue::String(value)
            | LiteralValue::Character(value)
            | LiteralValue::Keyword(value) => Some(value.as_str()),
            _ => None,
        });
    if let Some(mode) = mode_literal {
        let mode = mode.trim();
        if !matches!(
            mode.to_ascii_lowercase().as_str(),
            "decimal" | "decimals" | "significant"
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-MODE",
                "round requires mode to be \"decimals\" or \"significant\"",
                2,
            ));
        }
        if mode.eq_ignore_ascii_case("significant")
            && matches!(
                request.literals.literal_args.get(1),
                Some(LiteralValue::Number(value)) if *value <= 0.0
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-DIGITS",
                "round requires positive N for significant-digit rounding",
                1,
            ));
        }
    }
}
