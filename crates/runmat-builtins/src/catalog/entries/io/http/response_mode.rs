//! Shared typed decoding of HTTP response controls for runtime and inference.

use runmat_types::{CallRequest, ValueKindFact};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpResponseMode {
    Auto,
    Json,
    Text,
    Binary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpResponseDialect {
    WebRead,
    WebWrite,
}

impl HttpResponseMode {
    pub fn decode(value: &str, dialect: HttpResponseDialect) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "json" => Some(Self::Json),
            "text" => Some(Self::Text),
            "binary" => Some(Self::Binary),
            "char" | "string" if dialect == HttpResponseDialect::WebRead => Some(Self::Text),
            "octet-stream" | "raw" if dialect == HttpResponseDialect::WebRead => Some(Self::Binary),
            _ => None,
        }
    }
}

/// Only a fully known final ContentType value proves a response class. A
/// dynamic name can override it; a first argument of unknown aggregate kind
/// can also change where name/value pairs begin.
pub fn literal_response_mode(
    request: &CallRequest,
    first_rest: usize,
    dialect: HttpResponseDialect,
) -> Option<HttpResponseMode> {
    let first = request.arguments.get(first_rest)?;
    let mut index = if matches!(first.kind, ValueKindFact::Cell(_)) {
        first_rest + 1
    } else if matches!(first.kind, ValueKindFact::Struct(_)) {
        // A Struct fact may also denote a runtime StructArray. Only the former
        // is consumed as the leading options object by these builtins.
        return None;
    } else if matches!(first.kind, ValueKindFact::Unknown)
        && request.literals.literal_string_at(first_rest).is_none()
    {
        return None;
    } else {
        first_rest
    };
    let mut mode = None;
    while index + 1 < request.arguments.len() {
        match request.literals.literal_string_at(index) {
            Some(name) if name.eq_ignore_ascii_case("contenttype") => {
                mode = request
                    .literals
                    .literal_string_at(index + 1)
                    .and_then(|value| HttpResponseMode::decode(&value, dialect));
            }
            None => mode = None,
            _ => {}
        }
        index += 2;
    }
    mode
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
        StructFact, ValueFact,
    };
    use std::collections::BTreeMap;

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        }
    }

    #[test]
    fn public_response_spellings_are_dialect_specific() {
        assert_eq!(
            HttpResponseMode::decode("  Char  ", HttpResponseDialect::WebRead),
            Some(HttpResponseMode::Text)
        );
        assert_eq!(
            HttpResponseMode::decode("RAW", HttpResponseDialect::WebRead),
            Some(HttpResponseMode::Binary)
        );
        assert_eq!(
            HttpResponseMode::decode("char", HttpResponseDialect::WebWrite),
            None
        );
        assert_eq!(
            HttpResponseMode::decode("unsupported", HttpResponseDialect::WebRead),
            None
        );
    }

    #[test]
    fn final_known_content_type_wins_but_dynamic_tail_or_first_aggregate_does_not() {
        let text = ValueFact::scalar(ValueKindFact::String);
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let options = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            BTreeMap::new(),
            false,
        )));
        let with_options = request(
            vec![text.clone(), options, text.clone(), text.clone()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("ContentType".into()),
                LiteralValue::String("Binary".into()),
            ],
        );
        assert_eq!(
            literal_response_mode(&with_options, 1, HttpResponseDialect::WebRead),
            None
        );
        let dynamic_tail = request(
            vec![
                text.clone(),
                text.clone(),
                text.clone(),
                unknown.clone(),
                unknown.clone(),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("ContentType".into()),
                LiteralValue::String("text".into()),
                LiteralValue::Unknown,
                LiteralValue::Unknown,
            ],
        );
        assert_eq!(
            literal_response_mode(&dynamic_tail, 1, HttpResponseDialect::WebRead),
            None
        );
        let uncertain_first = request(
            vec![text.clone(), unknown, text.clone(), text],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("ContentType".into()),
                LiteralValue::String("binary".into()),
            ],
        );
        assert_eq!(
            literal_response_mode(&uncertain_first, 1, HttpResponseDialect::WebRead),
            None
        );
    }
}
