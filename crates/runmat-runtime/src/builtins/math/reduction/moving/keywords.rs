//! Borrow the original text storage; count-window parsing needs no String copy.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Keyword {
    Omit,
    Include,
    Endpoints,
    SamplePoints,
    Shrink,
    Discard,
    Fill,
    Other,
}

pub(super) fn keyword(value: &Value) -> Option<Keyword> {
    let is_text = matches!(value, Value::String(_))
        || matches!(value, Value::StringArray(array) if array.data().len() == 1)
        || matches!(value, Value::CharArray(array) if array.rows == 1);
    if !is_text {
        return None;
    }
    for (text, keyword) in [
        ("omitnan", Keyword::Omit),
        ("omitmissing", Keyword::Omit),
        ("includenan", Keyword::Include),
        ("includemissing", Keyword::Include),
        ("endpoints", Keyword::Endpoints),
        ("samplepoints", Keyword::SamplePoints),
        ("shrink", Keyword::Shrink),
        ("discard", Keyword::Discard),
        ("fill", Keyword::Fill),
    ] {
        if text_eq(value, text) {
            return Some(keyword);
        }
    }
    Some(Keyword::Other)
}

fn text_eq(value: &Value, expected: &str) -> bool {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            text.as_str().eq_ignore_ascii_case(expected)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            array.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .eq_ignore_ascii_case(expected)
        }
        Value::CharArray(array) if array.rows == 1 => array
            .data()
            .iter()
            .map(|unit| {
                let unit = unit.raw();
                if (65..=90).contains(&unit) {
                    unit + 32
                } else {
                    unit
                }
            })
            .eq(expected.encode_utf16()),
        _ => false,
    }
}
