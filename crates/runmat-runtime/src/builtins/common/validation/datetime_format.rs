//! One narrow MATLAB datetime input-format translation used by datetime and
//! tabular variable import. Unsupported patterns fail at their caller boundary.

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

pub(crate) fn datetime_format_to_strftime(format: &str) -> String {
    let mut out = format.to_string();
    for (src, dst) in [
        ("yyyy", "%Y"),
        ("MMM", "%b"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    ] {
        out = out.replace(src, dst);
    }
    out
}

pub(crate) fn supported_datetime_input_format(format: &str) -> bool {
    let mut rest = format;
    let mut year = false;
    let mut month = false;
    let mut day = false;
    while !rest.is_empty() {
        if let Some(token) = ["yyyy", "MMM", "MM", "dd", "HH", "mm", "ss"]
            .into_iter()
            .find(|token| rest.starts_with(token))
        {
            year |= token == "yyyy";
            month |= matches!(token, "MMM" | "MM");
            day |= token == "dd";
            rest = &rest[token.len()..];
        } else {
            let Some(ch) = rest.chars().next() else { break };
            if !matches!(ch, '-' | '/' | '.' | ':' | ' ' | ',') {
                return false;
            }
            rest = &rest[ch.len_utf8()..];
        }
    }
    year && month && day
}

pub(crate) fn parse_datetime_with_input_format(
    text: &str,
    format: &str,
) -> Option<(NaiveDateTime, bool)> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let chrono_format = datetime_format_to_strftime(format);
    if let Ok(value) = NaiveDateTime::parse_from_str(trimmed, &chrono_format) {
        return Some((value, true));
    }
    NaiveDate::parse_from_str(trimmed, &chrono_format)
        .ok()
        .map(|date| (date.and_time(NaiveTime::MIN), false))
}
