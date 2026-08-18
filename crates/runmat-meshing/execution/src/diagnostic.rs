const MAX_DIAGNOSTIC_BYTES: usize = 2048;

pub(crate) fn bounded_diagnostic_text(value: &str, fallback: &str) -> String {
    let text = value
        .chars()
        .map(|character| {
            if character.is_ascii() && !character.is_ascii_control() {
                character
            } else {
                '?'
            }
        })
        .take(MAX_DIAGNOSTIC_BYTES)
        .collect::<String>();
    let text = text.trim();
    if text.is_empty() {
        fallback.into()
    } else {
        text.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_text_is_bounded_printable_and_nonempty() {
        let input = "x".repeat(MAX_DIAGNOSTIC_BYTES * 2);
        let text = bounded_diagnostic_text(&input, "fallback");
        assert_eq!(text.len(), MAX_DIAGNOSTIC_BYTES);
        assert!(text
            .bytes()
            .all(|byte| byte.is_ascii() && !byte.is_ascii_control()));
        assert_eq!(bounded_diagnostic_text("\n\t", "fallback"), "??");
        assert_eq!(bounded_diagnostic_text("   ", "fallback"), "fallback");
    }
}
