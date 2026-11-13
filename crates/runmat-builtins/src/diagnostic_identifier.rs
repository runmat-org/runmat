//! Allocation-free diagnostic identifier grammar shared by catalog inference
//! and host runtime dispatch. Legacy loose forms remain explicitly separate.

pub fn is_qualified(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.contains(':') && trimmed.split(':').all(is_field)
}

pub fn is_unqualified_field(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.contains(':') && is_field(trimmed)
}

pub fn is_field(field: &str) -> bool {
    let mut chars = field.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && chars.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// The older `error` RunMat-only form accepts dots and leading digits too.
pub fn is_error_legacy_unqualified(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && !trimmed.contains(char::is_whitespace)
        && trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_and_legacy_grammars_keep_whitespace_and_field_rules() {
        assert!(is_qualified(" Acme:good_2 "));
        assert!(!is_qualified(" Acme:9bad "));
        assert!(!is_qualified("Acme::bad"));
        assert!(is_unqualified_field(" good_2 "));
        assert!(!is_unqualified_field("9bad"));
        assert!(is_error_legacy_unqualified("9.bad"));
        assert!(!is_error_legacy_unqualified("bad name"));
    }
}
