pub const PACKAGE_SEGMENT_MAX_LEN: usize = 64;
pub const REGISTRY_SEGMENT_MAX_LEN: usize = 64;
pub const PACKAGE_ALIAS_MAX_LEN: usize = 64;

pub(crate) fn validate_canonical_segment(
    value: &str,
    kind: &'static str,
    max_len: usize,
) -> Result<(), crate::IdentityError> {
    if value.is_empty() {
        return Err(invalid_name(kind, value, "must not be empty"));
    }
    if value.len() > max_len {
        return Err(invalid_name(kind, value, "exceeds the maximum length"));
    }
    if !value.is_ascii() {
        return Err(invalid_name(
            kind,
            value,
            "must contain ASCII characters only",
        ));
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(invalid_name(
            kind,
            value,
            "must use canonical lowercase ASCII",
        ));
    }
    let first = value.as_bytes()[0];
    let last = value.as_bytes()[value.len() - 1];
    if !first.is_ascii_alphanumeric() || !last.is_ascii_alphanumeric() {
        return Err(invalid_name(
            kind,
            value,
            "must begin and end with an ASCII alphanumeric",
        ));
    }
    if value
        .bytes()
        .any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(invalid_name(
            kind,
            value,
            "may contain only lowercase ASCII alphanumerics, `-`, `_`, or `.`",
        ));
    }
    if is_reserved_portable_name(value) {
        return Err(invalid_name(
            kind,
            value,
            "is reserved by portable filesystem policy",
        ));
    }
    Ok(())
}

fn invalid_name(kind: &'static str, value: &str, reason: &'static str) -> crate::IdentityError {
    crate::IdentityError::InvalidName {
        kind,
        value: value.to_string(),
        reason,
    }
}

fn is_reserved_portable_name(value: &str) -> bool {
    matches!(value, "con" | "prn" | "aux" | "nul")
        || value
            .strip_prefix("com")
            .or_else(|| value.strip_prefix("lpt"))
            .is_some_and(|suffix| suffix.len() == 1 && matches!(suffix.as_bytes()[0], b'1'..=b'9'))
}
