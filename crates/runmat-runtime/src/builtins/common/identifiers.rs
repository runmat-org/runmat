//! Shared MATLAB identifier and keyword compatibility helpers.

/// Current MATLAB maximum identifier length reported by `namelengthmax`.
pub const MATLAB_NAME_LENGTH_MAX: usize = 2048;

/// MATLAB reserved keywords accepted by `iskeyword` and rejected by `isvarname`.
pub const MATLAB_KEYWORDS: &[&str] = &[
    "break",
    "case",
    "catch",
    "classdef",
    "continue",
    "else",
    "elseif",
    "end",
    "for",
    "function",
    "global",
    "if",
    "otherwise",
    "parfor",
    "persistent",
    "return",
    "spmd",
    "switch",
    "try",
    "while",
];

pub fn is_matlab_keyword(name: &str) -> bool {
    MATLAB_KEYWORDS.contains(&name)
}

pub fn is_valid_varname(name: &str) -> bool {
    if name.chars().count() > MATLAB_NAME_LENGTH_MAX || is_matlab_keyword(name) {
        return false;
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_alphabetic() {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_list_matches_current_core_language_keywords() {
        assert_eq!(MATLAB_KEYWORDS.len(), 20);
        assert!(is_matlab_keyword("classdef"));
        assert!(is_matlab_keyword("parfor"));
        assert!(!is_matlab_keyword("plot"));
    }

    #[test]
    fn valid_varname_uses_namelengthmax() {
        assert!(is_valid_varname(&"a".repeat(MATLAB_NAME_LENGTH_MAX)));
        assert!(!is_valid_varname(&"a".repeat(MATLAB_NAME_LENGTH_MAX + 1)));
    }
}
