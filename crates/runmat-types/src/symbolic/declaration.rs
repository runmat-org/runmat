use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicDeclaration {
    pub name: String,
    pub parameters: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicDeclarationError {
    Empty,
    InvalidName,
    InvalidParameter,
    DuplicateParameter,
    EmptyParameterList,
    UnexpectedSyntax,
}

impl fmt::Display for SymbolicDeclarationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty symbolic declaration"),
            Self::InvalidName => write!(f, "invalid symbolic name"),
            Self::InvalidParameter => write!(f, "invalid symbolic parameter"),
            Self::DuplicateParameter => write!(f, "duplicate symbolic function parameter"),
            Self::EmptyParameterList => write!(
                f,
                "symbolic function declaration requires at least one parameter"
            ),
            Self::UnexpectedSyntax => write!(f, "invalid symbolic function declaration syntax"),
        }
    }
}

impl std::error::Error for SymbolicDeclarationError {}

pub fn parse_symbolic_declaration(
    text: &str,
) -> Result<SymbolicDeclaration, SymbolicDeclarationError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(SymbolicDeclarationError::Empty);
    }
    let Some(open) = trimmed.find('(') else {
        return is_valid_symbolic_identifier(trimmed)
            .then(|| SymbolicDeclaration {
                name: trimmed.to_string(),
                parameters: Vec::new(),
            })
            .ok_or(SymbolicDeclarationError::InvalidName);
    };
    if !trimmed.ends_with(')') {
        return Err(SymbolicDeclarationError::UnexpectedSyntax);
    }
    let inner = &trimmed[open + 1..trimmed.len() - 1];
    if inner.contains(['(', ')']) {
        return Err(SymbolicDeclarationError::UnexpectedSyntax);
    }
    let name = trimmed[..open].trim();
    if !is_valid_symbolic_identifier(name) {
        return Err(SymbolicDeclarationError::InvalidName);
    }
    if inner.trim().is_empty() {
        return Err(SymbolicDeclarationError::EmptyParameterList);
    }
    let mut parameters = Vec::new();
    for parameter in inner.split(',').map(str::trim) {
        if !is_valid_symbolic_identifier(parameter) {
            return Err(SymbolicDeclarationError::InvalidParameter);
        }
        if parameters.iter().any(|existing| existing == parameter) {
            return Err(SymbolicDeclarationError::DuplicateParameter);
        }
        parameters.push(parameter.to_string());
    }
    Ok(SymbolicDeclaration {
        name: name.to_string(),
        parameters,
    })
}

pub fn is_valid_symbolic_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic() && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

pub fn symbolic_declaration_tokens(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = None;
    let mut paren_depth = 0usize;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() && paren_depth == 0 {
            if let Some(token_start) = start.take() {
                tokens.push(&text[token_start..idx]);
            }
            continue;
        }
        start.get_or_insert(idx);
        match ch {
            '(' => paren_depth = paren_depth.saturating_add(1),
            ')' => paren_depth = paren_depth.saturating_sub(1),
            _ => {}
        }
    }
    if let Some(token_start) = start {
        tokens.push(&text[token_start..]);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_tokenizes_declarations() {
        assert_eq!(
            parse_symbolic_declaration("Y(X)").unwrap().parameters,
            vec!["X"]
        );
        assert_eq!(
            symbolic_declaration_tokens("x f(a, b) real"),
            vec!["x", "f(a, b)", "real"]
        );
    }

    #[test]
    fn rejects_ambiguous_declarations() {
        assert_eq!(
            parse_symbolic_declaration("f()").unwrap_err(),
            SymbolicDeclarationError::EmptyParameterList
        );
        assert_eq!(
            parse_symbolic_declaration("f(x,x)").unwrap_err(),
            SymbolicDeclarationError::DuplicateParameter
        );
    }
}
