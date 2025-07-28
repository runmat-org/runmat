use rustmat_lexer::tokenize;

/// Tokenize the input string and return a space separated string of token names.
pub fn format_tokens(input: &str) -> String {
    let tokens = tokenize(input);
    tokens
        .into_iter()
        .map(|t| format!("{:?}", t))
        .collect::<Vec<_>>()
        .join(" ")
}
