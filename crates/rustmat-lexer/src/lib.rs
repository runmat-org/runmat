use logos::Logos;

#[derive(Logos, Debug, PartialEq)]
pub enum Token {
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,
    #[regex(r"[0-9]+(\.[0-9]+)?")]
    Number,
    #[token("+")]
    Plus,
}

pub fn tokenize(input: &str) -> Vec<Token> {
    Token::lexer(input).filter_map(|tok| tok.ok()).collect()
}
