use runmat_lexer::{tokenize, Token};

#[test]
fn keywords() {
    let src = "function if else elseif for while break continue return end";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::Function,
            Token::If,
            Token::Else,
            Token::ElseIf,
            Token::For,
            Token::While,
            Token::Break,
            Token::Continue,
            Token::Return,
            Token::End,
        ]
    );
}

#[test]
fn identifiers_and_numbers() {
    let src = "foo bar123 42 3.14";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![Token::Ident, Token::Ident, Token::Integer, Token::Float]
    );
}

#[test]
fn string_literal() {
    let src = "'hello world'";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Str]);
}

#[test]
fn operators() {
    let src = "+ - * / \\ ^ .* ./ .\\ .^ && || & | ~ == ~= <= >= < > = . ; , : ( ) [ ] { } ...";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::Backslash,
            Token::Caret,
            Token::DotStar,
            Token::DotSlash,
            Token::DotBackslash,
            Token::DotCaret,
            Token::AndAnd,
            Token::OrOr,
            Token::And,
            Token::Or,
            Token::Tilde,
            Token::Equal,
            Token::NotEqual,
            Token::LessEqual,
            Token::GreaterEqual,
            Token::Less,
            Token::Greater,
            Token::Assign,
            Token::Dot,
            Token::Semicolon,
            Token::Comma,
            Token::Colon,
            Token::LParen,
            Token::RParen,
            Token::LBracket,
            Token::RBracket,
            Token::LBrace,
            Token::RBrace,
            Token::Ellipsis,
        ]
    );
}

#[test]
fn comments_and_whitespace() {
    let src = "foo % comment\n+";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Ident, Token::Plus]);
}

#[test]
fn uppercase_keywords_are_identifiers() {
    let src = "IF ELSE";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Ident, Token::Ident]);
}

#[test]
fn unknown_character_yields_error() {
    let src = "$";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Error]);
}

#[test]
fn unterminated_string_produces_error() {
    let src = "'oops";
    let tokens = tokenize(src);
    // println!("unterminated tokens: {:?}", tokens);
    assert!(tokens.contains(&Token::Error));
}

#[test]
fn whitespace_variations_are_ignored() {
    let src = "  foo\t\n\r  +\t  bar  ";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Ident, Token::Plus, Token::Ident]);
}

#[test]
fn percent_in_string_is_not_comment() {
    let src = "'100% legit' + 1";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Str, Token::Plus, Token::Integer]);
}

#[test]
fn comments_until_end_of_line() {
    let src = "foo % comment here\n+ bar % another\n- baz";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![Token::Ident, Token::Plus, Token::Ident, Token::Minus, Token::Ident]
    );
}

#[test]
fn ellipsis_line_continuation() {
    let src = "1 + ...\n  2";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Integer, Token::Plus, Token::Ellipsis, Token::Integer]);
}

#[test]
fn ellipsis_inside_string_is_string() {
    let src = "'...'";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Str]);
}

#[test]
fn two_dots_are_two_dots_not_ellipsis() {
    let src = "1..3";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Integer, Token::Dot, Token::Dot, Token::Integer]);
}

#[test]
fn unknown_character_mixed_with_idents() {
    let src = "a!b";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Ident, Token::Error, Token::Ident]);
}

#[test]
fn numeric_exponent_forms() {
    // Integer with exponent and float with exponent
    let src = "1e3 2.0e-2 3E+5 4.5E2";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![Token::Integer, Token::Float, Token::Integer, Token::Float]
    );
}

#[test]
fn indexing_parentheses_and_braces() {
    let src = "A(1,2) A{3} A{1, 2}(3)";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::Ident, Token::LParen, Token::Integer, Token::Comma, Token::Integer, Token::RParen,
            Token::Ident, Token::LBrace, Token::Integer, Token::RBrace,
            Token::Ident, Token::LBrace, Token::Integer, Token::Comma, Token::Integer, Token::RBrace,
            Token::LParen, Token::Integer, Token::RParen,
        ]
    );
}

#[test]
fn matrix_literal_tokens() {
    let src = "[1, 2; 3, 4]";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::LBracket, Token::Integer, Token::Comma, Token::Integer, Token::Semicolon,
            Token::Integer, Token::Comma, Token::Integer, Token::RBracket,
        ]
    );
}

#[test]
#[ignore]
fn transpose_vs_string_disambiguation() {
    // A' ; 'text'
    let src = "A'; 'text'";
    let tokens = tokenize(src);
    // Debug print to aid future regressions
    // println!("tokens: {:?}", tokens);
    assert_eq!(tokens, vec![Token::Ident, Token::Transpose, Token::Str]);
}

#[test]
fn fprintf_with_string_then_transpose_usage() {
    let src = "fprintf('done'); B = A'";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::Ident, Token::LParen, Token::Str, Token::RParen, Token::Semicolon,
            Token::Ident, Token::Assign, Token::Ident, Token::Transpose,
        ]
    );
}

#[test]
fn unterminated_string_in_assignment_produces_error() {
    let src = "A = 'oops";
    let tokens = tokenize(src);
    // println!("unterminated assignment tokens: {:?}", tokens);
    assert!(tokens.contains(&Token::Error));
}

#[test]
fn empty_and_whitespace_only_inputs_produce_no_tokens() {
    let tokens = tokenize("");
    assert!(tokens.is_empty());
    let tokens_ws = tokenize("   \t\n\r  ");
    assert!(tokens_ws.is_empty());
}
