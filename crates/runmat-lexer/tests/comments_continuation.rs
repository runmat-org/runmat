use runmat_lexer::{tokenize, Token};

#[test]
fn ellipsis_skips_to_end_of_line_even_with_comment() {
    let src = "1 + ... % comment\n  2";
    assert_eq!(
        tokenize(src),
        vec![Token::Integer, Token::Plus, Token::Ellipsis, Token::Integer]
    );
}

#[test]
fn block_comment_skipped() {
    let src = "1 + %{ block\nmore % inside %}\n 2";
    let toks = tokenize(src);
    println!("TOKS: {:?}", toks);
    assert_eq!(toks, vec![Token::Integer, Token::Plus, Token::Integer]);
}

#[test]
fn section_marker_at_line_start() {
    let src = "%% Section Title\n1+2";
    assert_eq!(
        tokenize(src),
        vec![Token::Section, Token::Integer, Token::Plus, Token::Integer]
    );
}

#[test]
fn line_comment_is_ignored() {
    let src = "a + b % comment";
    assert_eq!(tokenize(src), vec![Token::Ident, Token::Plus, Token::Ident]);
}


