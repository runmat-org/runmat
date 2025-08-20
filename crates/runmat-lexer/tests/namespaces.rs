use runmat_lexer::{tokenize, Token};

#[test]
fn import_keyword_and_package_path_tokens() {
    let src = "import pkg.subpkg.Class";
    assert_eq!(
        tokenize(src),
        vec![
            Token::Import,
            Token::Ident,
            Token::Dot,
            Token::Ident,
            Token::Dot,
            Token::Ident
        ]
    );
}

#[test]
fn import_wildcard() {
    let src = "import pkg.*";
    // Lexer tokenizes ".*" as a single DotStar token
    assert_eq!(
        tokenize(src),
        vec![Token::Import, Token::Ident, Token::DotStar]
    );
}

#[test]
fn meta_class_query_operator() {
    let src = "?MyClass";
    assert_eq!(tokenize(src), vec![Token::Question, Token::Ident]);
}
