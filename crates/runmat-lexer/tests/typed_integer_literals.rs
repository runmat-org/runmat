use runmat_lexer::{tokenize, Token};

#[test]
fn radix_integer_literals_are_single_numeric_tokens() {
    assert_eq!(
        tokenize("0x2A 0XFFs8 0b101010 0B1u64"),
        vec![
            Token::RadixInteger,
            Token::RadixInteger,
            Token::RadixInteger,
            Token::RadixInteger,
        ]
    );
}

#[test]
fn invalid_radix_content_stays_whole_for_parser_diagnostics() {
    assert_eq!(
        tokenize("0xGG 0b102 0xFFu9"),
        vec![
            Token::RadixInteger,
            Token::RadixInteger,
            Token::RadixInteger,
        ]
    );
}
