use runmat_lexer::{tokenize, Token};

#[test]
fn transpose_after_ident_then_semicolon() {
    let input = "B = A';";
    let tokens = tokenize(input);
    // Expect: Ident(B), Assign, Ident(A), Transpose, Semicolon
    assert_eq!(
        tokens,
        vec![
            Token::Ident,
            Token::Assign,
            Token::Ident,
            Token::Transpose,
            Token::Semicolon,
        ]
    );
}

#[test]
fn simple_string_literal() {
    let input = "fprintf('done');";
    let tokens = tokenize(input);
    // Expect: Ident, LParen, Str, RParen, Semicolon
    assert_eq!(
        tokens,
        vec![
            Token::Ident,
            Token::LParen,
            Token::Str,
            Token::RParen,
            Token::Semicolon,
        ]
    );
}

#[test]
fn debug_print_tokens_for_apostrophe_case() {
    let input = "B = A';";
    let tokens = tokenize(input);
    println!("TOKENS: {tokens:?}");
}

#[test]
fn test_issue_isolation() {
    // Test each step to isolate where it breaks
    let test_cases = vec![
        "A",
        "A'",
        "A';",
        "A'; ",
        "A'; t",
        "A'; tr",
        "A'; tra",
        "A'; trac",
        "A'; trace",
    ];

    for test_case in test_cases {
        println!("\nTest: '{test_case}'");
        let tokens = runmat_lexer::tokenize_detailed(test_case);
        for (i, token) in tokens.iter().enumerate() {
            println!("  {}: {:?} ('{}')", i, token.token, token.lexeme);
            if token.token == runmat_lexer::Token::Error {
                println!("    ^ ERROR - lexer broke here");
                break; // Stop at first error to see exactly where it breaks
            }
        }
    }
}

#[test]
fn debug_complex_transpose_case() {
    let input = "tic; A = randn(1000, 1000); B = A * A'; trace(B)";
    let tokens = runmat_lexer::tokenize_detailed(input);

    println!("Input: {input}");
    for (i, token) in tokens.iter().enumerate() {
        println!("{}: {:?} ('{}')", i, token.token, token.lexeme);
        if token.token == runmat_lexer::Token::Error {
            println!("ERROR FOUND AT TOKEN {}: '{}'", i, token.lexeme);
        }
    }

    // Test simpler cases
    println!("\nSimpler test cases:");

    let test_cases = vec![
        "A'",
        "A' ",
        "A'; ",
        "A'; trace",
        "'; trace",
        " trace",
        "trace",
    ];

    for test_case in test_cases {
        println!("\nTest: '{test_case}'");
        let tokens = runmat_lexer::tokenize_detailed(test_case);
        for (i, token) in tokens.iter().enumerate() {
            println!("  {}: {:?} ('{}')", i, token.token, token.lexeme);
        }
    }

    // Test minimal failing case
    println!("\nMinimal reproduction:");
    let failing = "A'; trace";
    println!("Full input: '{failing}'");
    let tokens = runmat_lexer::tokenize_detailed(failing);
    for (i, token) in tokens.iter().enumerate() {
        println!(
            "  {}: {:?} ('{}') span: {}..{}",
            i, token.token, token.lexeme, token.start, token.end
        );
        if token.token == runmat_lexer::Token::Error {
            println!("    ^ ERROR");
        }
    }
}

#[test]
fn debug_string_after_semicolon_case() {
    let input = "A'; 'text'";
    let tokens = runmat_lexer::tokenize_detailed(input);
    println!("Input: {input}");
    for (i, token) in tokens.iter().enumerate() {
        println!("{}: {:?} ('{}')", i, token.token, token.lexeme);
    }
}
