use runmat_parser::{parse_simple as parse, BinOp, Expr, Program, Stmt};

/// Test that semicolon termination is correctly parsed and preserved
#[test]
fn test_expression_without_semicolon() {
    let program = parse("1 + 2").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(
                Expr::Binary(
                    Box::new(Expr::Number("1".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("2".into()))
                ),
                false // false = not semicolon-terminated, output should be shown
            )]
        }
    );
}

/// Test that semicolon termination is correctly parsed and preserved
#[test]
fn test_expression_with_semicolon() {
    let program = parse("1 + 2;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(
                Expr::Binary(
                    Box::new(Expr::Number("1".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("2".into()))
                ),
                true // true = semicolon-terminated, output should be suppressed
            )]
        }
    );
}

/// Test mixed statements with and without semicolons
#[test]
fn test_mixed_semicolon_statements() {
    let program = parse("x = 1; y = 2").unwrap();
    assert_eq!(program.body.len(), 2);

    // First statement is assignment (semicolon-terminated)
    if let Stmt::Assign(name, _, suppressed) = &program.body[0] {
        assert_eq!(name, "x");
        assert!(*suppressed); // Has semicolon
    } else {
        panic!("Expected assignment statement");
    }

    // Second statement is assignment (not semicolon-terminated)
    if let Stmt::Assign(name, _, suppressed) = &program.body[1] {
        assert_eq!(name, "y");
        assert!(!*suppressed); // No semicolon
    } else {
        panic!("Expected assignment statement");
    }
}

/// Test multiple expression statements with different semicolon patterns
#[test]
fn test_multiple_expressions_semicolon_patterns() {
    let program = parse("1 + 2; 3 + 4").unwrap();
    assert_eq!(program.body.len(), 2);

    // First expression: semicolon-terminated (suppressed)
    if let Stmt::ExprStmt(_, suppressed) = &program.body[0] {
        assert!(*suppressed);
    } else {
        panic!("Expected expression statement");
    }

    // Second expression: not semicolon-terminated (shown)
    if let Stmt::ExprStmt(_, suppressed) = &program.body[1] {
        assert!(!*suppressed);
    } else {
        panic!("Expected expression statement");
    }
}

/// Test that function calls preserve semicolon information
#[test]
fn test_function_call_semicolon_preservation() {
    let program = parse("sin(x);").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(
                Expr::FuncCall("sin".into(), vec![Expr::Ident("x".into())]),
                true // Semicolon-terminated
            )]
        }
    );
}

/// Test that matrix literals preserve semicolon information
#[test]
fn test_matrix_literal_semicolon_preservation() {
    let program = parse("[1, 2, 3];").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(
                Expr::Matrix(vec![vec![
                    Expr::Number("1".into()),
                    Expr::Number("2".into()),
                    Expr::Number("3".into())
                ]]),
                true // Semicolon-terminated
            )]
        }
    );
}
