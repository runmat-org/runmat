use rustmat_parser::{parse, BinOp, Expr, Program, Stmt};

#[test]
fn parse_expression() {
    let program = parse("1 + 2 * 3").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("1".into())),
                BinOp::Add,
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("2".into())),
                    BinOp::Mul,
                    Box::new(Expr::Number("3".into()))
                ))
            ))]
        }
    );
}

#[test]
fn parse_assignment() {
    let program = parse("x = 4 + 5;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::Assign(
                "x".into(),
                Expr::Binary(
                    Box::new(Expr::Number("4".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("5".into()))
                )
            )]
        }
    );
}

#[test]
fn precedence_and_associativity() {
    let program = parse("1 - 2 - 3").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("1".into())),
                    BinOp::Sub,
                    Box::new(Expr::Number("2".into()))
                )),
                BinOp::Sub,
                Box::new(Expr::Number("3".into()))
            ))]
        }
    );
}

#[test]
fn parentheses_override_precedence() {
    let program = parse("1 * (2 + 3)").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("1".into())),
                BinOp::Mul,
                Box::new(Expr::Binary(
                    Box::new(Expr::Number("2".into())),
                    BinOp::Add,
                    Box::new(Expr::Number("3".into()))
                ))
            ))]
        }
    );
}

#[test]
fn multiple_statements() {
    let program = parse("x = 1; y = 2;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![
                Stmt::Assign("x".into(), Expr::Number("1".into())),
                Stmt::Assign("y".into(), Expr::Number("2".into()))
            ]
        }
    );
}

#[test]
fn trailing_semicolon_is_allowed() {
    let program = parse("1 + 2;").unwrap();
    assert_eq!(
        program,
        Program {
            body: vec![Stmt::ExprStmt(Expr::Binary(
                Box::new(Expr::Number("1".into())),
                BinOp::Add,
                Box::new(Expr::Number("2".into()))
            ))]
        }
    );
}

#[test]
fn final_semicolon_not_required() {
    let program = parse("1 + 2").unwrap();
    assert_eq!(program.body.len(), 1);
}

#[test]
fn empty_input_yields_empty_program() {
    let program = parse("").unwrap();
    assert!(program.body.is_empty());
}

#[test]
fn missing_closing_paren_produces_error() {
    assert!(parse("(1 + 2").is_err());
}

#[test]
fn invalid_token_produces_error() {
    assert!(parse("1 + $").is_err());
}

#[test]
fn incomplete_expression_produces_error() {
    assert!(parse("1 +").is_err());
}
