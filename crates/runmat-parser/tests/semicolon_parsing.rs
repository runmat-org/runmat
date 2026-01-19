use runmat_parser::{parse_simple as parse, BinOp, Expr, Program, Span, Stmt};

mod support;
use support::{binary_boxed, expr_stmt, func_call, ident, num, tensor};

fn assert_program_eq(actual: Program, expected: Program) {
    assert_eq!(strip_program(&actual), strip_program(&expected));
}

fn strip_program(program: &Program) -> Program {
    Program {
        body: program.body.iter().map(strip_stmt).collect(),
    }
}

fn strip_stmt(stmt: &Stmt) -> Stmt {
    match stmt {
        Stmt::ExprStmt(expr, suppressed, _) => {
            Stmt::ExprStmt(strip_expr(expr), *suppressed, Span::default())
        }
        Stmt::Assign(name, expr, suppressed, _) => {
            Stmt::Assign(name.clone(), strip_expr(expr), *suppressed, Span::default())
        }
        _ => panic!("unexpected stmt in semicolon tests"),
    }
}

fn strip_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Number(value, _) => Expr::Number(value.clone(), Span::default()),
        Expr::Ident(value, _) => Expr::Ident(value.clone(), Span::default()),
        Expr::Binary(lhs, op, rhs, _) => Expr::Binary(
            Box::new(strip_expr(lhs)),
            *op,
            Box::new(strip_expr(rhs)),
            Span::default(),
        ),
        Expr::FuncCall(name, args, _) => Expr::FuncCall(
            name.clone(),
            args.iter().map(strip_expr).collect(),
            Span::default(),
        ),
        Expr::Tensor(rows, _) => Expr::Tensor(strip_rows(rows), Span::default()),
        Expr::String(value, _) => Expr::String(value.clone(), Span::default()),
        _ => panic!("unexpected expr in semicolon tests"),
    }
}

fn strip_rows(rows: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    rows.iter()
        .map(|row| row.iter().map(strip_expr).collect())
        .collect()
}

/// Test that semicolon termination is correctly parsed and preserved
#[test]
fn test_expression_without_semicolon() {
    let program = parse("1 + 2").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("1".to_string())),
                    BinOp::Add,
                    Box::new(num("2".to_string())),
                ),
                false, // false = not semicolon-terminated, output should be shown
            )]
        }
    );
}

/// Test that semicolon termination is correctly parsed and preserved
#[test]
fn test_expression_with_semicolon() {
    let program = parse("1 + 2;").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("1".to_string())),
                    BinOp::Add,
                    Box::new(num("2".to_string())),
                ),
                true, // true = semicolon-terminated, output should be suppressed
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
    if let Stmt::Assign(name, _, suppressed, _) = &program.body[0] {
        assert_eq!(name, "x");
        assert!(*suppressed); // Has semicolon
    } else {
        panic!("Expected assignment statement");
    }

    // Second statement is assignment (not semicolon-terminated)
    if let Stmt::Assign(name, _, suppressed, _) = &program.body[1] {
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
    if let Stmt::ExprStmt(_, suppressed, _) = &program.body[0] {
        assert!(*suppressed);
    } else {
        panic!("Expected expression statement");
    }

    // Second expression: not semicolon-terminated (shown)
    if let Stmt::ExprStmt(_, suppressed, _) = &program.body[1] {
        assert!(!*suppressed);
    } else {
        panic!("Expected expression statement");
    }
}

/// Test newline-separated statements behave like implicit separators
#[test]
fn test_newline_separates_statements() {
    let program = parse("x = 1\ny = x + 1").unwrap();
    assert_eq!(program.body.len(), 2);

    match &program.body[0] {
        Stmt::Assign(name, _, suppressed, _) => {
            assert_eq!(name, "x");
            assert!(!*suppressed);
        }
        other => panic!("expected assignment for first stmt, got {other:?}"),
    }

    match &program.body[1] {
        Stmt::Assign(name, _, suppressed, _) => {
            assert_eq!(name, "y");
            assert!(!*suppressed);
        }
        other => panic!("expected assignment for second stmt, got {other:?}"),
    }
}

/// Test that function calls preserve semicolon information
#[test]
fn test_function_call_semicolon_preservation() {
    let program = parse("sin(x);").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                func_call("sin".to_string(), vec![ident("x".to_string())]),
                true, // Semicolon-terminated
            )]
        }
    );
}

/// Test that matrix literals preserve semicolon information
#[test]
fn test_matrix_literal_semicolon_preservation() {
    let program = parse("[1, 2, 3];").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                tensor(vec![vec![
                    num("1".to_string()),
                    num("2".to_string()),
                    num("3".to_string()),
                ]]),
                true, // Semicolon-terminated
            )]
        }
    );
}
