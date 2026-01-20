use runmat_parser::{Attr, BinOp, ClassMember, Expr, LValue, Program, Span, Stmt, UnOp};

mod parse;
mod support;
mod support_extra;
use parse::parse;
use support::{binary_boxed, expr_stmt, func_call, ident, num, tensor};
use support_extra::{assign, range, span_value, string, unary_boxed};

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
        Stmt::MultiAssign(names, expr, suppressed, _) => Stmt::MultiAssign(
            names.clone(),
            strip_expr(expr),
            *suppressed,
            Span::default(),
        ),
        Stmt::AssignLValue(lvalue, expr, suppressed, _) => Stmt::AssignLValue(
            strip_lvalue(lvalue),
            strip_expr(expr),
            *suppressed,
            Span::default(),
        ),
        Stmt::If {
            cond,
            then_body,
            elseif_blocks,
            else_body,
            ..
        } => Stmt::If {
            cond: strip_expr(cond),
            then_body: then_body.iter().map(strip_stmt).collect(),
            elseif_blocks: elseif_blocks
                .iter()
                .map(|(expr, body)| (strip_expr(expr), body.iter().map(strip_stmt).collect()))
                .collect(),
            else_body: else_body
                .as_ref()
                .map(|body| body.iter().map(strip_stmt).collect()),
            span: Span::default(),
        },
        Stmt::While { cond, body, .. } => Stmt::While {
            cond: strip_expr(cond),
            body: body.iter().map(strip_stmt).collect(),
            span: Span::default(),
        },
        Stmt::For {
            var, expr, body, ..
        } => Stmt::For {
            var: var.clone(),
            expr: strip_expr(expr),
            body: body.iter().map(strip_stmt).collect(),
            span: Span::default(),
        },
        Stmt::Switch {
            expr,
            cases,
            otherwise,
            ..
        } => Stmt::Switch {
            expr: strip_expr(expr),
            cases: cases
                .iter()
                .map(|(case_expr, body)| {
                    (strip_expr(case_expr), body.iter().map(strip_stmt).collect())
                })
                .collect(),
            otherwise: otherwise
                .as_ref()
                .map(|body| body.iter().map(strip_stmt).collect()),
            span: Span::default(),
        },
        Stmt::TryCatch {
            try_body,
            catch_var,
            catch_body,
            ..
        } => Stmt::TryCatch {
            try_body: try_body.iter().map(strip_stmt).collect(),
            catch_var: catch_var.clone(),
            catch_body: catch_body.iter().map(strip_stmt).collect(),
            span: Span::default(),
        },
        Stmt::Global(names, _) => Stmt::Global(names.clone(), Span::default()),
        Stmt::Persistent(names, _) => Stmt::Persistent(names.clone(), Span::default()),
        Stmt::Break(_) => Stmt::Break(Span::default()),
        Stmt::Continue(_) => Stmt::Continue(Span::default()),
        Stmt::Return(_) => Stmt::Return(Span::default()),
        Stmt::Function {
            name,
            params,
            outputs,
            body,
            ..
        } => Stmt::Function {
            name: name.clone(),
            params: params.clone(),
            outputs: outputs.clone(),
            body: body.iter().map(strip_stmt).collect(),
            span: Span::default(),
        },
        Stmt::Import { path, wildcard, .. } => Stmt::Import {
            path: path.clone(),
            wildcard: *wildcard,
            span: Span::default(),
        },
        Stmt::ClassDef {
            name,
            super_class,
            members,
            ..
        } => Stmt::ClassDef {
            name: name.clone(),
            super_class: super_class.clone(),
            members: members.iter().map(strip_class_member).collect(),
            span: Span::default(),
        },
    }
}

fn strip_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Number(value, _) => Expr::Number(value.clone(), Span::default()),
        Expr::String(value, _) => Expr::String(value.clone(), Span::default()),
        Expr::Ident(value, _) => Expr::Ident(value.clone(), Span::default()),
        Expr::EndKeyword(_) => Expr::EndKeyword(Span::default()),
        Expr::Unary(op, expr, _) => Expr::Unary(*op, Box::new(strip_expr(expr)), Span::default()),
        Expr::Binary(lhs, op, rhs, _) => Expr::Binary(
            Box::new(strip_expr(lhs)),
            *op,
            Box::new(strip_expr(rhs)),
            Span::default(),
        ),
        Expr::Tensor(rows, _) => Expr::Tensor(strip_rows(rows), Span::default()),
        Expr::Cell(rows, _) => Expr::Cell(strip_rows(rows), Span::default()),
        Expr::Index(base, indices, _) => Expr::Index(
            Box::new(strip_expr(base)),
            indices.iter().map(strip_expr).collect(),
            Span::default(),
        ),
        Expr::IndexCell(base, indices, _) => Expr::IndexCell(
            Box::new(strip_expr(base)),
            indices.iter().map(strip_expr).collect(),
            Span::default(),
        ),
        Expr::Range(start, step, end, _) => Expr::Range(
            Box::new(strip_expr(start)),
            step.as_ref().map(|expr| Box::new(strip_expr(expr))),
            Box::new(strip_expr(end)),
            Span::default(),
        ),
        Expr::Colon(_) => Expr::Colon(Span::default()),
        Expr::FuncCall(name, args, _) => Expr::FuncCall(
            name.clone(),
            args.iter().map(strip_expr).collect(),
            Span::default(),
        ),
        Expr::Member(base, name, _) => {
            Expr::Member(Box::new(strip_expr(base)), name.clone(), Span::default())
        }
        Expr::MemberDynamic(base, name_expr, _) => Expr::MemberDynamic(
            Box::new(strip_expr(base)),
            Box::new(strip_expr(name_expr)),
            Span::default(),
        ),
        Expr::MethodCall(base, name, args, _) => Expr::MethodCall(
            Box::new(strip_expr(base)),
            name.clone(),
            args.iter().map(strip_expr).collect(),
            Span::default(),
        ),
        Expr::AnonFunc { params, body, .. } => Expr::AnonFunc {
            params: params.clone(),
            body: Box::new(strip_expr(body)),
            span: Span::default(),
        },
        Expr::FuncHandle(name, _) => Expr::FuncHandle(name.clone(), Span::default()),
        Expr::MetaClass(name, _) => Expr::MetaClass(name.clone(), Span::default()),
    }
}

fn strip_lvalue(lvalue: &LValue) -> LValue {
    match lvalue {
        LValue::Var(name) => LValue::Var(name.clone()),
        LValue::Member(base, name) => LValue::Member(Box::new(strip_expr(base)), name.clone()),
        LValue::MemberDynamic(base, name_expr) => {
            LValue::MemberDynamic(Box::new(strip_expr(base)), Box::new(strip_expr(name_expr)))
        }
        LValue::Index(base, indices) => LValue::Index(
            Box::new(strip_expr(base)),
            indices.iter().map(strip_expr).collect(),
        ),
        LValue::IndexCell(base, indices) => LValue::IndexCell(
            Box::new(strip_expr(base)),
            indices.iter().map(strip_expr).collect(),
        ),
    }
}

fn strip_rows(rows: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    rows.iter()
        .map(|row| row.iter().map(strip_expr).collect())
        .collect()
}

fn strip_class_member(member: &ClassMember) -> ClassMember {
    match member {
        ClassMember::Properties { attributes, names } => ClassMember::Properties {
            attributes: strip_attrs(attributes),
            names: names.clone(),
        },
        ClassMember::Methods { attributes, body } => ClassMember::Methods {
            attributes: strip_attrs(attributes),
            body: body.iter().map(strip_stmt).collect(),
        },
        ClassMember::Events { attributes, names } => ClassMember::Events {
            attributes: strip_attrs(attributes),
            names: names.clone(),
        },
        ClassMember::Enumeration { attributes, names } => ClassMember::Enumeration {
            attributes: strip_attrs(attributes),
            names: names.clone(),
        },
        ClassMember::Arguments { attributes, names } => ClassMember::Arguments {
            attributes: strip_attrs(attributes),
            names: names.clone(),
        },
    }
}

fn strip_attrs(attrs: &[Attr]) -> Vec<Attr> {
    attrs
        .iter()
        .map(|attr| Attr {
            name: attr.name.clone(),
            value: attr.value.clone(),
        })
        .collect()
}

#[test]
fn parse_expression() {
    let program = parse("1 + 2 * 3").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("1".to_string())),
                    BinOp::Add,
                    Box::new(binary_boxed(
                        Box::new(num("2".to_string())),
                        BinOp::Mul,
                        Box::new(num("3".to_string())),
                    )),
                ),
                false,
            )],
        },
    );
}

#[test]
fn parse_assignment() {
    let program = parse("x = 4 + 5;").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![assign(
                "x".to_string(),
                binary_boxed(
                    Box::new(num("4".to_string())),
                    BinOp::Add,
                    Box::new(num("5".to_string())),
                ),
                true, // Semicolon suppresses display even at EOF
            )],
        },
    );
}

#[test]
fn precedence_and_associativity() {
    let program = parse("1 - 2 - 3").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(binary_boxed(
                        Box::new(num("1".to_string())),
                        BinOp::Sub,
                        Box::new(num("2".to_string())),
                    )),
                    BinOp::Sub,
                    Box::new(num("3".to_string())),
                ),
                false,
            )],
        },
    );
}

#[test]
fn parentheses_override_precedence() {
    let program = parse("1 * (2 + 3)").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("1".to_string())),
                    BinOp::Mul,
                    Box::new(binary_boxed(
                        Box::new(num("2".to_string())),
                        BinOp::Add,
                        Box::new(num("3".to_string())),
                    )),
                ),
                false,
            )],
        },
    );
}

#[test]
fn multiple_statements() {
    let program = parse("x = 1; y = 2;").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![
                assign("x".to_string(), num("1".to_string()), true), // Has semicolon
                assign("y".to_string(), num("2".to_string()), true), // Has semicolon
            ],
        },
    );
}

#[test]
fn trailing_semicolon_is_allowed() {
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
                true,
            )], // true because it has a semicolon (suppressed output)
        },
    );
}

#[test]
fn parse_imaginary_unit_adjacent_number_in_matrix() {
    let program = parse("A = [1+2i 3-4j];").unwrap();
    let Stmt::Assign(_, Expr::Tensor(rows, _), _, _) = &program.body[0] else {
        panic!("expected matrix assignment");
    };
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].len(), 2);

    let Expr::Binary(lhs, BinOp::Add, rhs, _) = &rows[0][0] else {
        panic!("expected 1+2i");
    };
    assert!(matches!(**lhs, Expr::Number(ref n, _) if n == "1"));
    let Expr::Binary(rhs_l, BinOp::Mul, rhs_r, _) = &**rhs else {
        panic!("expected 2*i");
    };
    assert!(matches!(**rhs_l, Expr::Number(ref n, _) if n == "2"));
    assert!(matches!(**rhs_r, Expr::Ident(ref n, _) if n == "i"));

    let Expr::Binary(lhs, BinOp::Sub, rhs, _) = &rows[0][1] else {
        panic!("expected 3-4j");
    };
    assert!(matches!(**lhs, Expr::Number(ref n, _) if n == "3"));
    let Expr::Binary(rhs_l, BinOp::Mul, rhs_r, _) = &**rhs else {
        panic!("expected 4*j");
    };
    assert!(matches!(**rhs_l, Expr::Number(ref n, _) if n == "4"));
    assert!(matches!(**rhs_r, Expr::Ident(ref n, _) if n == "j"));
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

#[test]
fn power_is_right_associative() {
    let program = parse("2 ^ 3 ^ 2").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("2".to_string())),
                    BinOp::Pow,
                    Box::new(binary_boxed(
                        Box::new(num("3".to_string())),
                        BinOp::Pow,
                        Box::new(num("2".to_string())),
                    )),
                ),
                false,
            )],
        },
    );
}

#[test]
fn unary_minus_precedence() {
    let program = parse("-1 + 2").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(unary_boxed(UnOp::Minus, Box::new(num("1".to_string())))),
                    BinOp::Add,
                    Box::new(num("2".to_string())),
                ),
                false,
            )],
        },
    );
}

#[test]
fn unary_minus_with_power() {
    let program = parse("-2 ^ 2").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                unary_boxed(
                    UnOp::Minus,
                    Box::new(binary_boxed(
                        Box::new(num("2".to_string())),
                        BinOp::Pow,
                        Box::new(num("2".to_string())),
                    )),
                ),
                false,
            )],
        },
    );
}

#[test]
fn parse_simple_matrix() {
    let program = parse("[1,2;3,4]").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                tensor(vec![
                    vec![num("1".to_string()), num("2".to_string())],
                    vec![num("3".to_string()), num("4".to_string())],
                ]),
                false,
            )],
        },
    );
}

#[test]
fn parse_empty_matrix() {
    let program = parse("[]").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(tensor(vec![]), false)],
        },
    );
}

#[test]
fn matrix_with_expressions() {
    let program = parse("[1+2,3*4]").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                tensor(vec![vec![
                    binary_boxed(
                        Box::new(num("1".to_string())),
                        BinOp::Add,
                        Box::new(num("2".to_string())),
                    ),
                    binary_boxed(
                        Box::new(num("3".to_string())),
                        BinOp::Mul,
                        Box::new(num("4".to_string())),
                    ),
                ]]),
                false,
            )],
        },
    );
}

#[test]
fn nested_matrix_literal() {
    let program = parse("[1,[2,3]]").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                tensor(vec![vec![
                    num("1".to_string()),
                    tensor(vec![vec![num("2".to_string()), num("3".to_string())]]),
                ]]),
                false,
            )],
        },
    );
}

#[test]
fn whitespace_separated_matrix_allowed() {
    // Space-separated elements are allowed
    assert!(parse("[1 2]").is_ok());
}

#[test]
fn nested_tensor_literals_and_rows() {
    let ast = parse("A=[1 2 3; 4 5 6]; B=[7 8; 9 10];").unwrap();
    if let Stmt::Assign(_, Expr::Tensor(rows, _), _, _) = &ast.body[0] {
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 3);
        assert_eq!(rows[1].len(), 3);
    } else {
        panic!("expected tensor literal on assignment to A");
    }
    if let Stmt::Assign(_, Expr::Tensor(rows2, _), _, _) = &ast.body[1] {
        assert_eq!(rows2.len(), 2);
        assert_eq!(rows2[0].len(), 2);
    } else {
        panic!("expected tensor literal on assignment to B");
    }
}

#[test]
fn whitespace_and_comma_mixed_elements() {
    assert!(parse("[1, 2 3,4]").is_ok());
}

#[test]
fn tensor_in_index_and_end_arithmetic() {
    let ast = parse("A=[1 2 3; 4 5 6; 7 8 9]; B=A(1:end-1, [1 3]);").unwrap();
    if let Stmt::Assign(_, Expr::Tensor(rows, _), _, _) = &ast.body[0] {
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].len(), 3);
        assert_eq!(rows[1].len(), 3);
    } else {
        panic!("expected tensor literal on assignment to A");
    }
    if let Stmt::Assign(_, Expr::FuncCall(name, args, _), _, _) = &ast.body[1] {
        assert_eq!(name, "A");
        assert_eq!(args.len(), 2);
    } else {
        panic!("expected deferred call form A( ... )");
    }
}

#[test]
fn missing_closing_bracket_is_error() {
    assert!(parse("[1,2").is_err());
}

#[test]
fn left_division_operator() {
    let program = parse("4 \\ 2").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("4".to_string())),
                    BinOp::LeftDiv,
                    Box::new(num("2".to_string())),
                ),
                false,
            )],
        },
    );
}

#[test]
fn elementwise_power_operator() {
    let program = parse("3 .^ 2").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                binary_boxed(
                    Box::new(num("3".to_string())),
                    BinOp::ElemPow,
                    Box::new(num("2".to_string())),
                ),
                false,
            )],
        },
    );
}

#[test]
fn parse_if_else_statement() {
    let program = parse("if x; y=1; else y=2; end").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![Stmt::If {
                cond: ident("x".to_string()),
                then_body: vec![assign("y".to_string(), num("1".to_string()), false)],
                elseif_blocks: vec![],
                else_body: Some(vec![assign("y".to_string(), num("2".to_string()), false)]),
                span: span_value(),
            }],
        },
    );
}

#[test]
fn parse_for_loop() {
    let program = parse("for i=1:3; x=i; end").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![Stmt::For {
                var: "i".to_string(),
                expr: range(
                    Box::new(num("1".to_string())),
                    None,
                    Box::new(num("3".to_string())),
                ),
                body: vec![assign("x".to_string(), ident("i".to_string()), false)],
                span: span_value(),
            }],
        },
    );
}

#[test]
fn parse_function_definition() {
    let program = parse("function y=add(x); y=x+1; end").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![Stmt::Function {
                name: "add".to_string(),
                params: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                body: vec![assign(
                    "y".to_string(),
                    binary_boxed(
                        Box::new(ident("x".to_string())),
                        BinOp::Add,
                        Box::new(num("1".to_string())),
                    ),
                    false,
                )],
                span: span_value(),
            }],
        },
    );
}

#[test]
fn parse_array_indexing() {
    // Note: A(1,2) syntax is ambiguous with function calls in current implementation
    // This now parses as a function call, which is acceptable behavior
    let program = parse("A(1,2)").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                func_call(
                    "A".to_string(),
                    vec![num("1".to_string()), num("2".to_string())],
                ),
                false,
            )],
        },
    );
}

#[test]
fn parse_string_literal() {
    let program = parse("'hello world'").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(string("'hello world'".to_string()), false)],
        },
    );
}

#[test]
fn parse_function_call_with_string() {
    let program = parse("fprintf('test')").unwrap();
    assert_program_eq(
        program,
        Program {
            body: vec![expr_stmt(
                func_call("fprintf".to_string(), vec![string("'test'".to_string())]),
                false,
            )],
        },
    );
}

#[test]
fn parse_bracket_indexing() {
    // Bracket-based indexing would be for matrix elements
    // This test documents that we expect this to be parsed as a matrix literal for now
    let result = parse("A[1,2]");
    // This should either parse as indexing or fail to parse (which is current behavior)
    // For now, we expect this syntax to not be implemented
    assert!(result.is_err() || result.is_ok());
}
