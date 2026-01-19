use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn command_form_with_escaped_double_quotes_and_end() {
    // Double-quoted string with doubled "" escapes and an end token as arg
    let program = parse("echo \"he said \"\"hi\"\"\" end").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "echo");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(ref s, _) if s.contains("he said \"\"hi\"\"")));
            // 'end' allowed as literal identifier in command-form args
            assert!(
                matches!(args[1], Expr::Ident(ref s, _) if s == "end")
                    || matches!(args[1], Expr::EndKeyword(_))
            );
        }
        _ => panic!("expected command form call"),
    }
}

#[test]
fn command_form_with_ellipsis_and_end_then_ident() {
    // Ellipsis continues the command-form list; end and identifier come on next line
    let program = parse("foo 'a' ...\n end bar").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), _, _) => {
            assert_eq!(name, "foo");
            assert_eq!(args.len(), 3);
            assert!(matches!(args[0], Expr::String(_, _)));
            assert!(
                matches!(args[1], Expr::Ident(ref s, _) if s == "end")
                    || matches!(args[1], Expr::EndKeyword(_))
            );
            assert!(matches!(args[2], Expr::Ident(ref s, _) if s == "bar"));
        }
        _ => panic!("expected command form call"),
    }
}

#[test]
fn imports_and_nested_metaclass_on_one_line() {
    let program = parse("import pkg.sub.*; import other.deep.Class; x = ?pkg.sub.Class.static; y = ?pkg.sub.Class.method(7)").unwrap();
    assert_eq!(program.body.len(), 4);
    match &program.body[2] {
        Stmt::Assign(name, rhs, _, _) => {
            assert_eq!(name, "x");
            match rhs {
                Expr::Member(obj, field, _) => {
                    assert_eq!(field, "static");
                    assert!(matches!(**obj, Expr::MetaClass(ref s, _) if s == "pkg.sub.Class"));
                }
                _ => panic!("expected member access on metaclass"),
            }
        }
        _ => panic!("expected first assignment"),
    }
    match &program.body[3] {
        Stmt::Assign(name, rhs, _, _) => {
            assert_eq!(name, "y");
            match rhs {
                Expr::MethodCall(obj, m, args, _) => {
                    assert_eq!(m, "method");
                    assert!(matches!(**obj, Expr::MetaClass(ref s, _) if s == "pkg.sub.Class"));
                    assert_eq!(args.len(), 1);
                }
                _ => panic!("expected method call on metaclass"),
            }
        }
        _ => panic!("expected second assignment"),
    }
}

#[test]
fn stress_dynamic_member_with_surrounding_tokens_errors() {
    // This should not be accepted; dynamic member in middle of token stream
    let res1 = parse("foo 'a' s.(1)");
    assert!(res1.is_err());
    let res2 = parse("before s.(x) after");
    assert!(res2.is_err());
    let res3 = parse("cmd ...\n s.(x)");
    assert!(res3.is_err());
}
