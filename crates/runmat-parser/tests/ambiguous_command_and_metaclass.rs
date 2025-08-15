use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn command_form_with_single_quoted_and_number_args() {
    let program = parse("echo 'hello world' 42").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args), _) => {
            assert_eq!(name, "echo");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[0], Expr::String(_)));
            assert!(matches!(args[1], Expr::Number(_)));
        }
        _ => panic!("expected command-form func call"),
    }
}

#[test]
fn command_form_not_triggered_when_arg_would_be_indexing() {
    // Our parser treats this as command-form followed by simple tokens until '('; thus, this is invalid
    // because '(' starts expression syntax without separator. Expect a parse error.
    let res = parse("foo b(1)");
    assert!(res.is_err());
}

#[test]
fn metaclass_basic_and_qualified() {
    // Simple class
    let program = parse("?ClassName").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::MetaClass(s), _) => assert_eq!(s, "ClassName"),
        _ => panic!("expected metaclass expr"),
    }

    // Qualified package path
    let program = parse("?pkg.sub.Class").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::MetaClass(s), _) => assert_eq!(s, "pkg.sub.Class"),
        _ => panic!("expected metaclass expr"),
    }
}

#[test]
fn metaclass_in_assignment_and_postfix_member() {
    // Now supports postfix after metaclass
    let program = parse("?pkg.Class.size").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::Member(obj, member), _) => {
            assert_eq!(member, "size");
            assert!(matches!(**obj, Expr::MetaClass(ref s) if s == "pkg.Class"));
        }
        _ => panic!("expected member access on metaclass"),
    }
}


