use runmat_parser::{parse_simple as parse, Expr, Stmt};

#[test]
fn import_wildcard_and_specific() {
    let program = parse("import pkg.*; import pkg.sub.Class").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::Import { path, wildcard } => {
            let expected: Vec<String> = vec!["pkg".to_string()];
            assert_eq!(path, &expected);
            assert!(*wildcard);
        }
        _ => panic!("expected import"),
    }
    match &program.body[1] {
        Stmt::Import { path, wildcard } => {
            let expected: Vec<String> = vec!["pkg".to_string(), "sub".to_string(), "Class".to_string()];
            assert_eq!(path, &expected);
            assert!(!*wildcard);
        }
        _ => panic!("expected import"),
    }
}

#[test]
fn metaclass_with_qualified_name() {
    let program = parse("?pkg.sub.Class").unwrap();
    match &program.body[0] {
        Stmt::ExprStmt(Expr::MetaClass(name), _) => {
            assert_eq!(name, "pkg.sub.Class");
        }
        _ => panic!("expected metaclass expression"),
    }
}


