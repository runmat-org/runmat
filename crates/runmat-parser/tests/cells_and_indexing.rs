use runmat_parser::{BinOp, Expr, Stmt};

mod parse;
use parse::parse;

#[test]
fn cell_literal_and_indexing() {
    let program = parse("C = {1,2;3,4}; C{1,2}").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::Assign(name, Expr::Cell(rows, _), true, _) => {
            assert_eq!(name, "C");
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
        }
        _ => panic!("expected cell assignment"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::IndexCell(base, idxs, _), _, _) => {
            assert!(matches!(**base, Expr::Ident(ref n, _) if n == "C"));
            assert_eq!(idxs.len(), 2);
        }
        _ => panic!("expected cell indexing"),
    }
}

#[test]
fn indexing_with_end_and_member_method() {
    let program = parse("A(5:end); obj.method(a,b)").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::ExprStmt(Expr::FuncCall(name, args, _), true, _) => {
            assert_eq!(name, "A");
            assert_eq!(args.len(), 1);
            assert!(matches!(
                args.as_slice(),
                [Expr::Binary(_, BinOp::Colon, _, _)] | [Expr::Range(_, _, _, _)]
            ));
        }
        _ => panic!("expected deferred call form for A(5:end)"),
    }
    match &program.body[1] {
        Stmt::ExprStmt(Expr::DottedInvoke(obj, name, args, _), false, _) => {
            assert!(matches!(**obj, Expr::Ident(ref n, _) if n == "obj"));
            assert_eq!(name, "method");
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected dotted invoke expression"),
    }
}

#[test]
fn function_call_accepts_name_value_arguments() {
    let program =
        parse(r#"viewer = siteviewer(Buildings="map.osm", Basemap="topographic");"#).unwrap();
    assert_eq!(program.body.len(), 1);

    let Stmt::Assign(name, Expr::FuncCall(function_name, args, _), true, _) = &program.body[0]
    else {
        panic!("expected assignment from function call");
    };
    assert_eq!(name, "viewer");
    assert_eq!(function_name, "siteviewer");
    assert_eq!(args.len(), 2);

    let Expr::NameValueArg(first_name, first_value, _) = &args[0] else {
        panic!("expected first name-value argument");
    };
    assert_eq!(first_name, "Buildings");
    assert!(matches!(**first_value, Expr::String(ref value, _) if value == "\"map.osm\""));

    let Expr::NameValueArg(second_name, second_value, _) = &args[1] else {
        panic!("expected second name-value argument");
    };
    assert_eq!(second_name, "Basemap");
    assert!(matches!(**second_value, Expr::String(ref value, _) if value == "\"topographic\""));
}

#[test]
fn method_call_accepts_positional_and_name_value_arguments() {
    let program = parse(r#"obj.update(1, Mode="fast", Count=2)"#).unwrap();
    assert_eq!(program.body.len(), 1);

    let Stmt::ExprStmt(Expr::DottedInvoke(base, method, args, _), false, _) = &program.body[0]
    else {
        panic!("expected dotted invoke expression");
    };
    assert!(matches!(**base, Expr::Ident(ref name, _) if name == "obj"));
    assert_eq!(method, "update");
    assert_eq!(args.len(), 3);
    assert!(matches!(args[0], Expr::Number(ref value, _) if value == "1"));

    let Expr::NameValueArg(first_name, first_value, _) = &args[1] else {
        panic!("expected first name-value method argument");
    };
    assert_eq!(first_name, "Mode");
    assert!(matches!(**first_value, Expr::String(ref value, _) if value == "\"fast\""));

    let Expr::NameValueArg(second_name, second_value, _) = &args[2] else {
        panic!("expected second name-value method argument");
    };
    assert_eq!(second_name, "Count");
    assert!(matches!(**second_value, Expr::Number(ref value, _) if value == "2"));
}

#[test]
fn dynamic_member_expression_and_indexing_parse() {
    let program = parse("f = 'x'; y = s.(f); z = s.(f){2};").unwrap();
    assert_eq!(program.body.len(), 3);
    match &program.body[1] {
        Stmt::Assign(name, Expr::MemberDynamic(base, dyn_name, _), true, _) => {
            assert_eq!(name, "y");
            assert!(matches!(**base, Expr::Ident(ref n, _) if n == "s"));
            assert!(matches!(**dyn_name, Expr::Ident(ref n, _) if n == "f"));
        }
        other => panic!("expected dynamic member expression assignment, got {other:?}"),
    }
    match &program.body[2] {
        Stmt::Assign(name, Expr::IndexCell(base, idxs, _), true, _) => {
            assert_eq!(name, "z");
            assert_eq!(idxs.len(), 1);
            assert!(matches!(idxs[0], Expr::Number(ref n, _) if n == "2"));
            assert!(matches!(&**base, Expr::MemberDynamic(_, _, _)));
        }
        other => panic!("expected indexed dynamic member expression assignment, got {other:?}"),
    }
}
