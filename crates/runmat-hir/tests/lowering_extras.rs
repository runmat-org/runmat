use runmat_hir::{lower, HirExprKind, HirProgram, HirStmt};
use runmat_parser::parse_simple as parse;

fn lower_src(src: &str) -> HirProgram {
    let ast = parse(src).unwrap_or_else(|e| panic!("parse: {e:?} src: {src}"));
    lower(&ast).unwrap()
}

#[test]
fn import_and_metaclass_lowering() {
    let prog = lower_src("import pkg.*; ?pkg.Class");
    assert_eq!(prog.body.len(), 2);
    // Import lowers to Import node
    match &prog.body[0] {
        HirStmt::Import { path, wildcard, .. } => {
            assert_eq!(path, &vec!["pkg".to_string()]);
            assert!(*wildcard);
        }
        _ => panic!("expected import node"),
    }
    // MetaClass lowers to a MetaClass expr carrying the qualified name
    match &prog.body[1] {
        HirStmt::ExprStmt(expr, false, _) => match &expr.kind {
            HirExprKind::MetaClass(s) => assert_eq!(s, "pkg.Class"),
            _ => panic!("expected metaclass expr"),
        },
        _ => panic!("expected expr stmt for metaclass"),
    }
}

#[test]
fn lvalue_assignment_lowering_total() {
    // Validate each lvalue assignment form lowers independently (parser hardened for adjacency)
    let p1 = lower_src("A=1;");
    assert_eq!(p1.body.len(), 1);
    let p2 = lower_src("A=1; A(1)=2;");
    assert!(p2
        .body
        .iter()
        .any(|s| matches!(s, HirStmt::AssignLValue(_, _, _, _))));
    let p3 = lower_src("A=1; A{1}=3;");
    assert!(p3
        .body
        .iter()
        .any(|s| matches!(s, HirStmt::AssignLValue(_, _, _, _))));
    let p4 = lower_src("s = struct(); s.f = 4;");
    assert!(p4
        .body
        .iter()
        .any(|s| matches!(s, HirStmt::AssignLValue(_, _, _, _))));
}

#[test]
fn import_normalization_and_ambiguity() {
    use runmat_parser::parse_simple as parse;
    let ast = parse("import pkg.*; import pkg.sub.Class; import other.Class").unwrap();
    let hir = lower(&ast).unwrap();
    let err = runmat_hir::validate_imports(&hir);
    assert!(err.is_err());
}
