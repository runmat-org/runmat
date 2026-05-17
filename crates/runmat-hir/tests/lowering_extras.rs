use runmat_hir::{
    compatibility, lower, HirAssembly, HirExprKind, HirPlace, HirStmtKind, IndexKind,
    LoweringContext,
};
use runmat_parser::parse;

fn lower_assembly(src: &str) -> HirAssembly {
    let ast = parse(src).unwrap_or_else(|e| panic!("parse: {e:?} src: {src}"));
    lower(&ast, &LoweringContext::empty()).unwrap().assembly
}

fn entry_body(assembly: &HirAssembly) -> &[runmat_hir::HirStmt] {
    let entry = assembly.entrypoints[0].target;
    &assembly.functions[entry.0].body.statements
}

#[test]
fn import_and_metaclass_lowering() {
    let assembly = lower_assembly("import pkg.*; ?pkg.Class");
    assert_eq!(assembly.modules[0].imports.len(), 1);
    assert_eq!(assembly.modules[0].imports[0].path.0[0].0, "pkg");
    assert!(assembly.modules[0].imports[0].wildcard);
    let metaclass = entry_body(&assembly)
        .iter()
        .find_map(|stmt| match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => Some(expr),
            _ => None,
        })
        .expect("metaclass expr stmt");
    match &metaclass.kind {
        HirExprKind::MetaClass(name) => {
            let text = name
                .0
                .iter()
                .map(|segment| segment.0.as_str())
                .collect::<Vec<_>>()
                .join(".");
            assert_eq!(text, "pkg.Class");
        }
        _ => panic!("expected metaclass expr"),
    }
}

#[test]
fn lvalue_assignment_lowering_total() {
    let p1 = lower_assembly("A=1;");
    assert_eq!(entry_body(&p1).len(), 1);

    let p2 = lower_assembly("A=1; A(1)=2;");
    assert!(entry_body(&p2).iter().any(|s| matches!(
        &s.kind,
        HirStmtKind::Assign(HirPlace::Index(_, indexing), _, _) if indexing.kind == IndexKind::Paren
    )));

    let p3 = lower_assembly("A=1; A{1}=3;");
    assert!(entry_body(&p3).iter().any(|s| matches!(
        &s.kind,
        HirStmtKind::Assign(HirPlace::IndexCell(_, indexing), _, _) if indexing.kind == IndexKind::Brace
    )));

    let p4 = lower_assembly("s = struct(); s.f = 4;");
    assert!(entry_body(&p4)
        .iter()
        .any(|s| matches!(&s.kind, HirStmtKind::Assign(HirPlace::Member(_, _), _, _))));
}

#[test]
fn import_normalization_and_ambiguity() {
    let ast = parse("import pkg.*; import pkg.sub.Class; import other.Class").unwrap();
    let result = compatibility::lower(&ast, &LoweringContext::empty());
    assert!(result.is_err());
}
