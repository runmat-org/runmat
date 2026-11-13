use runmat_hir::{lower, HirExprKind, LoweringContext, MemberName};
use runmat_parser::parse;

#[test]
fn class_only_defaults_have_module_scope_without_a_callable_entrypoint() {
    let ast = parse(
        "classdef Defaults\n properties\n enabled = true;\n disabled = false;\n count = 8 \\ 2;\n end\nend",
    )
    .expect("class syntax");
    let assembly = lower(&ast, &LoweringContext::empty())
        .expect("class defaults")
        .assembly;
    assert!(assembly.functions.is_empty());
    assert!(assembly.entrypoints.is_empty());
    let class = &assembly.classes[0];
    assert!(matches!(
        &class.property_default(&MemberName("enabled".into())).unwrap().kind,
        HirExprKind::Constant(name) if name.0 == "true"
    ));
    assert!(matches!(
        &class.property_default(&MemberName("disabled".into())).unwrap().kind,
        HirExprKind::Constant(name) if name.0 == "false"
    ));
    assert!(matches!(
        &class
            .property_default(&MemberName("count".into()))
            .unwrap()
            .kind,
        HirExprKind::Binary(..)
    ));
}

#[test]
fn undeclared_class_default_variable_is_a_semantic_error() {
    let ast = parse("classdef Defaults\n properties\n value = unboundDefaultInput;\n end\nend")
        .expect("class syntax");
    assert!(lower(&ast, &LoweringContext::empty()).is_err());
}
