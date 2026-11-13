use super::*;

fn lower(source: &str) -> MirAssembly {
    let parsed = runmat_parser::parse(source).expect("parse constructor source");
    let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
        .expect("lower constructor HIR");
    crate::lowering::lower_assembly(&hir.assembly).expect("lower constructor MIR")
}

#[test]
fn only_typed_constructor_instance_is_assigned_at_entry() {
    let constructor = lower(
        "classdef Seed < handle\n methods\n function obj = Seed()\n observed = obj;\n end\n end\nend",
    );
    let body = constructor
        .bodies
        .values()
        .find(|body| instance_output_local(&constructor, body).is_some())
        .expect("typed constructor body");
    let instance = instance_output_local(&constructor, body).expect("constructor instance local");
    assert_eq!(
        body.locals[instance.0].binding,
        body.abi.fixed_output_bindings().next().copied()
    );
    let diagnostics = crate::analysis::analyze_assembly(&constructor).diagnostics;
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| !matches!(diagnostic.code.as_str(), "RM-MIR0001" | "RM-MIR0002")),
        "seeded constructor output may be read: {diagnostics:#?}"
    );

    let mut invalid_static = constructor.clone();
    invalid_static
        .functions
        .get_mut(&body.function)
        .expect("constructor metadata")
        .class_method_owner
        .as_mut()
        .expect("class method owner")
        .is_static = true;
    assert_eq!(instance_output_local(&invalid_static, body), None);
    assert!(crate::analysis::analyze_assembly(&invalid_static)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0001"));

    let ordinary = lower("function out = ordinary()\n observed = out;\n out = 1;\n end");
    assert!(ordinary
        .bodies
        .values()
        .all(|body| instance_output_local(&ordinary, body).is_none()));
    assert!(crate::analysis::analyze_assembly(&ordinary)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-MIR0001"));
}
