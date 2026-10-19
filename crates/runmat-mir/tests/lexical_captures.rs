use runmat_hir::{lower, LoweringContext};
use runmat_mir::{lowering::lower_assembly, MirAssembly, MirLocalKind};

fn lower_mir(source: &str) -> MirAssembly {
    let parsed = runmat_parser::parse(source).expect("parse lexical function source");
    let hir = lower(&parsed, &LoweringContext::empty()).expect("lower lexical HIR");
    lower_assembly(&hir.assembly).expect("lower lexical MIR")
}

fn captures<'a>(assembly: &'a MirAssembly, name: &str) -> &'a [runmat_hir::CapturedBinding] {
    assembly
        .functions
        .values()
        .find(|metadata| metadata.name.0 == name)
        .unwrap_or_else(|| panic!("missing {name}"))
        .captures
        .as_slice()
}

#[test]
fn sibling_static_call_propagates_shared_binding_through_caller() {
    let assembly = lower_mir(
        "function y = outer()\n shared = 1;\n y = writer();\n function z = reader()\n z = shared;\n end\n function z = writer()\n z = reader();\n end\n end",
    );
    let reader = captures(&assembly, "reader");
    let writer = captures(&assembly, "writer");
    assert_eq!(reader.len(), 1, "reader must capture outer binding");
    assert_eq!(
        writer, reader,
        "writer must pass the same binding to its sibling"
    );
    let writer_id = assembly
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == "writer")
        .map(|(id, _)| *id)
        .expect("writer identity");
    assert!(assembly.bodies[&writer_id].locals.iter().any(|local| {
        local.kind == MirLocalKind::Capture && local.binding == Some(writer[0].binding)
    }));
}

#[test]
fn explicit_sibling_handle_propagates_capture_demand() {
    let assembly = lower_mir(
        "function y = outer()\n shared = 1;\n y = caller();\n function z = reader()\n z = shared;\n end\n function z = caller()\n h = @reader;\n z = feval(h);\n end\n end",
    );
    assert_eq!(captures(&assembly, "caller"), captures(&assembly, "reader"));
    assert_eq!(captures(&assembly, "caller").len(), 1);
}

#[test]
fn descendant_call_propagates_binding_through_unreferencing_parent() {
    let assembly = lower_mir(
        "function y = outer()\n shared = 1;\n y = middle();\n function z = middle()\n z = descendant();\n function value = descendant()\n shared = shared + 1;\n value = shared;\n end\n end\n end",
    );
    assert_eq!(
        captures(&assembly, "middle"),
        captures(&assembly, "descendant")
    );
    assert_eq!(captures(&assembly, "middle").len(), 1);
}

#[test]
fn isolated_sibling_cannot_acquire_new_pass_through_capture() {
    let parsed = runmat_parser::parse(
        "function y = outer()\n shared = 1;\n y = caller();\n function z = reader()\n z = shared;\n end\n isolated function z = caller()\n z = reader();\n end\n end",
    )
    .expect("parse isolated lexical source");
    let hir = lower(&parsed, &LoweringContext::empty()).expect("HIR has no direct capture");
    let error = lower_assembly(&hir.assembly).expect_err("MIR must reject pass-through capture");
    assert!(
        error.to_string().contains("isolated function cannot carry"),
        "unexpected lexical isolation error: {error}"
    );
}
