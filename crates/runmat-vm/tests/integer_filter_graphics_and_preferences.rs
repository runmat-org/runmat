#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_graphics_integer_extensions_have_stable_strict_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "h = fsurf(@(x,y) x+y, int16([-2 2]));",
            "RunMat:compatibility:FsurfIntegerDomainExtension",
        ),
        (
            "ax = gca(uint32(1));",
            "RunMat:compatibility:GcaIntegerFigureAliasExtension",
        ),
        (
            "value = get(uint32(1));",
            "RunMat:compatibility:GetIntegerHandleAliasExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_integer_inapplicable_text_and_method_roles_reject() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "name = genvarname(uint8(1));",
        "method = getmethod(uint8(1), 'name');",
        "method = getmethod(classref('Example'), uint8(1));",
    ] {
        assert!(execute_source(source).is_err(), "{source}");
    }
}

#[test]
fn compiled_gobjects_accepts_typed_integer_dimensions() {
    let values = execute_source("h = gobjects(uint16(2), int8(3));")
        .expect("compiled typed-integer gobjects dimensions");
    assert!(values.iter().any(|value| {
        matches!(value, runmat_builtins::Value::Tensor(tensor) if tensor.shape == vec![2, 3])
    }));
}
