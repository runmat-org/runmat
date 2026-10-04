use runmat_hir::{
    lower, HirCallableRef, HirExprKind, HirStmtKind, LoweringContext, RequestedOutputCount,
};
use runmat_types::{standard, MemberAccess, QualifiedName, StaticMethodName, SymbolName};
use std::collections::HashMap;

fn first_call(source: &str, context: &LoweringContext<'_>) -> runmat_hir::HirCall {
    let ast = runmat_parser::parse(source).unwrap();
    let assembly = lower(&ast, context).unwrap().assembly;
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let HirStmtKind::Assign(_, value, _) = &function.body.statements[0].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Call(call) = &value.kind else {
        panic!("expected static call, got {value:?}");
    };
    call.clone()
}

#[test]
fn all_integer_empty_methods_allow_omitted_parentheses() {
    for class in [
        standard::INT8,
        standard::INT16,
        standard::INT32,
        standard::INT64,
        standard::UINT8,
        standard::UINT16,
        standard::UINT32,
        standard::UINT64,
    ] {
        let name = format!("{}.empty", class.display_name());
        let bare = first_call(&format!("A = {name};"), &LoweringContext::empty());
        let explicit = first_call(&format!("A = {name}();"), &LoweringContext::empty());
        assert_eq!(bare.callee, explicit.callee);
        assert!(matches!(bare.callee,HirCallableRef::Builtin(ref id) if id.0 == name));
        assert!(bare.args.is_empty());
        assert_eq!(bare.requested_outputs, RequestedOutputCount::One);
        assert_eq!(bare.requested_outputs, explicit.requested_outputs);
    }
}

#[test]
fn string_empty_allows_omitted_parentheses() {
    let bare = first_call("S = string.empty;", &LoweringContext::empty());
    let explicit = first_call("S = string.empty();", &LoweringContext::empty());
    assert_eq!(bare.callee, explicit.callee);
    assert!(matches!(bare.callee, HirCallableRef::Builtin(ref id) if id.0 == "string.empty"));
    assert!(bare.args.is_empty());
}

#[test]
fn implicit_class_arguments_are_preserved_for_bare_methods() {
    let bare = first_call("A = uint8.zeros;", &LoweringContext::empty());
    let explicit = first_call("A = uint8.zeros();", &LoweringContext::empty());
    assert_eq!(bare.callee, explicit.callee);
    assert_eq!(bare.args.len(), 1);
    assert!(matches!(&bare.args[0].kind,HirExprKind::String(value) if value.0 == "uint8"));
}

#[test]
fn gpu_zeros_lowers_to_canonical_constructor_with_typed_class_selector() {
    for source in [
        "A = gpuArray.zeros;",
        "A = gpuArray.zeros();",
        "A = gpuArray.zeros(2, 3, 'single');",
    ] {
        let call = first_call(source, &LoweringContext::empty());
        assert!(matches!(call.callee, HirCallableRef::Builtin(ref id) if id.0 == "zeros"));
        assert!(
            matches!(&call.args.last().unwrap().kind, HirExprKind::String(value) if value.0 == "gpuArray")
        );
        if source.contains("single") {
            // Source literals retain their quotes in HIR; only the appended
            // declaration selector is synthesized without source delimiters.
            assert!(
                matches!(&call.args[call.args.len() - 2].kind, HirExprKind::String(value) if value.0 == "'single'")
            );
        }
    }
}

#[test]
fn omitted_parentheses_preserve_requested_output_arity() {
    for (bare, explicit) in [
        ("uint8.empty;", "uint8.empty();"),
        ("[a,b] = uint8.empty;", "[a,b] = uint8.empty();"),
    ] {
        let counts = [bare, explicit].map(|source| {
            let ast = runmat_parser::parse(source).unwrap();
            let result = lower(&ast,&LoweringContext::empty()).unwrap();
            result.hir_index.calls.into_iter().find(|call| matches!(&call.callee,HirCallableRef::Builtin(id) if id.0 == "uint8.empty")).unwrap().requested_outputs
        });
        assert_eq!(counts[0], counts[1]);
    }
}

#[test]
fn qualified_external_static_methods_share_access_checks() {
    let mut class = runmat_builtins::standard_class_declaration(&standard::UINT8.owned()).unwrap();
    class.name = QualifiedName(vec![SymbolName("pkg".into()), SymbolName("Factory".into())]);
    class.methods.truncate(1);
    class.methods[0].name = StaticMethodName::new("make").owned();
    for access in [MemberAccess::Public, MemberAccess::Private] {
        class.methods[0].attributes.access = access;
        let declarations = [class.clone()];
        let context = LoweringContext::empty().with_external_class_declarations(&declarations);
        for suffix in ["", "()"] {
            let ast = runmat_parser::parse(&format!("A = pkg.Factory.make{suffix};")).unwrap();
            let result = lower(&ast, &context);
            assert_eq!(result.is_ok(), access == MemberAccess::Public);
        }
    }
}

#[test]
fn bound_receivers_shadow_standard_classes_in_both_syntaxes() {
    let variables = HashMap::from([("uint8".to_owned(), 0)]);
    let context = LoweringContext::new(&variables);
    for source in ["A = uint8.empty;", "A = uint8.empty();"] {
        let ast = runmat_parser::parse(source).unwrap();
        let result = lower(&ast, &context).unwrap();
        assert!(!result.hir_index.calls.iter().any(|call|matches!(&call.callee,HirCallableRef::Builtin(id) if id.0 == "uint8.empty" || id.0 == "uint8")));
    }
}
