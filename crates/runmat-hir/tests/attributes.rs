use runmat_hir::{lower, FunctionKind, LoweringContext, MemberAccess};
use runmat_parser::parse;

#[test]
fn classdef_property_attributes_round_trip() {
    let src =
        "classdef A\n  properties(GetAccess=private, SetAccess=public, Static)\n    p\n  end\nend";
    let ast = parse(src).unwrap();
    let assembly = lower(&ast, &LoweringContext::empty()).unwrap().assembly;
    let class = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "A")
        .unwrap();
    let prop = class
        .properties
        .iter()
        .find(|property| property.name.0 == "p")
        .expect("property p");
    assert!(prop.attributes.is_static);
    assert_eq!(prop.attributes.get_access, MemberAccess::Private);
    assert_eq!(prop.attributes.set_access, MemberAccess::Public);
}

#[test]
fn classdef_method_attributes_round_trip() {
    let src = "classdef B\n  methods(Static, Access=private)\n    function y = foo(x); y = x; end\n  end\nend";
    let ast = parse(src).unwrap();
    let assembly = lower(&ast, &LoweringContext::empty()).unwrap().assembly;
    let class = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "B")
        .unwrap();
    let method = class
        .methods
        .iter()
        .find(|method| method.name.0 == "foo")
        .expect("method foo");
    assert!(method.is_static);
    assert_eq!(method.attributes.access, MemberAccess::Private);
    assert!(matches!(
        assembly.functions[method.function.0].kind,
        FunctionKind::ClassMethod { is_static: true }
    ));
}

#[test]
fn classdef_property_attributes_enforced() {
    let src = "classdef C\n  properties(Static, Dependent)\n    p\n  end\nend";
    let ast = parse(src).unwrap();
    let err = lower(&ast, &LoweringContext::empty())
        .expect_err("conflicting class property attributes should fail");
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassPropertyAttributeConflict")
    );
}

#[test]
fn classdef_access_values_validated() {
    let src = "classdef D\n  properties(Access=public)\n    p\n  end\n  methods(Access=internal)\n    function y = f(x); y = x; end\n  end\nend";
    let ast = parse(src).unwrap();
    let err = lower(&ast, &LoweringContext::empty())
        .expect_err("invalid class access attribute values should fail");
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassAccessValueInvalid")
    );
}

#[test]
fn classdef_protected_access_values_round_trip() {
    let src = "classdef D\n  properties(GetAccess=protected, SetAccess=protected)\n    p\n  end\n  methods(Access=protected)\n    function y = f(x); y = x; end\n  end\nend";
    let ast = parse(src).unwrap();
    let assembly = lower(&ast, &LoweringContext::empty()).unwrap().assembly;
    let class = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "D")
        .unwrap();
    let prop = class
        .properties
        .iter()
        .find(|property| property.name.0 == "p")
        .expect("property p");
    assert_eq!(prop.attributes.get_access, MemberAccess::Protected);
    assert_eq!(prop.attributes.set_access, MemberAccess::Protected);
    let method = class
        .methods
        .iter()
        .find(|method| method.name.0 == "f")
        .expect("method f");
    assert_eq!(method.attributes.access, MemberAccess::Protected);
}
