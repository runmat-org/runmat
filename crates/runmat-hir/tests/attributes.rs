use runmat_hir::{lower, EventAccess, FunctionKind, LoweringContext, MemberAccess};
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
        .find(|class| class.declaration.name.0[0].0 == "A")
        .unwrap();
    let prop = class
        .declaration
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
        .find(|class| class.declaration.name.0[0].0 == "B")
        .unwrap();
    let method = class
        .declaration
        .methods
        .iter()
        .find(|method| method.name.0 == "foo")
        .expect("method foo");
    assert!(method.is_static);
    assert_eq!(method.attributes.access, MemberAccess::Private);
    assert!(matches!(
        assembly.functions[method.function.0].kind,
        FunctionKind::ClassMethod {
            is_static: true,
            role: runmat_types::ClassMethodRole::Ordinary
        }
    ));
}

#[test]
fn constructor_role_is_classified_once_from_its_declaration() {
    let src = "classdef Owner\n  methods\n    function obj = Owner(); end\n    function out = other(obj); out = obj; end\n  end\nend";
    let ast = parse(src).expect("parse constructor owner");
    let assembly = lower(&ast, &LoweringContext::empty())
        .expect("lower constructor owner")
        .assembly;
    let class = &assembly.classes[0];
    for method in &class.declaration.methods {
        let expected = if method.name.display_name() == "Owner" {
            runmat_types::ClassMethodRole::Constructor
        } else {
            runmat_types::ClassMethodRole::Ordinary
        };
        assert!(matches!(
            &assembly.functions[method.function.0].kind,
            FunctionKind::ClassMethod {
                is_static: false,
                role,
            } if *role == expected
        ));
    }
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
        .find(|class| class.declaration.name.0[0].0 == "D")
        .unwrap();
    let prop = class
        .declaration
        .properties
        .iter()
        .find(|property| property.name.0 == "p")
        .expect("property p");
    assert_eq!(prop.attributes.get_access, MemberAccess::Protected);
    assert_eq!(prop.attributes.set_access, MemberAccess::Protected);
    let method = class
        .declaration
        .methods
        .iter()
        .find(|method| method.name.0 == "f")
        .expect("method f");
    assert_eq!(method.attributes.access, MemberAccess::Protected);
}

#[test]
fn classdef_event_attributes_are_typed_and_handle_only() {
    let src = "classdef E < handle\n  events(ListenAccess=protected, NotifyAccess=private, Hidden)\n    Changed\n  end\nend";
    let ast = parse(src).unwrap();
    let assembly = lower(&ast, &LoweringContext::empty()).unwrap().assembly;
    let event = &assembly.classes[0].declaration.events[0];
    assert_eq!(event.name.0, "Changed");
    assert!(event.attributes.is_hidden);
    assert_eq!(event.attributes.listen_access, EventAccess::Protected);
    assert_eq!(event.attributes.notify_access, EventAccess::Private);

    let value_class = parse("classdef V\n  events\n    Changed\n  end\nend").unwrap();
    let error = lower(&value_class, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:ClassEventAttributeInvalid")
    );
}

#[test]
fn classdef_event_attributes_reject_unknown_and_duplicate_forms() {
    for source in [
        "classdef E < handle\n events(Unsupported=true)\n Changed\n end\nend",
        "classdef E < handle\n events(ListenAccess=public, ListenAccess=private)\n Changed\n end\nend",
        "classdef E < handle\n events(Hidden=unknown)\n Changed\n end\nend",
    ] {
        let ast = parse(source).unwrap();
        let error = lower(&ast, &LoweringContext::empty()).unwrap_err();
        assert_eq!(error.identifier.as_deref(), Some("RunMat:ClassEventAttributeInvalid"));
    }
}

#[test]
fn standard_event_data_subclass_is_a_handle_class() {
    let ast =
        parse("classdef Payload < event.EventData\n properties\n PayloadValue\n end\nend").unwrap();
    let assembly = lower(&ast, &LoweringContext::empty()).unwrap().assembly;
    assert_eq!(
        assembly.classes[0].declaration.kind,
        runmat_hir::ClassKind::Handle
    );
    assert_eq!(
        assembly.classes[0]
            .declaration
            .inheritance
            .builtin_super_class
            .as_deref(),
        Some("event.EventData")
    );
}

#[test]
fn event_cannot_reuse_its_class_name() {
    let ast = parse("classdef Same < handle\n events\n Same\n end\nend").unwrap();
    let error = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:ClassMemberNameConflict")
    );
}

#[test]
fn event_access_lists_preserve_typed_class_identities() {
    let ast = parse(
        "classdef E < handle\n events(ListenAccess={?ClientA, ?pkg.ClientB}, NotifyAccess={})\n Changed\n end\nend",
    )
    .unwrap();
    let assembly = lower(&ast, &LoweringContext::empty()).unwrap().assembly;
    let event = &assembly.classes[0].declaration.events[0];
    assert_eq!(
        event.attributes.listen_access,
        EventAccess::Classes(vec!["ClientA".into(), "pkg.ClientB".into()])
    );
    assert_eq!(event.attributes.notify_access, EventAccess::Private);
}
