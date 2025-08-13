use runmat_parser::parse;
use runmat_hir::{lower, HirStmt, HirClassMember};

#[test]
fn classdef_property_attributes_round_trip() {
    let src = "classdef A\n  properties(GetAccess=private, SetAccess=public, Static)\n    p\n  end\nend";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let mut found = false;
    for stmt in hir.body {
        if let HirStmt::ClassDef { members, .. } = stmt {
            for m in members {
                if let HirClassMember::Properties { attributes, names } = m {
                    assert!(names.contains(&"p".to_string()));
                    // Attributes should include GetAccess and SetAccess
                    let mut has_get = false;
                    let mut has_set = false;
                    let mut has_static = false;
                    for a in attributes {
                        if a.name.eq_ignore_ascii_case("GetAccess") { has_get = true; }
                        if a.name.eq_ignore_ascii_case("SetAccess") { has_set = true; }
                        if a.name.eq_ignore_ascii_case("Static") { has_static = true; }
                    }
                    assert!(has_get && has_set && has_static);
                    found = true;
                }
            }
        }
    }
    assert!(found, "No classdef properties block found");
}

#[test]
fn classdef_method_attributes_round_trip() {
    let src = "classdef B\n  methods(Static, Access=private)\n    function y = foo(x); y = x; end\n  end\nend";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let mut found = false;
    for stmt in hir.body {
        if let HirStmt::ClassDef { members, .. } = stmt {
            for m in members {
                if let HirClassMember::Methods { attributes, body } = m {
                    // Have at least one function
                    assert!(body.iter().any(|s| matches!(s, HirStmt::Function { name, .. } if name == "foo")));
                    // Attributes include Static and Access
                    let mut has_static = false;
                    let mut has_access = false;
                    for a in attributes {
                        if a.name.eq_ignore_ascii_case("Static") { has_static = true; }
                        if a.name.eq_ignore_ascii_case("Access") { has_access = true; }
                    }
                    assert!(has_static && has_access);
                    found = true;
                }
            }
        }
    }
    assert!(found, "No classdef methods block found");
}


