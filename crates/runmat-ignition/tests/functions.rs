use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;

#[test]
fn function_definition_and_calls() {
    let ast = parse("function y = f(x); y = x + 1; end; a = f(2);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
}

#[test]
fn nested_function_calls() {
    let ast = parse("function y = add(a, b); y = a + b; end; function y = multiply_and_add(x); y = add(x * 2, x * 3); end; result = multiply_and_add(4);").unwrap();
    let hir = lower(&ast).unwrap();
    let result = execute(&hir);
    assert!(result.is_ok());
}

#[test]
fn member_get_set_and_method_call_skeleton() {
    let input = "obj = new_object('Point'); obj = setfield(obj, 'x', 3); ax = getfield(obj, 'x');";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < f64::EPSILON)));

    // call Point.move which exists as example method: dx=1,dy=2
    let input2 = "obj = new_object('Point'); obj = setfield(obj,'x',5); obj = setfield(obj,'y',7); obj = call_method(obj, 'move', 1, 2); rx = getfield(obj,'x'); ry = getfield(obj,'y');";
    let ast2 = parse(input2).unwrap();
    let hir2 = lower(&ast2).unwrap();
    let vars2 = runmat_ignition::execute(&hir2).unwrap();
    assert!(vars2.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < f64::EPSILON)));
    assert!(vars2.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < f64::EPSILON)));
}

#[test]
fn function_handle_anon_round_trip() {
    let input = "h = make_handle('sin'); g = make_anon('x', 'x+1');";
    let ast = parse(input).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::String(s) if s.starts_with("@"))));
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::String(s) if s.starts_with("@anon"))));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_static_and_inheritance() {
    // Register classes
    let ast = parse("__register_test_classes();").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_ok());

    // Default init and namespaced
    let ast2 = parse("__register_test_classes(); p = new_object('Point'); ax = getfield(p,'x'); ns = new_object('pkg.PointNS'); nsx = getfield(ns,'x');").unwrap();
    let hir2 = lower(&ast2).unwrap();
    let vars2 = execute(&hir2).unwrap();
    assert!(vars2.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 0.0).abs()<f64::EPSILON)));
    assert!(vars2.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<f64::EPSILON)));

    // Static method and property
    let ast3 = parse("__register_test_classes(); o = classref('Point').origin(); sv = classref('Point').staticValue;").unwrap();
    let hir3 = lower(&ast3).unwrap();
    let vars3 = execute(&hir3).unwrap();
    assert!(vars3.iter().any(|v| matches!(v, runmat_builtins::Value::Object(_))));
    assert!(vars3.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs()<f64::EPSILON)));

    // Inheritance override: use feval(getmethod(...))()
    let ast4 = parse("__register_test_classes(); c = new_object('Circle'); c = setfield(c,'r', 2); a = feval(getmethod(c,'area'));").unwrap();
    let hir4 = lower(&ast4).unwrap();
    let vars4 = execute(&hir4).unwrap();
    assert!(vars4.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - std::f64::consts::PI*4.0).abs() < 1e-9)));

    // Access control violations
    let ast5 = parse("__register_test_classes(); p = new_object('Point'); s = getfield(p,'secret');").unwrap();
    let hir5 = lower(&ast5).unwrap();
    assert!(runmat_ignition::execute(&hir5).is_err());
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_constructor_and_overloaded_indexing() {
    // Register classes (includes Ctor and OverIdx definitions)
    let ast = parse("__register_test_classes();").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_ok());

    // Call Ctor constructor; exercise OverIdx subsref/subsasgn
    let program = "__register_test_classes(); c = Ctor(7); o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5); t = call_method(o,'subsref','.', 'k');";
    let hir2 = lower(&parse(program).unwrap()).unwrap();
    let vars = execute(&hir2).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Object(_))));
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_property_access_attributes() {
    // Register classes
    let _ = execute(&lower(&parse("__register_test_classes();").unwrap()).unwrap());
    // Private get already covered by existing test; ensure private set is also rejected
    let ast = parse("__register_test_classes(); p = new_object('Point'); p = setfield(p,'secret', 7);").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(runmat_ignition::execute(&hir).is_err());
}

#[test]
fn import_builtin_resolution_for_static_method() {
    // Register classes and import Point.* so we can call origin() unqualified
    let program = "__register_test_classes(); import Point.*; o = origin();";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Object(_))));
}

#[test]
fn classdef_with_attributes_enforced() {
    // Define class A with private get and public set on property p, then enforce via getfield/setfield
    let src = "classdef A\n  properties(GetAccess=private, SetAccess=public)\n    p\n  end\nend\n a = new_object('A'); a = setfield(a,'p',5); try; v = getfield(a,'p'); catch e; ok=1; end";
    let ast = runmat_parser::parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn builtin_call_with_expanded_middle_argument() {
    // Use deal to produce a cell row and index into it to pass as middle arg
    // max(a,b) with b coming from C{1}
    let program = "C = deal(10, 20); r = max(5, C{1});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // Expect 10 as result appears in vars somewhere
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn builtin_call_with_two_expanded_args() {
    let program = "C = deal(3, 4); D = deal(5, 6); r = max(C{1}, D{1});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn user_function_with_two_expanded_args() {
    let program = "function y = sum2(a,b); y = a + b; end; C = deal(7,8); D = deal(11,12); r = sum2(C{2}, D{1});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 19.0).abs() < 1e-9)));
}

#[test]
fn expansion_on_non_cell_errors() {
    let program = "r = max(5, 10{1});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    assert!(runmat_ignition::execute(&hir).is_err());
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn object_cell_expansion_via_subsref() {
    let program = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn','{}', {1}, 42); r = max(o{1}, 5);";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs() < 1e-9)));
}

#[test]
fn expand_all_elements_in_args() {
    // C{:} expands all elements of C into separate arguments
    // max takes two args; here C has more; we only assert no crash and presence of some expected nums
    let program = "C = deal(1,2); a = max(C{:});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(_))));
}

#[test]
fn builtin_vector_index_expansion() {
    let program = "C = deal(9, 2); r = max(C{[1 2]});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn user_function_vector_index_expansion() {
    let program = "function y = sum2(a,b); y = a + b; end; C = deal(3,4); r = sum2(C{[1 2]});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn function_call_base_expand_all() {
    let program = "function y = sum2(a,b); y = a + b; end; r = sum2(deal(5,6){:});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 11.0).abs() < 1e-9)));
}

