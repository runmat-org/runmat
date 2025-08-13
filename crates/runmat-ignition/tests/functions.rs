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

