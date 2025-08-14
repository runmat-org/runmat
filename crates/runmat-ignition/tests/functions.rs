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
fn import_specific_resolution_for_builtin() {
    // Use specific import to bring a qualified builtin into scope
    let program = "import pkg.missing.*; import Point.origin; __register_test_classes(); o = origin();";
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
fn end_minus_one_1d_slice_collect() {
    let program = "A = [1 2 3]; B = A(1:1:end-1); s = sum(B);";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn end_minus_one_1d_slice_assign_broadcast() {
    let program = "A = [1 2 3 4]; A(2:1:end-1) = 9; r = sum(A);";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // A becomes [1 9 9 4] => sum 23
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 23.0).abs() < 1e-9)));
}

#[test]
fn multidim_range_end_assign() {
    // Assign on second dim using range with end-1
    let program = "A = [1 2 3; 4 5 6]; A(:,2:2:end-1) = 9; s = sum(A);";
    // Original A sum is 21. We set column 2 to 9 across all rows: [1 9 3; 4 9 6] => sum 32
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 32.0).abs() < 1e-9)));
}

#[test]
fn multidim_range_end_assign_non_scalar_rhs_broadcast() {
    // 2x3; assign the middle column selection with a 2x1 rhs, which should broadcast along the selection length
    let program = "A = [1 2 3; 4 5 6]; B = [7;8]; A(:,2:2:end-1) = B; s = sum(A);";
    // Becomes [1 7 3; 4 8 6] => sum 29
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 29.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_vector_broadcast() {
    // 3x4 matrix; select rows 2:end (rows {2,3}) and cols 1:2:end-1 (cols {1,3}); assign 2x1 vector broadcast across selected cols
    let program = "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [100;200]; A(2:end, 1:2:end-1) = B; s = sum(A);";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // Expected sum 646 (see analysis)
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 646.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_matrix_rhs_exact_shape() {
    // Assign 2x2 block with exact-shaped RHS
    let program = "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [1 3; 2 4]; A(2:end, 1:2:end-1) = B; s = sum(A);";
    // Note: [1 3; 2 4] in MATLAB column-major maps to data [1,2,3,4] in our Tensor internal
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // New values: positions (2,1)=1,(3,1)=2,(2,3)=3,(3,3)=4. Change from original 5,9,7,11 -> delta = (1-5)+(2-9)+(3-7)+(4-11) = -22; sum 78-22=56
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 56.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_shape_mismatch_error() {
    // RHS shape 3x1 does not match rows 2:end (len 2) and cannot broadcast
    let program = "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [1;2;3]; A(2:end, 1:2:end-1) = B;";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let res = runmat_ignition::execute(&hir);
    assert!(res.is_err());
}

#[test]
fn broadcasting_roundtrip_property_like() {
    // After assignment with broadcasted column vector, selected columns equal the vector
    let program = "A = zeros(3,4); v = [7;8;9]; A(:, 1:2:end-1) = v; x = A(:,1); y = A(:,3);";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // Expect 7,8,9 present for x and y
    let mut count = 0;
    for v in vars { if let runmat_builtins::Value::Tensor(t) = v { if t.shape == vec![3,1] && (t.data[0]-7.0).abs()<1e-9 && (t.data[1]-8.0).abs()<1e-9 && (t.data[2]-9.0).abs()<1e-9 { count+=1; } } }
    assert!(count >= 1);
}
#[test]
fn builtin_call_with_function_return_propagation() {
    // g returns two numbers; propagate as args to max
    let program = "function [a,b] = g(); a=9; b=4; end; r = max(g());";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn function_call_base_expand_all() {
    let program = "function y = sum2(a,b); y = a + b; end; r = sum2(deal(5,6){:});";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 11.0).abs() < 1e-9)));
}

#[test]
fn function_return_propagation_in_args() {
    // g returns [a,b]; f takes two inputs; call f(g()) and ensure both outputs flow into f
    let program = "function [a,b] = g(); a=2; b=3; end; function y = f(x1,x2); y = x1 + x2; end; r = f(g());";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn varargin_pack_and_forward() {
    // f sums all inputs using varargin; g forwards varargin into f via feval(@f, varargin{:})
    let program = r#"
        function y = f(varargin)
            y = sum(cat(2, varargin{:}));
        end
        function out = g(varargin)
            out = feval(@f, varargin{:});
        end
        r1 = f(1,2,3);
        r2 = g(4,5,6,7);
    "#;
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-6.0).abs()<1e-9)));
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-22.0).abs()<1e-9)));
}

#[test]
fn varargout_expand_into_outer_call() {
    // h returns varargout with three numbers; max(h()) should consume two (max takes two args)
    let program = r#"
        function varargout = h(x)
            varargout = {x+1, x+2, x+3};
        end
        r = max(h(10));
    "#;
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // max(11,12) = 12
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-12.0).abs()<1e-9)));
}

#[test]
fn user_function_consumes_varargout_exact_needed() {
    // f takes three args; g returns varargout [1,2,3]; call f(g())
    let program = r#"
        function y = f(a,b,c)
            y = a + 10*b + 100*c;
        end
        function varargout = g()
            varargout = {1,2,3};
        end
        r = f(g());
    "#;
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // 1 + 20 + 300 = 321
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-321.0).abs()<1e-9)));
}

#[test]
fn operator_overloading_plus_times_lt_eq() {
    let setup = "__register_test_classes();";
    // Try plus with mixed object/scalar; OverIdx has no plus, so fallback should not crash
    let program = format!("{} o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5); r1 = o + 3;", setup);
    let hir = lower(&runmat_parser::parse(&program).unwrap()).unwrap();
    let _ = execute(&hir).unwrap();
}

#[test]
fn operator_overloading_elementwise_vs_mtimes_and_mixed() {
    let setup = "__register_test_classes();";
    // Exercise times and mtimes overload paths (will fallback if not implemented)
    let program = format!("{} o = new_object('OverIdx'); a = o .* 2; b = o * 2; c = 2 .* o; d = 2 * o;", setup);
    let hir = lower(&runmat_parser::parse(&program).unwrap()).unwrap();
    let _ = execute(&hir).unwrap();
}

#[test]
fn operator_overloading_relational_lt_eq() {
    let setup = "__register_test_classes();";
    // lt and eq with mixed object/scalar on both sides
    let program = format!("{} o = new_object('OverIdx'); t1 = (o < 10); t2 = (10 < o); t3 = (o == 0); t4 = (0 == o);", setup);
    let hir = lower(&runmat_parser::parse(&program).unwrap()).unwrap();
    let _ = execute(&hir).unwrap();
}

#[test]
fn operator_overloading_numeric_results_and_bitwise_arrays() {
    // Verify explicit numeric outcomes for OverIdx overloads
    let program = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5); r1 = o + 3; r2 = o .* 2; r3 = o * 4; a = (o < 10); b = (o == 5);";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 8.0).abs() < 1e-9))); // r1 = 5+3
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9))); // r2 = 5*2
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 20.0).abs() < 1e-9))); // r3 = 5*4
    // a and b push logicals (1.0/0.0) somewhere in vars
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9))); // 5<10 true
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9))); // 5==5 true

    // Bitwise and/or on arrays: verify element-wise behavior
    let program2 = "A = [1 0 1; 0 1 0]; B = [1 1 0; 0 0 1]; C = A & B; D = A | B; Sc = sum(C(:)); Sd = sum(D(:));";
    let hir2 = lower(&runmat_parser::parse(program2).unwrap()).unwrap();
    let vars2 = execute(&hir2).unwrap();
    assert!(vars2.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    assert!(vars2.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-5.0).abs()<1e-9)));
}

#[test]
fn function_return_propagation_partial_fill() {
    // g returns [1,2,3]; h takes 2 inputs; ensure leftmost two feed h
    let program = "function [a,b,c] = g(); a=1; b=2; c=3; end; function y = h(x1,x2); y = x1*10 + x2; end; r = h(g());";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    // 1*10 + 2 = 12
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 12.0).abs() < 1e-9)));
}

#[test]
fn nested_function_return_propagation_mixed_with_fixed() {
    // g()->[4,5]; f(x,p,z)=x+p+z; call f(1, g()) => 1+4+5=10
    let program = "function [a,b] = g(); a=4; b=5; end; function y = f(x,p,z); y = x + p + z; end; r = f(1, g());";
    let hir = lower(&runmat_parser::parse(program).unwrap()).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

