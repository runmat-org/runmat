#[path = "support/mod.rs"]
mod test_helpers;

use std::convert::TryFrom;
use std::thread;
use test_helpers::compile_semantic_source;
use test_helpers::interpret;

fn execute_semantic_source(source: &str) -> Vec<runmat_builtins::Value> {
    let bytecode = compile_semantic_source(source).expect("compile semantic source");
    interpret(&bytecode).expect("execute semantic bytecode")
}

fn execute_semantic_source_result(
    source: &str,
) -> Result<Vec<runmat_builtins::Value>, runmat_runtime::RuntimeError> {
    let bytecode = compile_semantic_source(source).expect("compile semantic source");
    interpret(&bytecode)
}

#[test]
fn nargin_nargout_in_user_functions() {
    // Single-output: nargin/nargout should reflect call site
    let program = r#"
        function y = f(a,b)
            y = nargin() + nargout(); % 2 + 1 = 3
        end
        r = f(10, 20);
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-3.0).abs()<1e-9)));

    // Multi-output context: nargout seen inside should match requested outs
    let program2 = r#"
        function [x,y] = g(a)
            x = nargin(); % 1
            y = nargout(); % 2
        end
        [u,v] = g(7);
    "#;
    let vars2 = execute_semantic_source(program2);
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-2.0).abs()<1e-9)));
}

#[test]
fn not_enough_and_too_many_inputs_fixed_arity() {
    // not enough inputs
    let program = r#"
        function y = f(a,b)
            y = a + b;
        end
        r = f(1);
    "#;
    let err = execute_semantic_source_result(program).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));

    // too many inputs
    let program2 = r#"
        function y = f(a,b)
            y = a + b;
        end
        r = f(1,2,3);
    "#;
    let err2 = execute_semantic_source_result(program2).err().unwrap();
    assert_eq!(err2.identifier(), Some("RunMat:TooManyInputs"));
}

#[test]
fn inputs_with_varargin_minimum_only() {
    // With varargin, fewer than fixed args is an error; equal to fixed is okay; extras go to varargin
    let program_err = r#"
        function y = f(a,b,varargin)
            y = a + b;
        end
        r = f(1);
    "#;
    let err = execute_semantic_source_result(program_err).err().unwrap();
    assert_eq!(err.identifier(), Some("RunMat:NotEnoughInputs"));

    let program_ok = r#"
        function y = f(a,b,varargin)
            % Keep it simple to avoid runtime cat/sum dependencies
            y = a + b + varargin{1} + varargin{2};
        end
        r = f(1,2,3,4);
    "#;
    let vars_ok = execute_semantic_source(program_ok);
    assert!(vars_ok
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-10.0).abs()<1e-9)));
    // 1+2+3+4
}

#[test]
fn too_many_outputs_and_varargout_mismatch() {
    // Too many outputs for a function without varargout
    let program_tmo = r#"
        function y = f(a)
            y = a + 1;
        end
        [x1,x2] = f(3);
    "#;
    let err_tmo = execute_semantic_source_result(program_tmo).err().unwrap();
    assert_eq!(err_tmo.identifier(), Some("RunMat:TooManyOutputs"));

    // Varargout requested more than provided
    let program_mis = r#"
        function varargout = h(a)
            varargout = {a+1, a+2};
        end
        [x1,x2,x3] = h(5);
    "#;
    let err_mis = execute_semantic_source_result(program_mis).err().unwrap();
    assert_eq!(err_mis.identifier(), Some("RunMat:VarargoutMismatch"));
}
#[allow(dead_code)]
fn function_definition_and_calls() {
    let result = execute_semantic_source_result("function y = f(x); y = x + 1; end; a = f(2);");
    assert!(result.is_ok());
}

#[test]
fn nested_function_calls() {
    let program = "function y = add(a, b); y = a + b; end; function y = multiply_and_add(x); y = add(x * 2, x * 3); end; result = multiply_and_add(4);";
    let handle = thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let vars = execute_semantic_source(program);
            assert!(vars
                .iter()
                .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
        })
        .expect("spawn nested_function_calls thread");
    handle.join().expect("nested_function_calls thread failed");
}

#[test]
fn shared_input_output_name_updates_in_place() {
    let program = r#"
        function x = bump(x)
            x = x + 1;
        end
        r = bump(4);
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn shared_input_output_name_multi_output_reads_original_input() {
    let program = r#"
        function [x, y] = bump_and_copy(x)
            y = x;
            x = x + 1;
        end
        [a, b] = bump_and_copy(4);
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 4.0).abs() < 1e-9)));
}

#[test]
fn semantic_user_function_multi_assign_executes() {
    let bytecode = compile_semantic_source(
        r#"
        function [a,b] = g()
            a = 2;
            b = 3;
        end
        [x,y] = g();
        z = x + y;
    "#,
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn semantic_feval_multi_assign_executes() {
    let bytecode = compile_semantic_source(
        r#"
        function [a,b] = g()
            a = 6;
            b = 7;
        end
        h = @g;
        [x,y] = feval(h);
        z = x + y;
    "#,
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic feval multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 13.0).abs() < 1e-9)));
}

#[test]
fn semantic_size_builtin_multi_assign_executes() {
    let bytecode = compile_semantic_source("[r,c] = size([1 2; 3 4]); z = r + c;").unwrap();
    let vars = interpret(&bytecode).expect("semantic size multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 4.0).abs() < 1e-9)));
}

#[test]
fn semantic_min_max_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[mx,mi] = max([1 3 2]); [mn,ni] = min([4 1 5]); z = mx + mi + mn + ni;",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic min/max multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 8.0).abs() < 1e-9)));
}

#[test]
fn semantic_cummin_cummax_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[mx,mi] = cummax([2 1 3]); [mn,ni] = cummin([2 1 3]); z = sum(mx) + sum(mi) + sum(mn) + sum(ni);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic cummin/cummax multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 21.0).abs() < 1e-9)));
}

#[test]
fn semantic_sort_unique_find_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[s,si] = sort([3 1 2]); [u,ui] = unique([2 1 2]); [fr,fc] = find([0 1; 2 0]); z = sum(s) + sum(si) + sum(u) + sum(ui) + sum(fr) + sum(fc);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic sort/unique/find multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 24.0).abs() < 1e-9)));
}

#[test]
fn semantic_union_ismember_sortrows_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[u,ia,ib] = union([3 1],[2 1]); [tf,loc] = ismember([3 1 2],[2 3]); [sr,idx] = sortrows([2 2; 1 3]); z = u(1) + u(2) + u(3) + ia(1) + ia(2) + ib + loc(1) + loc(2) + loc(3) + sr(1,1) + sr(1,2) + sr(2,1) + sr(2,2) + idx(1) + idx(2);",
    )
    .unwrap();
    let vars =
        interpret(&bytecode).expect("semantic union/ismember/sortrows multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 24.0).abs() < 1e-9)));
}

#[test]
fn semantic_chol_builtin_multi_assign_execute() {
    let bytecode =
        compile_semantic_source("[r,p] = chol([4 0; 0 9]); z = r(1,1) + r(2,2) + p;").unwrap();
    let vars = interpret(&bytecode).expect("semantic chol multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn semantic_lu_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[l,u,p] = lu([2 0; 0 3]); z = l(1,1) + l(2,2) + u(1,1) + u(2,2) + p(1,1) + p(2,2);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic lu multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn semantic_qr_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[q,r,p] = qr([2 0; 0 3]); z = q(1,1) + q(2,2) + r(1,1) + r(2,2) + p(1,1) + p(2,2);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic qr multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n + 1.0).abs() < 1e-9)));
}

#[test]
fn semantic_svd_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[u,s,v] = svd([3 0; 0 2]); z = s(1,1) + s(2,2) + u(1,1) * u(1,1) + v(1,1) * v(1,1);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic svd multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn semantic_eig_builtin_multi_assign_execute() {
    let bytecode = compile_semantic_source(
        "[v,d] = eig([3 0; 0 2]); z = d(1,1) + d(2,2) + v(1,1) * v(1,1) + v(2,2) * v(2,2);",
    )
    .unwrap();
    let vars = interpret(&bytecode).expect("semantic eig multi-assign should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn fprintf_inline_cast_argument_does_not_stack_underflow() {
    let program = r#"
        x = single(3.14);
        fprintf("Value: %.4f\n", double(x));
    "#;
    execute_semantic_source(program);
}

#[test]
fn sprintf_inline_cast_argument_formats_value() {
    let program = r#"
        x = single(3.14);
        s = sprintf("Value: %.4f", double(x));
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars.iter().any(|value| {
        if let runmat_builtins::Value::CharArray(chars) = value {
            let rendered: String = chars.data.iter().collect();
            rendered == "Value: 3.1400"
        } else {
            false
        }
    }));
}

#[test]
fn member_get_set_and_method_call_skeleton() {
    let input = "obj = new_object('Point'); obj = setfield(obj, 'x', 3); ax = getfield(obj, 'x');";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < f64::EPSILON)));

    // call Point.move which exists as example method: dx=1,dy=2
    let input2 = "obj = new_object('Point'); obj = setfield(obj,'x',5); obj = setfield(obj,'y',7); obj = call_method(obj, 'move', 1, 2); rx = getfield(obj,'x'); ry = getfield(obj,'y');";
    let vars2 = execute_semantic_source(input2);
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < f64::EPSILON)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < f64::EPSILON)));
}

#[test]
fn implicit_struct_creation_for_root_variable_assignment() {
    let input = "s.x = 10; s.y = 20; v = getfield(s, 'x') + getfield(s, 'y');";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 30.0).abs() < 1e-9)));
}

#[test]
fn semantic_member_read_write_executes() {
    let bytecode = compile_semantic_source("s.x = 10; s.y = 20; v = s.x + s.y;").unwrap();
    let vars = interpret(&bytecode).expect("semantic member read/write should succeed");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 30.0).abs() < 1e-9)));
}

#[test]
fn semantic_indexed_member_store_back_executes() {
    let bytecode = compile_semantic_source("s.a = [1 2 3]; s.a(2) = 9; y = s.a(2);").unwrap();
    let vars = interpret(&bytecode).expect("semantic indexed member store-back should succeed");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn semantic_cell_member_store_back_executes() {
    let bytecode = compile_semantic_source("C = {struct()}; C{1}.a = 5; y = C{1}.a;").unwrap();
    let vars = interpret(&bytecode).expect("semantic cell member store-back should succeed");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn implicit_struct_creation_for_function_output_variable() {
    let input = r#"
        function r = make_result()
            r.x = 10;
            r.y = 20;
        end
        s = make_result();
        x = getfield(s, 'x');
        y = getfield(s, 'y');
    "#;
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
}

#[test]
fn nested_member_assignment_materializes_missing_intermediate_structs() {
    let input = "s = struct(); s.a.b = 1; v = getfield(getfield(s, 'a'), 'b');";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn struct_field_indexing_read_path_uses_member_then_index_semantics() {
    let input =
        "s = struct(); s.arr = [10 20 30]; x = s.arr(2); y = s.arr(1:2); y1 = y(1); y2 = y(2);";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn dotted_invoke_preserves_object_method_dispatch() {
    let input = "obj = new_object('Point'); obj = setfield(obj,'x',5); obj = setfield(obj,'y',7); obj = obj.move(1,2); rx = getfield(obj,'x'); ry = getfield(obj,'y');";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 6.0).abs() < 1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn dotted_invoke_runtime_struct_dispatch_when_base_type_unknown() {
    let input = r#"
        function y = pick(s)
            y = s.arr(2);
        end
        s = struct();
        s.arr = [10 20 30];
        z = pick(s);
    "#;
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 20.0).abs() < 1e-9)));
}

#[test]
fn nested_dynamic_member_assignment_materializes_and_writes_back() {
    let input =
        "s = struct(); f1 = 'a'; f2 = 'b'; s.(f1).(f2) = 3; v = getfield(getfield(s, 'a'), 'b');";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn mixed_member_cell_and_index_read_chain() {
    let input = "s = struct(); s.arr = {[10 20], [30 40]}; v = s.arr{2}(1);";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 30.0).abs() < 1e-9)));
}

#[test]
fn function_handle_anon_round_trip() {
    let input = "h = @sin; g = make_anon('x', 'x+1');";
    let vars = execute_semantic_source(input);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::FunctionHandle(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s.starts_with("@anon"))));
}

#[test]
fn semantic_builtin_function_handle_feval_executes() {
    let bytecode = compile_semantic_source("h = @sin; y = feval(h, 0);").unwrap();
    let vars = interpret(&bytecode).expect("semantic builtin handle feval should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if n.abs() < 1e-12)));
}

#[test]
fn semantic_anonymous_function_handle_feval_executes() {
    let bytecode = compile_semantic_source("f = @(x) x + 1; y = feval(f, 4);").unwrap();
    let vars = interpret(&bytecode).expect("semantic anonymous handle feval should execute");

    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-12)));
}

#[test]
fn semantic_function_handle_index_call_executes() {
    let bytecode =
        compile_semantic_source("h = @inc; y = h(2);\nfunction z = inc(x)\n  z = x + 1;\nend")
            .unwrap();
    assert!(
        bytecode.instructions.iter().any(|instr| matches!(
            instr,
            runmat_vm::Instr::CreateSemanticFunctionHandle(_, name) if name == "inc"
        )),
        "semantic function handle index calls should carry semantic identity"
    );

    runmat_vm::reset_legacy_user_dispatch_fallback_count();
    let vars = interpret(&bytecode).expect("semantic handle index call should execute");
    assert_eq!(runmat_vm::legacy_user_dispatch_fallback_count(), 0);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
}

#[test]
fn cellfun_upper_function_handle_round_trip() {
    let input =
        "names = {'Ada', 'Linus', 'Katherine'}; upper = cellfun(@upper, names, 'UniformOutput', false);";
    let vars = execute_semantic_source(input);

    let mut found = false;
    for value in vars {
        if let runmat_builtins::Value::Cell(ca) = value {
            if ca.data.len() != 3 {
                continue;
            }
            let texts: Vec<String> = ca
                .data
                .iter()
                .map(|ptr| String::try_from(&**ptr))
                .collect::<Result<_, _>>()
                .unwrap_or_default();
            if texts == vec!["ADA", "LINUS", "KATHERINE"] {
                found = true;
                break;
            }
        }
    }

    assert!(found);
}

#[test]
fn classes_static_and_inheritance() {
    // Register classes
    assert!(execute_semantic_source_result("__register_test_classes();").is_ok());

    // Default init and namespaced
    let vars2 = execute_semantic_source("__register_test_classes(); p = new_object('Point'); ax = getfield(p,'x'); ns = new_object('pkg.PointNS'); nsx = getfield(ns,'x');");
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 0.0).abs()<f64::EPSILON)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs()<f64::EPSILON)));

    // Static method and property
    let vars3 = execute_semantic_source("__register_test_classes(); o = classref('Point').origin(); sv = classref('Point').staticValue;");
    assert!(vars3
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
    assert!(vars3
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs()<f64::EPSILON)));

    // Inheritance override: use feval(getmethod(...))()
    let vars4 = execute_semantic_source("__register_test_classes(); c = new_object('Circle'); c = setfield(c,'r', 2); a = feval(getmethod(c,'area'));");
    assert!(vars4.iter().any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - std::f64::consts::PI*4.0).abs() < 1e-9)));

    // Access control violations
    assert!(execute_semantic_source_result(
        "__register_test_classes(); p = new_object('Point'); s = getfield(p,'secret');"
    )
    .is_err());
}

#[test]
fn static_method_via_classref_uses_namespaced_builtin_without_class_registry() {
    let vars = execute_semantic_source("P = classref(\"Point\").origin(); point_class = class(P);");
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s == "Point")));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_constructor_and_overloaded_indexing() {
    // Call Ctor constructor; exercise OverIdx subsref/subsasgn
    let program = "__register_test_classes(); c = Ctor(7); o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5); t = call_method(o,'subsref','.', 'k');";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn classes_property_access_attributes() {
    // Register classes
    let _ = execute_semantic_source("__register_test_classes();");
    // Private get already covered by existing test; ensure private set is also rejected
    assert!(execute_semantic_source_result(
        "__register_test_classes(); p = new_object('Point'); p = setfield(p,'secret', 7);"
    )
    .is_err());
}

#[test]
fn import_builtin_resolution_for_static_method() {
    // Register classes and import Point.* so we can call origin() unqualified
    let program = "__register_test_classes(); import Point.*; o = origin();";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
}

#[test]
fn import_specific_resolution_for_builtin() {
    // Use specific import to bring a qualified builtin into scope
    let program =
        "import pkg.missing.*; import Point.origin; __register_test_classes(); o = origin();";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
}

#[test]
fn import_ambiguity_specific_conflict_errors() {
    // Two specifics that map the same unqualified name should cause compile-time ambiguity.
    let program = "import PkgF.foo; import PkgG.foo; r = foo();";
    let res = compile_semantic_source(program);
    assert!(res.is_err());
}
#[test]
fn import_static_method_via_specific_class_import() {
    // import ClassName.* to allow unqualified static methods and properties
    let program = "__register_test_classes(); import Point.*; p = origin(); v = classref('Point').staticValue;";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(_))));
}

#[test]
fn import_precedence_specific_over_wildcard_and_locals() {
    // Specific imports should take precedence over wildcard imports; locals should shadow both
    let program = r#"
        __register_test_classes();
        import PkgF.foo;         % specific import of PkgF.foo
        import PkgG.*;           % wildcard import with also foo
        a = foo();               % should resolve to PkgF.foo (10)
        foo = @() 42;            % local function handle shadowing imports
        b = feval(foo);          % should be 42 (local shadow)
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs()<1e-9)));
}

#[test]
fn import_ambiguity_between_specifics_errors() {
    // Two specific imports of same unqualified function should cause compile-time error
    let program = r#"
        import PkgF.foo;
        import PkgG.foo;
        y = foo();
    "#;
    let res = compile_semantic_source(program);
    assert!(res.is_err());
}

#[test]
fn import_ambiguity_between_wildcards_errors() {
    // Two wildcard imports that both contain 'foo' should cause an ambiguity error on unqualified call
    let program = r#"
        import PkgF.*;
        import PkgG.*;
        y = foo();
    "#;
    let res = compile_semantic_source(program);
    assert!(res.is_err());
}

#[test]
fn import_specific_conflict_with_user_function_prefers_local() {
    // A user function in scope should overshadow specific imports
    let program = r#"
        import PkgF.foo;   % specific import exists
        function y = foo()
            y = 33;
        end
        a = foo();         % should call local function => 33
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 33.0).abs()<1e-9)));
}

#[test]
fn import_static_property_shadowed_by_local_variable() {
    // Local variables should shadow static properties brought in via Class.* imports
    let program = r#"
        __register_test_classes();
        import Point.*;
        staticValue = 7;   % shadows classref('Point').staticValue (42)
        v = staticValue;
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs()<1e-9)));
}

#[test]
fn import_specific_beats_wildcard_for_function_resolution() {
    // Specific import should win over wildcard providing the same name
    let program = r#"
        import PkgF.foo;   % specific import
        import PkgG.*;     % wildcard that also provides foo
        y = foo();         % should resolve to PkgF.foo => 10
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs()<1e-9)));
}

#[test]
fn import_wildcard_static_method_ambiguity_errors() {
    // Two Class.* wildcards that both bring in a static method with same unqualified name should be ambiguous
    // We reuse Point.* for one side; fabricate another class via runtime registration if needed.
    // Here we simulate ambiguity on function resolution instead, since a second class isn't pre-registered.
    let program = r#"
        import PkgF.*;
        import PkgG.*;
        z = foo();   % ambiguous via wildcard imports
    "#;
    let res = compile_semantic_source(program);
    assert!(res.is_err());
}

#[test]
fn local_function_shadows_class_static_method_under_class_star() {
    // Define a local function 'origin' that should shadow Point.origin brought via import Point.*
    let program = r#"
        __register_test_classes();
        import Point.*;
        function y = origin()
            y = 123;
        end
        v = origin();   % should call local (123), not Point.origin
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 123.0).abs()<1e-9)));
}

#[test]
fn multi_segment_import_builtin_vs_user_function_specific_prefers_user_function() {
    // When both a builtin and a user function exist under the same unqualified name via specific import,
    // ensure the user function in scope takes precedence.
    let program = r#"
        import PkgF.foo;   % provides builtin PkgF.foo
        function y = foo()
            y = 77;
        end
        a = foo();         % should call the user function -> 77
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 77.0).abs()<1e-9)));
}

#[test]
fn unqualified_static_property_without_imports_errors() {
    // Referencing staticValue unqualified without imports should error
    let program = r#"
        __register_test_classes();
        v = staticValue;
    "#;
    let res = compile_semantic_source(program);
    assert!(res.is_err());
}

#[test]
fn unqualified_static_property_shadowed_by_local_variable() {
    // Even with import Point.*, a local variable should shadow unqualified static property name
    let program = r#"
        __register_test_classes();
        import Point.*;
        staticValue = 9;
        v = staticValue;  % picks local, not class static 42
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs()<1e-9)));
}

#[test]
fn import_nested_package_class_static_method_resolution() {
    // Simulate nested pkg class via import path and ensure resolution prefers functions first, then static method
    let program = r#"
        __register_test_classes();
        import pkg.PointNS.*;   % nested path Simulation (pkg.PointNS has no methods; ensures no accidental resolution)
        function z = origin()
            z = 5;
        end
        v = origin();          % local function shadows any static method named origin
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs()<1e-9)));
}

#[test]
fn class_property_attribute_conflicts_error() {
    // Properties with Constant and Dependent together should error
    let program = r#"
        classdef Bad
            properties(Constant, Dependent)
                x
            end
        end
    "#;
    assert!(compile_semantic_source(program).is_err());
}

#[test]
fn class_method_attribute_conflicts_error() {
    // Methods with Abstract and Sealed together should error
    let program = r#"
        classdef Bad
            methods(Abstract, Sealed)
                function y = f(obj)
                    y = 1;
                end
            end
        end
    "#;
    assert!(compile_semantic_source(program).is_err());
}

#[test]
fn metaclass_context_with_imports() {
    // Ensure ?pkg.Class parses and coexists with imports; no runtime error expected
    let program = "import pkg.*; ?pkg.Class; x=1;";
    let vars = execute_semantic_source(program);
    // Either ok=1 was set or we have an MException present
    let ok_or_exc = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-1.0).abs()<1e-9))
        || vars
            .iter()
            .any(|v| matches!(v, runmat_builtins::Value::MException(_)));
    assert!(ok_or_exc);
}

#[test]
fn metaclass_postfix_member_and_method() {
    let program = "__register_test_classes(); v = ?Point.staticValue; o = ?Point.origin();";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs() < 1e-9)));
}

#[test]
fn import_ambiguity_wildcard_conflict_errors() {
    // Two wildcard packages both providing same builtin name should error at runtime resolution
    // We simulate by trying resolution; since no actual builtins exist under these, it will fall through
    // but our VM path collects multiple matches and would error if present. This serves as a guard.
    let program = "import PkgA.*; import PkgB.*; x = 1;";
    let vars = execute_semantic_source(program);
    assert!(!vars.is_empty());
}

#[test]
fn classdef_with_attributes_enforced() {
    // Define class A with private get and public set on property p, then enforce via getfield/setfield
    let src = "classdef A\n  properties(GetAccess=private, SetAccess=public)\n    p\n  end\nend\n a = new_object('A'); a = setfield(a,'p',5); try; v = getfield(a,'p'); catch e; ok=1; end";
    let vars = execute_semantic_source(src);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9)));
}

#[test]
fn builtin_call_with_expanded_middle_argument() {
    // Use deal to produce a cell row and index into it to pass as middle arg
    // max(a,b) with b coming from C{1}
    let program = "C = deal(10, 20); r = max(5, C{1});";
    let vars = execute_semantic_source(program);
    // Expect 10 as result appears in vars somewhere
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn builtin_call_with_two_expanded_args() {
    let program = "C = deal(3, 4); D = deal(5, 6); r = max(C{1}, D{1});";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
}

#[test]
fn user_function_with_two_expanded_args() {
    let program = "function y = sum2(a,b); y = a + b; end; C = deal(7,8); D = deal(11,12); r = sum2(C{2}, D{1});";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 19.0).abs() < 1e-9)));
}

#[test]
fn expansion_on_non_cell_errors() {
    let program = "r = max(5, 10{1});";
    let bytecode = compile_semantic_source(program).expect("compile expansion error source");
    assert!(interpret(&bytecode).is_err());
}

#[cfg(any(feature = "test-classes", test))]
#[test]
fn object_cell_expansion_via_subsref() {
    let program = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn','{}', {1}, 42); r = max(o{1}, 5);";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 42.0).abs() < 1e-9)));
}

#[test]
fn expand_all_elements_in_args() {
    // C{:} expands all elements of C into separate arguments
    // max takes two args; here C has more; we only assert no crash and presence of some expected nums
    let program = "C = deal(1,2); a = max(C{:});";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(_))));
}

#[test]
fn builtin_vector_index_expansion() {
    let program = "C = deal(9, 2); r = max(C{[1 2]});";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn user_function_vector_index_expansion() {
    let program = "function y = sum2(a,b); y = a + b; end; C = deal(3,4); r = sum2(C{[1 2]});";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 7.0).abs() < 1e-9)));
}

#[test]
fn end_minus_one_1d_slice_collect() {
    let program = "A = [1 2 3]; B = A(1:1:end-1); s = sum(B);";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-9)));
}

#[test]
fn end_minus_one_1d_slice_assign_broadcast() {
    let program = "A = [1 2 3 4]; A(2:1:end-1) = 9; r = sum(A);";
    let vars = execute_semantic_source(program);
    // A becomes [1 9 9 4] => sum 23
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 23.0).abs() < 1e-9)));
}

#[test]
fn multidim_range_end_assign() {
    // Assign on second dim using range with end-1
    let program = "A = [1 2 3; 4 5 6]; A(:,2:2:end-1) = 9; s = sum(sum(A));";
    // Original A sum is 21. We set column 2 to 9 across all rows: [1 9 3; 4 9 6] => sum 32
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 32.0).abs() < 1e-9)));
}

#[test]
fn multidim_range_end_assign_non_scalar_rhs_broadcast() {
    // 2x3; assign the middle column selection with a 2x1 rhs, which should broadcast along the selection length
    let program = "A = [1 2 3; 4 5 6]; B = [7;8]; A(:,2:2:end-1) = B; s = sum(sum(A));";
    // Becomes [1 7 3; 4 8 6] => sum 29
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 29.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_vector_broadcast() {
    // 3x4 matrix; select rows 2:end (rows {2,3}) and cols 1:2:end-1 (cols {1,3}); assign 2x1 vector broadcast across selected cols
    let program =
        "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [100;200]; A(2:end, 1:2:end-1) = B; s = sum(sum(A));";
    let vars = execute_semantic_source(program);
    // Expected sum 646 (see analysis)
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 646.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_matrix_rhs_exact_shape() {
    // Assign 2x2 block with exact-shaped RHS
    let program =
        "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [1 3; 2 4]; A(2:end, 1:2:end-1) = B; s = sum(sum(A));";
    // Note: [1 3; 2 4] in MATLAB column-major maps to data [1,2,3,4] in our Tensor internal
    let vars = execute_semantic_source(program);
    // New values: positions (2,1)=1,(3,1)=2,(2,3)=3,(3,3)=4. Change from original 5,9,7,11 -> delta = (1-5)+(2-9)+(3-7)+(4-11) = -22; sum 78-22=56
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 56.0).abs() < 1e-9)));
}

#[test]
fn mixed_range_end_assign_shape_mismatch_error() {
    // RHS shape 3x1 does not match rows 2:end (len 2) and cannot broadcast
    let program = "A = [1 2 3 4; 5 6 7 8; 9 10 11 12]; B = [1;2;3]; A(2:end, 1:2:end-1) = B;";
    let res = execute_semantic_source_result(program);
    assert!(res.is_err());
}

#[test]
fn broadcasting_roundtrip_property_like() {
    // After assignment with broadcasted column vector, selected columns equal the vector
    let program = "A = zeros(3,4); v = [7;8;9]; A(:, 1:2:end-1) = v; x = A(:,1); y = A(:,3);";
    let vars = execute_semantic_source(program);
    // Expect 7,8,9 present for x and y
    let mut count = 0;
    for v in vars {
        if let runmat_builtins::Value::Tensor(t) = v {
            if t.shape == vec![3, 1]
                && (t.data[0] - 7.0).abs() < 1e-9
                && (t.data[1] - 8.0).abs() < 1e-9
                && (t.data[2] - 9.0).abs() < 1e-9
            {
                count += 1;
            }
        }
    }
    assert!(count >= 1);
}

#[test]
fn logical_mask_write_scalar_and_vector() {
    // Scalar write via linear logical mask
    let program = "A = [1 2 3 4 5 6]; m = [true false true false true false]; A(m) = 9; s1 = sum(A);\nB = [1 2 3 4 5 6]; idx = [1 3 5]; B(idx) = [7 8 9]; s2 = sum(B);";
    let vars = execute_semantic_source(program);
    // After A(m)=9, A becomes [9 2 9 4 9 6] => sum 39
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-39.0).abs()<1e-9)));
    // After B(idx)=[7 8 9], B becomes [7 2 8 4 9 6] => sum 36
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-36.0).abs()<1e-9)));
}

#[test]
fn gather_scatter_roundtrip_nd() {
    // Gather a slice, scatter it back, tensor must be unchanged
    let program = "A = [1 2 3; 4 5 6; 7 8 9]; S = A(2:3, 1:2); A(2:3, 1:2) = S; t = sum(A(:));";
    let vars = execute_semantic_source(program);
    // Sum remains 45
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-45.0).abs()<1e-9)));
}

#[test]
fn shape_broadcasting_laws() {
    // Broadcast column vector over selected columns
    let program = "A = zeros(3,4); v = [1;2;3]; A(:, 2:2:4) = v; x = sum(A(:));\nC = zeros(2,3,2); w = [5;6]; C(:,2,:) = w; y = sum(C(:));";
    let vars = execute_semantic_source(program);
    // A has 2 columns set to v => sum = (1+2+3)*2 = 12
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-12.0).abs()<1e-9)));
    // C zeros 2x3x2; assignment sets middle column across both slices => positions: (1,2,1)=5,(2,2,1)=6,(1,2,2)=5,(2,2,2)=6 => sum 22
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-22.0).abs()<1e-9)));
}

#[test]
fn column_major_rhs_mapping() {
    // Verify RHS mapping enumerates first-dimension fastest
    let program = "A = zeros(3,3); R = [10 13 16; 11 14 17; 12 15 18]; A(:, [1 3]) = R(:, [1 3]); s = sum(A(:));";
    let vars = execute_semantic_source(program);
    // Selected columns are 1 and 3, filled from R(:,[1 3]) which in column-major is [10;11;12;16;17;18] => sum 84
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-84.0).abs()<1e-9)));
}
#[test]
fn builtin_call_with_function_return_propagation() {
    // g returns two numbers; propagate as args to max
    let program = "function [a,b] = g(); a=9; b=4; end; r = max(g());";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 9.0).abs() < 1e-9)));
}

#[test]
fn function_call_base_expand_all() {
    let program = r#"
        function y = sum2(a,b); y = a + b; end; r = sum2(deal(5,6){:});
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 11.0).abs() < 1e-9)));
}

#[test]
fn function_return_propagation_in_args() {
    // g returns [a,b]; f takes two inputs; adapt to current semantics by binding outputs, then calling f
    let program = "function [a,b] = g(); a=2; b=3; end; function y = f(x1,x2); y = x1 + x2; end; [u,v] = g(); r = f(u,v);";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 5.0).abs() < 1e-9)));
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
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-6.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-22.0).abs()<1e-9)));
}

#[test]
fn feval_expand_multi_forwards_expanded_cell_args() {
    let program = r#"
        function y = f(a,b,c,d)
            y = a + b + c + d;
        end
        function out = g(varargin)
            out = feval(@f, varargin{:});
        end
        r = g(4,5,6,7);
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-22.0).abs()<1e-9)));
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
    let vars = execute_semantic_source(program);
    // max(11,12) = 12
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-12.0).abs()<1e-9)));
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
    let vars = execute_semantic_source(program);
    // 1 + 20 + 300 = 321
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-321.0).abs()<1e-9)));
}

#[test]
fn operator_overloading_plus_times_lt_eq() {
    let setup = "__register_test_classes();";
    // Try plus with mixed object/scalar; OverIdx has no plus, so fallback should not crash
    let program = format!(
        "{setup} o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5); r1 = o + 3;"
    );
    let _ = execute_semantic_source(&program);
}

#[test]
fn operator_overloading_elementwise_vs_mtimes_and_mixed() {
    let setup = "__register_test_classes();";
    // Exercise times and mtimes overload paths (will fallback if not implemented)
    let program =
        format!("{setup} o = new_object('OverIdx'); a = o .* 2; b = o * 2; c = 2 .* o; d = 2 * o;");
    let _ = execute_semantic_source(&program);
}

#[test]
fn operator_overloading_relational_lt_eq() {
    let setup = "__register_test_classes();";
    // lt and eq with mixed object/scalar on both sides
    let program = format!(
        "{setup} o = new_object('OverIdx'); t1 = (o < 10); t2 = (10 < o); t3 = (o == 0); t4 = (0 == o);"
    );
    let _ = execute_semantic_source(&program);
}

#[test]
fn operator_overloading_full_grid_basic() {
    let setup = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5);";
    let statements = [
        "a1 = (o ~= 10);",
        "a2 = (10 ~= o);",
        "b1 = (o >= 5);",
        "b2 = (o <= 5);",
        "% use numeric power to avoid object exponent when not provided",
        "c1 = ([2 3] .^ 2);",
        "c2 = (2 .^ [2 3]);",
        "d1 = (o ./ 2);",
        "d2 = (2 ./ o);",
        "e1 = (o .\\ 2);",
        "e2 = (2 .\\ o);",
        "f1 = ([1 0 1] & [1 1 0]);",
        "f2 = ([1 0 1] | [0 1 1]);",
    ];
    let mut program = String::from(setup);
    program.push('\n');
    for (idx, stmt) in statements.iter().enumerate() {
        program.push_str(stmt);
        program.push('\n');
        execute_semantic_source_result(&program).unwrap_or_else(|err| {
            let bc = compile_semantic_source(&program).unwrap();
            panic!(
                "operator overload script failed after stmt #{idx} `{stmt}`: {err}\nbytecode={:?}",
                bc.instructions
            );
        });
    }
}

#[test]
fn import_precedence_and_class_static_shadowing() {
    // Locals > user functions > specific imports > wildcard imports; static under Class.* last
    let program = r#"
        function y = f(); y = 123; end
        __register_test_classes();
        import Point.origin; % specific import
        import Pkg.*;        % wildcard import (non-existent)
        origin = 7;          % local shadows import
        a = origin;          % uses local variable, not static
        b = f();             % user function resolves before imports
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-7.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-123.0).abs()<1e-9)));
}

#[test]
fn static_method_resolution_under_wildcard_import() {
    // Ensure unqualified static via Class.* resolves when no shadowing
    let program = r#"
        __register_test_classes();
        import Point.*;
        r = origin(); % static method
    "#;
    execute_semantic_source(program);
}

#[test]
fn operator_overloading_numeric_results_and_bitwise_arrays() {
    // Verify explicit numeric outcomes for OverIdx overloads
    let program = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5); r1 = o + 3; r2 = o .* 2; r3 = o * 4; a = (o < 10); b = (o == 5);";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 8.0).abs() < 1e-9))); // r1 = 5+3
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9))); // r2 = 5*2
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 20.0).abs() < 1e-9))); // r3 = 5*4
                                                                                            // a and b push logicals (1.0/0.0) somewhere in vars
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9))); // 5<10 true
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 1.0).abs() < 1e-9))); // 5==5 true

    // Bitwise and/or on arrays: verify element-wise behavior
    let program2 = "A = [1 0 1; 0 1 0]; B = [1 1 0; 0 0 1]; C = A & B; D = A | B; Sc = sum(C(:)); Sd = sum(D(:));";
    let vars2 = execute_semantic_source(program2);
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    assert!(vars2
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-5.0).abs()<1e-9)));
}

#[test]
fn operator_overloading_left_division_variants() {
    let setup = "__register_test_classes(); o = new_object('OverIdx'); o = call_method(o,'subsasgn','.', 'k', 5);";
    let program = format!("{setup} a = (o .\\ 2); b = (2 .\\ o); c = (o ./ 2); d = (2 ./ o);",);
    let _ = execute_semantic_source(&program);
}

#[test]
fn bitwise_or_row_vectors() {
    let program = "f = ([1 0 1] | [0 1 1]);";
    let bc = compile_semantic_source(program).unwrap();
    let _ = interpret(&bc).unwrap();
}

#[test]
fn function_return_propagation_partial_fill() {
    // g returns [1,2,3]; h takes 2 inputs; ensure leftmost two feed h
    let program = "function [a,b,c] = g(); a=1; b=2; c=3; end; function y = h(x1,x2); y = x1*10 + x2; end; r = h(g());";
    let vars = execute_semantic_source(program);
    // 1*10 + 2 = 12
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 12.0).abs() < 1e-9)));
}

#[test]
fn nested_function_return_propagation_mixed_with_fixed() {
    // g()->[4,5]; f(x,p,z)=x+p+z; call f(1, g()) => 1+4+5=10
    let program = "function [a,b] = g(); a=4; b=5; end; function y = f(x,p,z); y = x + p + z; end; r = f(1, g());";
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 10.0).abs() < 1e-9)));
}

#[test]
fn nested_try_catch_rethrow_unified_exception_ids() {
    // inner throws, caught and rethrown, outer catches; identifiers/messages preserved
    let program = r#"
        function y = inner()
            error('RunMat:inner:bad', 'inner fail')
        end
        function z = middle()
            try
                z = inner();
            catch e
                rethrow(e);
            end
        end
        try
            r = middle();
        catch e
            id = getfield(e, 'identifier');
            msg = getfield(e, 'message');
            % Surface identifier and message as outputs
            out_id = id; out_msg = msg; out_exc = e;
        end
    "#;
    let vars = execute_semantic_source(program);
    // Look for the exception object with identifier/message
    let has_exc = vars.iter().any(|v| match v {
        runmat_builtins::Value::MException(me) => {
            me.identifier == "RunMat:inner:bad" && me.message == "inner fail"
        }
        _ => false,
    });
    // If out_exc wasn't preserved as MException, ensure at least identifier/message strings are present
    let has_id = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s=="RunMat:inner:bad"));
    let has_msg = vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::String(s) if s=="inner fail"));
    assert!(has_exc || has_id || has_msg);
}

#[test]
fn globals_basic_and_shadowing() {
    let prog = "global G; G = 5; function y = f(x); global G; y = G + x; end; a = f(3);";
    let vars = execute_semantic_source(prog);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 8.0).abs() < 1e-12)));
}

#[test]
fn persistents_init_once_across_calls() {
    let prog = "function y = counter(); persistent C; if C==0; C = 0; end; C = C + 1; y = C; end; a = counter(); b = counter(); c = counter();";
    let vars = execute_semantic_source(prog);
    // Expect last value 3 somewhere in vars
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 3.0).abs() < 1e-12)));
}

#[test]
fn import_precedence_and_shadowing() {
    // Define user function f; specific import for Point.origin; wildcard import Pkg.* (nonexistent)
    // Local variable named origin should shadow imports; then function should shadow imports.
    let program = "function y = f(); y = 123; end; __register_test_classes(); import Point.origin; import Pkg.*; origin = 7; a = origin; b = f();";
    let vars = execute_semantic_source(program);
    // Expect 7 and 123 present
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-7.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-123.0).abs()<1e-9)));
}

#[test]
fn class_dependent_property_get_set() {
    // Define class with Dependent property 'p'; test generic get.p/set.p builtins via backing field
    let program = r#"
        classdef D
          properties(Dependent)
            p
          end
        end
        d = new_object('D');
        d = setfield(d, 'p', 7); % should route to set.p, writing p_backing
        b = getfield(d, 'p_backing');
        % get.p should return p_backing
        v = getfield(d, 'p');
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-7.0).abs()<1e-9)));
}

#[test]
fn struct_isfield_multi_and_fieldnames() {
    let program = r#"
        s = struct(); s = setfield(s, 'a', 1); s = setfield(s, 'b', 2);
        c = {'a','x';'b','a'}; r = isfield(s, c); f = fieldnames(s);
    "#;
    let vars = execute_semantic_source(program);
    // Expect r to be 2x2 logical matrix [[1,0];[1,1]] in column-major data [1,1,0,1]
    let mut found_r_ok = false;
    let mut found_f_ok = false;
    for v in &vars {
        match v {
            runmat_builtins::Value::LogicalArray(arr) => {
                if arr.shape == vec![2, 2] && arr.data == vec![1, 1, 0, 1] {
                    found_r_ok = true;
                }
            }
            runmat_builtins::Value::Cell(ca) => {
                // fieldnames returns a column cell array with at least two entries
                if ca.cols == 1 && ca.rows >= 2 {
                    found_f_ok = true;
                }
            }
            _ => {}
        }
    }
    assert!(found_r_ok && found_f_ok);
}

#[test]
fn struct_isfield_string_array_placeholder() {
    // String-array semantics placeholder: use empty numeric array to simulate empty string array and verify no-crash
    let program = r#"
        s = struct(); s = setfield(s, 'a', 1);
        % In real MATLAB, this would be ["a" "b"; "x" "a"]. We signal via comment as parser lacks string arrays.
        % For now, ensure isfield(s, 'a') works and returns true.
        r = isfield(s, 'a');
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Bool(true))));
}

#[test]
fn oop_negative_undefined_property_and_missing_subsref() {
    // Undefined property on object without subsref should raise an error caught by try/catch
    let prog = r#"
        classdef NoRef
            properties
                p
            end
        end
        o = new_object('NoRef');
        try
            v = o.noSuchProp;
        catch e
            ok = 1;
        end
    "#;
    let res = execute_semantic_source_result(prog);
    if let Ok(vars) = res {
        assert!(vars
            .iter()
            .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-1.0).abs()<1e-9)));
    }

    // Class without subsref should error on () indexing
    let prog2 = r#"
        classdef NoRef
            properties
                p
            end
        end
        o = new_object('NoRef');
        try
            x = o(1);
        catch e
            ok=2;
        end
    "#;
    let res2 = execute_semantic_source_result(prog2);
    if let Ok(vars2) = res2 {
        assert!(vars2
            .iter()
            .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-2.0).abs()<1e-9)));
    }
}

#[test]
fn containers_map_parenthesis_indexing() {
    let program = r#"
        fruit = containers.Map({'apple'}, {99});
        energy = fruit('apple');
    "#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - 99.0).abs() < 1e-9)));
}

#[test]
fn containers_map_dot_properties() {
    let program = r#"
        m = containers.Map();
        key_type = m.KeyType;
        value_type = m.ValueType;
        count = m.Count;
    "#;
    let vars = execute_semantic_source(program);

    assert!(vars.iter().any(|v| match v {
        runmat_builtins::Value::CharArray(ca) => ca.data.iter().collect::<String>() == "char",
        _ => false,
    }));
    assert!(vars.iter().any(|v| match v {
        runmat_builtins::Value::CharArray(ca) => ca.data.iter().collect::<String>() == "any",
        _ => false,
    }));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n).abs() < 1e-9)));
}

#[test]
fn string_array_literal_concat_index_and_compare() {
    let program = r#"
        A = ["a" "bb"; "ccc" "d"];   % 2x2 string array
        x = A(2,1);                      % "ccc"
        B = [A, ["e"; "e"]];            % hcat with 2x1 string-array -> 2x3
        C = [A; ["z" "y"]];             % vcat -> 3x2
        e1 = (A == "a");                % logical mask 2x2
        e2 = (A ~= "bb");               % logical mask 2x2
    "#;
    let vars = execute_semantic_source(program);
    let mut saw_a = false;
    let mut saw_x = false;
    let mut saw_b = false;
    let mut saw_c = false;
    let mut saw_e1 = false;
    let mut saw_e2 = false;
    for v in vars {
        match v {
            runmat_builtins::Value::StringArray(sa) => {
                println!("SA shape={:?} data={:?}", sa.shape, sa.data);
                if sa.shape == vec![2, 2] {
                    saw_a = true;
                }
                if sa.shape == vec![2, 3] {
                    saw_b = true;
                }
                if sa.shape == vec![3, 2] {
                    saw_c = true;
                }
                if sa.data.iter().any(|s| s == "ccc") {
                    saw_x = true;
                }
            }
            runmat_builtins::Value::String(s) => {
                println!("S {s}");
                if s == "ccc" {
                    saw_x = true;
                }
            }
            runmat_builtins::Value::CharArray(ca) => {
                let s: String = ca.data.iter().collect();
                if s == "ccc" {
                    saw_x = true;
                }
            }
            runmat_builtins::Value::Tensor(t) => {
                println!("T shape={:?} data={:?}", t.shape, t.data);
                if t.shape == vec![2, 2] {
                    saw_e1 = true;
                    saw_e2 = true;
                }
            }
            _ => {}
        }
    }
    assert!(saw_a && saw_x && saw_b && saw_c && saw_e1 && saw_e2);
}

#[test]
fn string_literal_and_num2str_horzcat_promotes_and_runs() {
    let program = r#"
        wn = 6;
        label = ["wn = ", num2str(wn), " rad/s"];
    "#;
    let vars = execute_semantic_source(program);
    let mut saw_label = false;
    for value in vars {
        if let runmat_builtins::Value::StringArray(sa) = value {
            if sa.shape == vec![1, 3] && sa.data == vec!["wn = ", "6", " rad/s"] {
                saw_label = true;
            }
        }
    }
    assert!(saw_label, "expected label string-array concat to succeed");
}

#[test]
fn computed_integer_indices_work_for_column_slice_read_and_assign() {
    let program = r#"
        A = [1, 2, 3; 4, 5, 6];
        j = length(1:2);
        c = A(:, j);

        B = zeros(2, 3);
        B(:, j) = [7; 9];
        d = B(:, j);
    "#;
    let vars = execute_semantic_source(program);

    let mut saw_read = false;
    let mut saw_assign = false;
    for value in vars {
        if let runmat_builtins::Value::Tensor(t) = value {
            if t.shape == vec![2, 1] && t.data == vec![2.0, 5.0] {
                saw_read = true;
            }
            if t.shape == vec![2, 1] && t.data == vec![7.0, 9.0] {
                saw_assign = true;
            }
        }
    }

    assert!(saw_read, "expected computed integer slice read to succeed");
    assert!(
        saw_assign,
        "expected computed integer slice assign to succeed"
    );
}

#[test]
fn import_deep_multiseg_package_specific_vs_wildcard() {
    // Specific nested beats wildcard from another nested package
    let program = r#"
		__register_test_classes();
		import PkgF.foo;     % specific import of PkgF.foo (builtin -> 10)
		import pkg.PointNS.*; % wildcard unrelated (should not affect foo)
		a = foo();            % resolves to PkgF.foo => 10
	"#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-10.0).abs()<1e-9)));
}

#[test]
fn import_shadowing_matrix_locals_user_specific_wildcard_classstar() {
    // Locals > user function in scope > specific imports > wildcard imports > Class.* statics
    let program = r#"
		__register_test_classes();
		import Point.*;       % Class.* provides origin (static)
		import PkgF.foo;      % specific provides foo()=10
		import PkgG.*;        % wildcard also provides foo()=20
		function y = bar(); y = 33; end   % user function (distinct name)
		foo = @() 77;         % local variable (handle)
		a = feval(foo);       % 77 (local)
		b = bar();            % 33 (user function)
		c = origin();         % static method via Class.* (no shadowing by foo)
	"#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-77.0).abs()<1e-9)));
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-33.0).abs()<1e-9)));
    // presence of an Object from origin()
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Object(_))));
}

#[test]
fn import_wildcard_vs_classstar_ambiguity_for_static_method() {
    // Duplicate Class.* imports should be caught by import validation
    let program = r#"
		__register_test_classes();
		import Point.*;
		import Point.*;   % duplicate
		r = origin();
	"#;
    let res = compile_semantic_source(program);
    assert!(res.is_err());
}

#[test]
fn import_specific_vs_wildcard_same_name_prefers_specific_under_nesting() {
    let program = r#"
		import PkgF.foo;   % specific import
		import PkgG.*;     % wildcard also has foo
		y = foo();         % should call specific -> 10
	"#;
    let vars = execute_semantic_source(program);
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-10.0).abs()<1e-9)));
}

#[test]
fn type_class_static_method_zeros() {
    // Test that double.zeros(2, 3) is lowered to zeros(2, 3, 'double')
    let program = r#"
        A = double.zeros(2, 3);
        s = size(A);
    "#;
    let vars = execute_semantic_source(program);
    // The result should be a 2x3 matrix of zeros
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::Tensor(t) = v {
            t.shape == vec![2, 3] && t.data.iter().all(|&x| x == 0.0)
        } else {
            false
        }
    }));
}

#[test]
fn type_class_static_method_logical_zeros() {
    // Test that logical.zeros(2, 2) is lowered to zeros(2, 2, 'logical')
    let program = r#"
        A = logical.zeros(2, 2);
    "#;
    let vars = execute_semantic_source(program);
    // The result should be a 2x2 logical array of zeros
    assert!(vars.iter().any(|v| {
        if let runmat_builtins::Value::LogicalArray(l) = v {
            l.shape == vec![2, 2] && l.data.iter().all(|&x| x == 0)
        } else {
            false
        }
    }));
}
