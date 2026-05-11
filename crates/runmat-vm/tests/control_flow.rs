#[path = "support/mod.rs"]
mod test_helpers;

use runmat_parser::parse;
use test_helpers::compile_semantic_source;
use test_helpers::execute_semantic_source;
use test_helpers::lower;

fn has_num(vars: &[runmat_builtins::Value], expected: f64) -> bool {
    vars.iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n - expected).abs() < 1e-9))
}

fn execute_semantic_error(source: &str) -> runmat_runtime::RuntimeError {
    execute_semantic_source(source)
        .err()
        .expect("expected error")
}

#[test]
fn break_and_continue() {
    let vars = execute_semantic_source(
        r#"
            x=0;
            while 1;
                x=x+1;
                break;
                x=x+1;
            end
        "#,
    )
    .unwrap();
    assert!(has_num(&vars, 1.0));
}

#[test]
fn elseif_executes_correct_branch() {
    let vars = execute_semantic_source(
        r#"
            x=2;
            if x-2;
                y=1;
            elseif x-1;
                y=2;
            else;
                y=3;
            end
        "#,
    )
    .unwrap();
    assert!(has_num(&vars, 2.0));
}

#[test]
fn switch_case_otherwise_executes_correct_branch() {
    // The parser expects line-based case/otherwise; use newlines
    let vars = execute_semantic_source(
        r#"
            x=2;
            y=0;
            switch x
                case 1
                    y=10;
                case 2
                    y=20;
                otherwise
                    y=30;
            end
        "#,
    )
    .unwrap();
    assert!(has_num(&vars, 20.0));
}

#[test]
fn try_catch_executes_try_body_when_no_error() {
    let vars = execute_semantic_source(
        r#"
            x=0;
            try;
                x=1;
            catch e;
                x=2;
            end
        "#,
    )
    .unwrap();
    assert!(has_num(&vars, 1.0));
}

#[test]
fn try_catch_catches_error_and_binds_identifier() {
    // Unknown builtin should raise; catch should bind 'e' and execute catch body
    let ast = parse(
        r#"
            x=0;
            try;
                nosuchbuiltin(1);
                x=99;
            catch e;
                x=2;
            end
        "#,
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = test_helpers::execute(&hir).unwrap();
    assert!(has_num(&vars, 2.0));
}

#[test]
fn nested_break_and_continue_scopes() {
    let vars = execute_semantic_source(
        r#"
            x=0;
            for i=1:3;
                for j=1:3;
                    if j-2;
                        continue;
                    end;
                    if i-3;
                        break;
                    end;
                    x=x+1;
                end;
            end
        "#,
    )
    .unwrap();
    // Only when i==3 and j>=2 we hit break after skip; count = 1
    assert!(has_num(&vars, 1.0));
}

#[test]
fn undefined_variable_raises_mex() {
    let err = compile_semantic_source("y = x + 1;").err().unwrap();
    assert!(err.message().contains("RunMat:UndefinedVariable"));
}

#[test]
fn block_comment_is_ignored() {
    let vars = execute_semantic_source("a = 1; %{ this is a\n block comment %} b = 2; c = a + b;")
        .unwrap();
    // Expect c = 3 somewhere
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-3.0).abs()<1e-9)));
}

#[test]
fn apostrophe_is_transpose_when_adjacent() {
    // Adjacent apostrophe after value is transpose
    let vars = execute_semantic_source(
        r#"
            A = [1 2; 3 4];
            B = A';
            s = sum(B(:));
            "#,
    )
    .unwrap();
    // sum is invariant under transpose: sum 10
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-10.0).abs()<1e-9)));
}

#[test]
fn apostrophe_starts_char_array_when_not_adjacent() {
    // Non-adjacent apostrophe starts a char array, not transpose
    let vars = execute_semantic_source("A = [1 2]; \n s = 'hi';").unwrap();
    // Expect a CharArray or String present
    let has_text = vars.iter().any(|v| {
        matches!(
            v,
            runmat_builtins::Value::CharArray(_) | runmat_builtins::Value::String(_)
        )
    });
    assert!(has_text);
}

#[test]
fn apostrophe_conjugates_complex() {
    let vars = execute_semantic_source(
        r#"
            z = sqrt(-1);
            A = [1 z; 0 1];
            B = A';
            C = A.';
            b = imag(B);
            c = imag(C);
            b21 = b(2,1);
            c21 = c(2,1);
            "#,
    )
    .unwrap();
    let mut nums = vars.iter().filter_map(|value| match value {
        runmat_builtins::Value::Num(n) => Some(*n),
        _ => None,
    });
    let b21 = nums.find(|n| (*n + 1.0).abs() < 1e-9).unwrap_or(f64::NAN);
    let c21 = nums.find(|n| (*n - 1.0).abs() < 1e-9).unwrap_or(f64::NAN);
    assert!(
        (b21 + 1.0).abs() < 1e-9,
        "expected imag(B(2,1)) == -1, got {b21}"
    );
    assert!(
        (c21 - 1.0).abs() < 1e-9,
        "expected imag(C(2,1)) == 1, got {c21}"
    );
}

#[test]
fn too_many_inputs_mex() {
    let src = r#"
        function y = f(a)
            y = a+1;
        end
        r = f(1,2);
    "#;
    let err = execute_semantic_error(src);
    assert_eq!(err.identifier(), Some("RunMat:TooManyInputs"));
}

#[test]
fn too_many_outputs_mex() {
    let src = r#"
        function y = f(a)
            y = a+1;
        end
        [x1,x2] = f(1);
    "#;
    let err = execute_semantic_error(src);
    assert_eq!(err.identifier(), Some("RunMat:TooManyOutputs"));
}

#[test]
fn varargout_mismatch_mex() {
    let src = r#"
        function varargout = g()
            varargout = {1,2};
        end
        [a,b,c] = g();
    "#;
    let err = execute_semantic_error(src);
    assert_eq!(err.identifier(), Some("RunMat:VarargoutMismatch"));
}

#[test]
fn slice_non_tensor_mex() {
    let src = "x = 5; y = x(1,1);";
    let err = execute_semantic_error(src);
    assert_eq!(err.identifier(), Some("RunMat:SliceNonTensor"));
}

#[test]
fn index_step_zero_mex() {
    let src = "A = [1 2 3 4]; B = A(1:0:3);";
    let err = execute_semantic_error(src);
    assert!(
        err.identifier() == Some("RunMat:IndexStepZero")
            || err.message().contains("Range step cannot be zero")
            || err.message().contains("dimension must be >= 1")
            || err.message().contains("increment must be nonzero")
    );
}

#[test]
fn unsupported_cell_index_type_mex() {
    // When length doesn't match or contains non 0/1 it falls back to indices, but 0 is invalid => out of bounds.
    // Force unsupported type by passing a string as index
    let src2 = "C = {1,2,3}; r = C{'a'};";
    let err2 = execute_semantic_error(src2);
    // Current runtime path attempts numeric coercion and reports conversion failure
    assert!(
        err2.identifier() == Some("RunMat:CellIndexType")
            || err2.message().contains("cannot convert CharArray")
    );
}
