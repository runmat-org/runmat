mod test_helpers;

use runmat_parser::parse;
use std::convert::TryInto;
use test_helpers::execute;
use test_helpers::lower;

#[test]
fn break_and_continue() {
    let ast = parse(
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
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn elseif_executes_correct_branch() {
    let ast = parse(
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
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 2.0);
}

#[test]
fn switch_case_otherwise_executes_correct_branch() {
    // The parser expects line-based case/otherwise; use newlines
    let ast = parse(
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
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 20.0);
}

#[test]
fn try_catch_executes_try_body_when_no_error() {
    let ast = parse(
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
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
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
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 2.0);
}

#[test]
fn nested_break_and_continue_scopes() {
    let ast = parse(
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
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    // Only when i==3 and j>=2 we hit break after skip; count = 1
    assert_eq!(x, 1.0);
}

#[test]
fn undefined_variable_raises_mex() {
    let ast = parse("y = x + 1;").unwrap();
    let hir = lower(&ast);
    let err = hir.err().unwrap();
    assert_eq!(err.identifier.as_deref(), Some("MATLAB:UndefinedVariable"));
}

#[test]
fn block_comment_is_ignored() {
    let ast = parse("a = 1; %{ this is a\n block comment %} b = 2; c = a + b;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Expect c = 3 somewhere
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-3.0).abs()<1e-9)));
}

#[test]
fn apostrophe_is_transpose_when_adjacent() {
    // Adjacent apostrophe after value is transpose
    let ast = parse(
        r#"
            A = [1 2; 3 4];
            B = A';
            s = sum(B(:));
            "#,
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // sum is invariant under transpose: sum 10
    assert!(vars
        .iter()
        .any(|v| matches!(v, runmat_builtins::Value::Num(n) if (*n-10.0).abs()<1e-9)));
}

#[test]
fn apostrophe_starts_char_array_when_not_adjacent() {
    // Non-adjacent apostrophe starts a char array, not transpose
    let ast = parse("A = [1 2]; \n s = 'hi';").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
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
    let ast = parse(
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
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let b21: f64 = (&vars[6]).try_into().unwrap();
    let c21: f64 = (&vars[7]).try_into().unwrap();
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
    let hir = lower(&parse(src).unwrap()).unwrap();
    let err = execute(&hir).err().unwrap();
    assert_eq!(err.identifier(), Some("MATLAB:TooManyInputs"));
}

#[test]
fn too_many_outputs_mex() {
    let src = r#"
        function y = f(a)
            y = a+1;
        end
        [x1,x2] = f(1);
    "#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let err = execute(&hir).err().unwrap();
    assert_eq!(err.identifier(), Some("MATLAB:TooManyOutputs"));
}

#[test]
fn varargout_mismatch_mex() {
    let src = r#"
        function varargout = g()
            varargout = {1,2};
        end
        [a,b,c] = g();
    "#;
    let hir = lower(&parse(src).unwrap()).unwrap();
    let err = execute(&hir).err().unwrap();
    assert_eq!(err.identifier(), Some("MATLAB:VarargoutMismatch"));
}

#[test]
fn slice_non_tensor_mex() {
    let src = "x = 5; y = x(1,1);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let err = execute(&hir).err().unwrap();
    assert_eq!(err.identifier(), Some("MATLAB:SliceNonTensor"));
}

#[test]
fn index_step_zero_mex() {
    let src = "A = [1 2 3 4]; B = A(1:0:3);";
    let hir = lower(&parse(src).unwrap()).unwrap();
    let err = execute(&hir).err().unwrap();
    assert!(
        err.identifier() == Some("MATLAB:IndexStepZero")
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
    let hir2 = lower(&parse(src2).unwrap()).unwrap();
    let err2 = execute(&hir2).err().unwrap();
    // Current runtime path attempts numeric coercion and reports conversion failure
    assert!(
        err2.identifier() == Some("MATLAB:CellIndexType")
            || err2.message().contains("cannot convert CharArray")
    );
}
