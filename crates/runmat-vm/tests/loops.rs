#[path = "support/mod.rs"]
mod test_helpers;

use runmat_vm::Instr;
use std::convert::TryInto;
use test_helpers::compile_source;
use test_helpers::interpret;

fn execute_program(source: &str) -> Vec<runmat_value::Value> {
    let bytecode = compile_source(source).expect("compile source");
    interpret(&bytecode).expect("execute bytecode")
}

#[test]
fn while_loop_decrements() {
    let vars = execute_program("x=3; y=0; while x; y=y+1; x=x-1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 0.0);
    assert_eq!(y, 3.0);
}

#[test]
fn for_loop_sum() {
    let vars = execute_program("s=0; for i=1:4; s=s+i; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 10.0);
}

#[test]
fn for_loop_with_step_and_descending() {
    // Ascending with step
    let vars = execute_program("s=0; for i=1:2:5; s=s+i; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 1.0 + 3.0 + 5.0);

    // Descending with negative step
    let vars = execute_program("s=0; for i=5:-2:0; s=s+i; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 5.0 + 3.0 + 1.0);
}

#[test]
fn for_loop_zero_step_terminates() {
    // Step=0 should not loop
    let vars = execute_program("s=0; for i=1:0:10; s=s+1; end");
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 0.0);
}

#[test]
fn while_false_never_enters() {
    let vars = execute_program("x=0; while 0; x=x+1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 0.0);
}

#[test]
fn for_start_equals_end_executes_once() {
    let vars = execute_program("x=0; for i=5:5; x=x+1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn for_negative_step_no_iterations_when_increasing() {
    let vars = execute_program("x=0; for i=1:-1:5; x=x+1; end");
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 0.0);
}

#[test]
fn stochastic_evolution_loop_emits_instruction() {
    let source = "
    M = 32;
    S = ones(M, 1, 'single');
    drift = single(0.1);
    scale = single(0.2);
    for t = 1:8
        Z = randn(M, 1, 'single');
        S = S .* exp(drift + scale .* Z);
    end
    ";
    let bytecode = compile_source(source).unwrap();
    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, Instr::StochasticEvolution)));
}

#[test]
fn stochastic_evolution_loop_preserves_native_single_state() {
    let vars = execute_program(
        "
        M = 2;
        S = ones(M, 1, 'single');
        drift = single(0.1);
        scale = single(0);
        for t = 1:3
            Z = randn(M, 1, 'single');
            S = S .* exp(drift + scale .* Z);
        end
        ",
    );
    let runmat_value::Value::Tensor(state) = &vars[1] else {
        panic!("expected evolved tensor state, got {:?}", vars[1]);
    };
    assert_eq!(state.numeric_dtype(), runmat_value::NumericDType::F32);
    let factor = (0.1f32).exp();
    let expected = factor * factor * factor;
    for index in 0..state.len() {
        assert_eq!(
            state.numeric_value_at(index),
            Some(runmat_value::NumericScalar::F32(expected))
        );
    }
}
