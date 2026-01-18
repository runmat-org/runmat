mod test_helpers;

use runmat_hir::lower;
use runmat_ignition::{compile, instr::Instr};
use runmat_parser::parse;
use std::convert::TryInto;
use test_helpers::execute;

#[test]
fn while_loop_decrements() {
    let ast = parse("x=3; y=0; while x; y=y+1; x=x-1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(x, 0.0);
    assert_eq!(y, 3.0);
}

#[test]
fn for_loop_sum() {
    let ast = parse("s=0; for i=1:4; s=s+i; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 10.0);
}

#[test]
fn for_loop_with_step_and_descending() {
    // Ascending with step
    let ast = parse("s=0; for i=1:2:5; s=s+i; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 1.0 + 3.0 + 5.0);

    // Descending with negative step
    let ast = parse("s=0; for i=5:-2:0; s=s+i; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 5.0 + 3.0 + 1.0);
}

#[test]
fn for_loop_zero_step_terminates() {
    // Step=0 should not loop
    let ast = parse("s=0; for i=1:0:10; s=s+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let s: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(s, 0.0);
}

#[test]
fn while_false_never_enters() {
    let ast = parse("x=0; while 0; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 0.0);
}

#[test]
fn for_start_equals_end_executes_once() {
    let ast = parse("x=0; for i=5:5; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn for_negative_step_no_iterations_when_increasing() {
    let ast = parse("x=0; for i=1:-1:5; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
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
    let ast = parse(source).unwrap();
    let hir = lower(&ast).unwrap();
    let bytecode = compile(&hir).unwrap();
    assert!(bytecode
        .instructions
        .iter()
        .any(|instr| matches!(instr, Instr::StochasticEvolution)));
}
