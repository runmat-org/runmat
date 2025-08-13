use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;
use std::convert::TryInto;

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



