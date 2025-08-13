use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;
use std::convert::TryInto;

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


