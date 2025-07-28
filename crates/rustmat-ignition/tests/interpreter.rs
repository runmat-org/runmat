use rustmat_hir::lower;
use rustmat_ignition::execute;
use rustmat_parser::parse;

#[test]
fn arithmetic_and_assignment() {
    let ast = parse("x=1+2; y=x*3;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars.len() >= 2, true);
    assert_eq!(vars[0].as_num(), 3.0);
    assert_eq!(vars[1].as_num(), 9.0);
}

#[test]
fn while_loop_decrements() {
    let ast = parse("x=3; y=0; while x; y=y+1; x=x-1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 0.0); // x
    assert_eq!(vars[1].as_num(), 3.0); // y
}

#[test]
fn for_loop_sum() {
    let ast = parse("s=0; for i=1:4; s=s+i; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 10.0);
}

#[test]
fn continue_in_loop() {
    let ast = parse("x=0; for i=1:3; if i-2; x=x+i; else; continue; end; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 4.0);
}

#[test]
fn break_in_loop() {
    let ast = parse("x=0; while 1; x=x+1; break; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 1.0);
}

#[test]
fn range_with_step_errors() {
    let ast = parse("for i=1:2:3; end").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_err());
}

#[test]
fn unsupported_matrix_errors() {
    let ast = parse("x=[1,2]").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_err());
}

#[test]
fn for_loop_start_greater_than_end() {
    let ast = parse("x=1; for i=3:1; x=2; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 1.0); // x unchanged
}
