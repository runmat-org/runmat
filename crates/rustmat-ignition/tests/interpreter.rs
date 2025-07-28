use rustmat_hir::lower;
use rustmat_ignition::execute;
use rustmat_parser::parse;

#[test]
fn arithmetic_and_assignment() {
    let ast = parse("x=1+2; y=x*3;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert!(vars.len() >= 2);
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

#[test]
fn elseif_executes_correct_branch() {
    let ast = parse("x=0; if 0; x=1; elseif 1; x=2; else; x=3; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 2.0);
}

#[test]
fn else_branch_when_all_false() {
    let ast = parse("x=0; if 0; x=1; elseif 0; x=2; else; x=3; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 3.0);
}
#[test]
fn break_outside_loop_errors() {
    let ast = parse("break").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_err());
}

#[test]
fn continue_outside_loop_errors() {
    let ast = parse("continue").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_err());
}

#[test]
fn nested_loops_break_only_inner() {
    let src = "x=0; while 1; while 1; break; end; x=1; break; end";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 1.0);
}

#[test]
fn nested_loops_continue_only_inner() {
    let src = "x=0; for i=1:2; for j=1:2; continue; x=x+1; end; end";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 0.0);
}

#[test]
fn multiple_elseif_branches() {
    let src = "x=0; if 0; x=1; elseif 0; x=2; elseif 1; x=3; else; x=4; end";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 3.0);
}

#[test]
fn return_statement_halts_execution() {
    let src = "x=1; return; x=2";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    assert_eq!(vars[0].as_num(), 1.0);
}

#[test]
fn colon_operator_errors() {
    let ast = parse("x=1:3").unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_err());
}

#[test]
fn function_definition_errors() {
    let src = "function y=add(x); y=x+1; end";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    assert!(execute(&hir).is_err());
}
