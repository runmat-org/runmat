use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;
use std::convert::TryInto;

#[test]
fn break_and_continue() {
    let ast = parse("x=0; while 1; x=x+1; break; x=x+1; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn elseif_executes_correct_branch() {
    let ast = parse("x=2; if x-2; y=1; elseif x-1; y=2; else; y=3; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 2.0);
}

#[test]
fn switch_case_otherwise_executes_correct_branch() {
    // The parser expects line-based case/otherwise; use newlines
    let ast = parse("x=2; y=0; switch x\n case 1\n y=10;\n case 2\n y=20;\n otherwise\n y=30;\n end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let y: f64 = (&vars[1]).try_into().unwrap();
    assert_eq!(y, 20.0);
}

#[test]
fn try_catch_executes_try_body_when_no_error() {
    let ast = parse("x=0; try; x=1; catch e; x=2; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 1.0);
}

#[test]
fn try_catch_catches_error_and_binds_identifier() {
    // Unknown builtin should raise; catch should bind 'e' and execute catch body
    let ast = parse("x=0; try; nosuchbuiltin(1); x=99; catch e; x=2; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    assert_eq!(x, 2.0);
}

#[test]
fn nested_break_and_continue_scopes() {
    let ast = parse("x=0; for i=1:3; for j=1:3; if j-2; continue; end; if i-3; break; end; x=x+1; end; end").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let x: f64 = (&vars[0]).try_into().unwrap();
    // Only when i==3 and j>=2 we hit break after skip; count = 1
    assert_eq!(x, 1.0);
}


