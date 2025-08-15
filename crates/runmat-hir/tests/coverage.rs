use runmat_hir::{lower, HirExprKind, HirProgram, HirStmt, Type};
use runmat_parser::parse_simple as parse;

fn lower_src(src: &str) -> HirProgram {
    let ast = parse(src).unwrap();
    lower(&ast).unwrap()
}

#[test]
fn expr_and_type_inference_arithmetic_and_comparisons() {
    let prog = lower_src("1 + 2 * 3; 4 .^ 2; 5 < 6; 7 && 8");
    assert_eq!(prog.body.len(), 4);
    // 1 + 2*3
    match &prog.body[0] {
        HirStmt::ExprStmt(expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Binary(_, _, _)));
            assert_eq!(expr.ty, Type::Num);
        }
        _ => panic!("expected expr stmt"),
    }
    // 4 .^ 2
    match &prog.body[1] {
        HirStmt::ExprStmt(expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Binary(_, _, _)));
            assert_eq!(expr.ty, Type::Num);
        }
        _ => panic!("expected expr stmt"),
    }
    // 5 < 6
    match &prog.body[2] {
        HirStmt::ExprStmt(expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Binary(_, _, _)));
            assert_eq!(expr.ty, Type::Bool);
        }
        _ => panic!("expected comparison"),
    }
    // 7 && 8
    match &prog.body[3] {
        HirStmt::ExprStmt(expr, false) => {
            assert!(matches!(expr.kind, HirExprKind::Binary(_, _, _)));
            assert_eq!(expr.ty, Type::Bool);
        }
        _ => panic!("expected logical"),
    }
}

#[test]
fn matrices_cells_and_indexing() {
    let prog = lower_src("A = [1,2;3,4]; C = {A, 2}; A(1,2); C{1}");
    assert_eq!(prog.body.len(), 4);
    match &prog.body[0] {
        HirStmt::Assign(_, expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Tensor(_)));
            assert!(matches!(expr.ty, Type::Tensor { .. }));
        }
        _ => panic!("expected matrix assignment"),
    }
    match &prog.body[1] {
        HirStmt::Assign(_, expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Cell(_)));
        }
        _ => panic!("expected cell assignment"),
    }
    match &prog.body[2] {
        HirStmt::ExprStmt(expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Index(_, _)));
        }
        _ => panic!("expected A(,) indexing"),
    }
    match &prog.body[3] {
        HirStmt::ExprStmt(expr, false) => {
            assert!(matches!(expr.kind, HirExprKind::IndexCell(_, _)));
        }
        _ => panic!("expected C indexing"),
    }
}

#[test]
fn end_colon_and_range() {
    let prog = lower_src("A = [1,2,3,4]; A(2:end); 1:2:5; :");
    assert_eq!(prog.body.len(), 4);
    match &prog.body[1] {
        HirStmt::ExprStmt(expr, true) => {
            // A(2:end)
            assert!(matches!(expr.kind, HirExprKind::Index(_, _)));
        }
        _ => panic!("expected end indexing"),
    }
    match &prog.body[2] {
        HirStmt::ExprStmt(expr, true) => {
            assert!(matches!(expr.kind, HirExprKind::Range(_, _, _)));
            assert!(matches!(expr.ty, Type::Tensor{..}));
        }
        _ => panic!("expected range"),
    }
    match &prog.body[3] {
        HirStmt::ExprStmt(expr, false) => {
            assert!(matches!(expr.kind, HirExprKind::Colon));
        }
        _ => panic!("expected colon"),
    }
}

#[test]
fn control_flow_if_while_for_switch_trycatch() {
    let src = r#"
x = 1; y = 0; z = 0;
if x
  y=1;
elseif x>0
  y=2;
else
  y=3;
end
while x
  y=4; x = 0;
end
for i=1:3
  y=i;
end
switch y
  case 1
    z=1;
  otherwise
    z=0;
end
try
  a=1;
catch e
  b=2;
end
"#;
    let prog = lower_src(src);
    // Just ensure all forms are present
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::If{..})));
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::While{..})));
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::For{..})));
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::Switch{..})));
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::TryCatch{..})));
}

#[test]
fn globals_persistents_and_multiassign() {
    let prog = lower_src("global a,b; persistent p; [x,y]=deal(1,2)");
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::Global(_))));
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::Persistent(_))));
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::MultiAssign(_, _, _))));
}

#[test]
fn functions_and_inference() {
    let src = r#"
function y=f(x)
  if x > 0
    y = 1;
  else
    y = 2;
  end
end
z = f(3)
"#;
    let prog = lower_src(src);
    // find function and call
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::Function{..})));
    // Call lowered with inferred numeric type
    let call_assign = prog.body.iter().find(|s| matches!(s, HirStmt::Assign(_, _, _))).unwrap();
    if let HirStmt::Assign(_, expr, _) = call_assign {
        // FuncCall return type should be numeric (Num) based on analysis
        assert_eq!(expr.ty, Type::Num);
    }
}

#[test]
fn methods_members_handles_and_anon() {
    let prog = lower_src("obj = 1; obj.method(1); obj.field; @sin; @(x) x+1");
    assert_eq!(prog.body.len(), 5);
    match &prog.body[1] { HirStmt::ExprStmt(expr, true) => assert!(matches!(expr.kind, HirExprKind::MethodCall(_, _, _))), _ => panic!() }
    match &prog.body[2] { HirStmt::ExprStmt(expr, true) => assert!(matches!(expr.kind, HirExprKind::Member(_, _))), _ => panic!() }
    match &prog.body[3] { HirStmt::ExprStmt(expr, true) => assert!(matches!(expr.kind, HirExprKind::FuncHandle(_))), _ => panic!() }
    match &prog.body[4] { HirStmt::ExprStmt(expr, false) => assert!(matches!(expr.kind, HirExprKind::AnonFunc{..})), _ => panic!() }
}

#[test]
fn classdef_lowering() {
    let src = r#"
classdef C < handle
  properties
    a, b
  end
  methods
    function z = f(x)
      z = x;
    end
  end
  events
    E1
  end
  enumeration
    Red
  end
  arguments
    x, y
  end
end
"#;
    let prog = lower_src(src);
    assert!(prog.body.iter().any(|s| matches!(s, HirStmt::ClassDef { .. })));
}


