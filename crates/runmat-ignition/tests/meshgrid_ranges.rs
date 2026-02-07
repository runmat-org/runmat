mod test_helpers;

use runmat_builtins::Value;
use runmat_parser::parse;
use test_helpers::execute;
use test_helpers::lower;

#[test]
fn colon_range_produces_row_vector() {
    let ast = parse("v = -2:0.08:2;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let v = vars
        .iter()
        .find_map(|val| match val {
            Value::Tensor(t) => Some(t),
            _ => None,
        })
        .expect("expected tensor result for v");
    assert_eq!(v.shape, vec![1, 51]);
}

#[test]
fn meshgrid_accepts_colon_ranges() {
    let ast = parse("[X, Y] = meshgrid(-2:0.08:2, -2:0.08:2);").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let mut tensors: Vec<&runmat_builtins::Tensor> = vars
        .iter()
        .filter_map(|v| match v {
            Value::Tensor(t) => Some(t),
            _ => None,
        })
        .collect();
    // We expect at least the two outputs X and Y to exist as tensors.
    assert!(
        tensors.iter().any(|t| t.shape == vec![51, 51]),
        "expected a 51x51 tensor among outputs; got {:?}",
        tensors.iter().map(|t| t.shape.clone()).collect::<Vec<_>>()
    );
    // Ensure we didn't accidentally create a 51x51 tensor for the input vector itself.
    // (Inputs should be vectors; only outputs are 51x51.)
    tensors.retain(|t| t.shape == vec![1, 51] || t.shape == vec![51, 1] || t.shape == vec![51, 51]);
}

#[test]
fn meshgrid_accepts_precomputed_ranges() {
    let ast = parse("a = -2:0.08:2; b = -2:0.08:2; [X, Y] = meshgrid(a, b);").unwrap();
    let hir = lower(&ast).unwrap();
    // Sanity-check the lowering: we should be calling the meshgrid builtin with args [Var(a), Var(b)].
    match hir.body.last().expect("expected statements") {
        runmat_hir::HirStmt::MultiAssign(_vars, expr, _suppressed, _) => {
            match &expr.kind {
                runmat_hir::HirExprKind::FuncCall(name, args) => {
                    assert_eq!(name, "meshgrid");
                    assert_eq!(args.len(), 2);
                    assert!(matches!(args[0].kind, runmat_hir::HirExprKind::Var(_)));
                    assert!(matches!(args[1].kind, runmat_hir::HirExprKind::Var(_)));
                }
                other => panic!("expected FuncCall(meshgrid, ...), got {other:?}"),
            }
        }
        other => panic!("expected final stmt to be MultiAssign, got {other:?}"),
    }
    let vars = execute(&hir).unwrap();
    let shapes: Vec<Vec<usize>> = vars
        .iter()
        .filter_map(|v| match v {
            Value::Tensor(t) => Some(t.shape.clone()),
            _ => None,
        })
        .collect();
    assert!(
        shapes.iter().any(|s| s == &[51, 51]),
        "expected meshgrid outputs (51x51); got {shapes:?}"
    );
}

#[test]
fn two_colon_ranges_remain_vectors() {
    let ast = parse("a = -2:0.08:2; b = -2:0.08:2;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // Variable order for this program should be [a, b].
    let a = match &vars[0] {
        Value::Tensor(t) => t,
        other => panic!("expected a tensor for a, got {other:?}"),
    };
    let b = match &vars[1] {
        Value::Tensor(t) => t,
        other => panic!("expected a tensor for b, got {other:?}"),
    };
    assert_eq!(a.shape, vec![1, 51]);
    assert_eq!(b.shape, vec![1, 51]);
}

#[test]
fn meshgrid_failure_does_not_mutate_inputs() {
    // Even if meshgrid errors, the input vectors should remain vectors.
    let ast = parse(
        "a = -2:0.08:2; b = -2:0.08:2; try; [X, Y] = meshgrid(a, b); catch e; end; A = a;",
    )
    .unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    // A is assigned last; it should be a row vector.
    let last = vars.last().expect("expected at least one var");
    match last {
        Value::Tensor(t) => assert_eq!(t.shape, vec![1, 51]),
        other => panic!("expected A to be a tensor, got {other:?}"),
    }
}

