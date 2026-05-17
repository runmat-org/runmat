#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::Value;
use runmat_vm::Instr;
use test_helpers::compile_semantic_source;
use test_helpers::execute_semantic_source;

#[test]
fn colon_range_produces_row_vector() {
    let vars = execute_semantic_source("v = -2:0.08:2;").unwrap();
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
    let vars = execute_semantic_source("[X, Y] = meshgrid(-2:0.08:2, -2:0.08:2);").unwrap();
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
    let source = "a = -2:0.08:2; b = -2:0.08:2; [X, Y] = meshgrid(a, b);";
    let bytecode = compile_semantic_source(source).unwrap();
    assert!(bytecode.instructions.iter().any(|instr| {
        matches!(instr, Instr::CallBuiltinMulti(name, 2, 1) if name == "meshgrid")
            || matches!(instr, Instr::CallBuiltinMulti(name, 2, 2) if name == "meshgrid")
            || matches!(instr, Instr::CallBuiltinExpandMultiOutput(name, specs, out_count)
                if name == "meshgrid" && *out_count == 1 && specs.len() == 2)
    }));
    let vars = execute_semantic_source(source).unwrap();
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
    let vars = execute_semantic_source("a = -2:0.08:2; b = -2:0.08:2;").unwrap();
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
    let vars = execute_semantic_source(
        "a = -2:0.08:2; b = -2:0.08:2; try; [X, Y] = meshgrid(a, b); catch e; end; A = a;",
    )
    .unwrap();
    assert!(
        vars.iter()
            .any(|value| matches!(value, Value::Tensor(t) if t.shape == vec![1, 51])),
        "expected a retained input row vector, got {vars:?}"
    );
}
