#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

fn logical_truth(value: &runmat_builtins::Value) -> bool {
    match value {
        runmat_builtins::Value::Bool(value) => *value,
        runmat_builtins::Value::Num(value) => *value != 0.0,
        other => panic!("expected logical value, got {other:?}"),
    }
}

#[test]
fn logical_operators_and_short_circuit() {
    let vars =
        execute_source("a = 0 && (1/0); b = 1 || (1/0); c = 0 & 5; d = 0 | 5; e = ~0; f = ~5;")
            .unwrap();
    assert!(!logical_truth(&vars[0]));
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(logical_truth(&vars[3]));
    assert!(logical_truth(&vars[4]));
    assert!(!logical_truth(&vars[5]));
}

#[test]
fn short_circuit_or_accepts_boolean_lhs_without_numeric_coercion() {
    let vars = execute_source(
        "tau = []; flight_duration = 10; guard = isempty(tau) || tau(end) < flight_duration;",
    )
    .unwrap();
    assert!(logical_truth(&vars[2]));
}
