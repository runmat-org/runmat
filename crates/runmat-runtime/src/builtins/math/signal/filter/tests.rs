//! Filter descriptor and shared test helpers.
use super::*;
use futures::executor::block_on;
use runmat_builtins::builtin_function_by_name;
use runmat_value::{IntegerStorage, Tensor};

fn error_message(error: RuntimeError) -> String {
    error.message().to_string()
}

fn evaluate(b: Value, a: Value, x: Value, rest: &[Value]) -> BuiltinResult<FilterEvaluation> {
    block_on(super::evaluate(b, a, x, rest))
}

fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(storage, shape).expect("typed integer tensor")
}

fn real_tensor_parts(value: Value, context: &str) -> (Vec<f64>, Vec<usize>) {
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected {context} tensor, got {other:?}"),
    };
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .unwrap_or_else(|err| panic!("invalid {context} tensor storage: {err}"));
    let data = storage
        .as_f64_slice()
        .unwrap_or_else(|| panic!("expected {context} tensor to use double storage"))
        .to_vec();
    (data, shape)
}

#[test]
fn filter_descriptor_signatures_and_errors() {
    assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
    let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
        .expect("canonical filter builtin");
    let descriptor = builtin.descriptor;
    assert_eq!(builtin.bindings, &runmat_builtins::REQUIRED_DEFAULT_BINDING);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        builtin.binding_identity(&builtin.bindings[0])
    );
    let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
    assert!(labels.contains(&"y = filter(b, a, x)"));
    assert!(labels.contains(&"y = filter(b, a, x, zi)"));
    assert!(labels.contains(&"y = filter(b, a, x, zi, dim)"));
    assert!(labels.contains(&"[y, zf] = filter(b, a, x)"));
    assert!(labels.contains(&"[y, zf] = filter(b, a, x, zi)"));
    assert!(labels.contains(&"[y, zf] = filter(b, a, x, zi, dim)"));
    assert!(descriptor
        .errors
        .iter()
        .any(|err| err.code == "RM.FILTER.ARG_COUNT"));
    assert_eq!(builtin.integer_capabilities.len(), 2);
}

fn approx_eq_slice(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "length mismatch ({} vs {})",
        lhs.len(),
        rhs.len()
    );
    let tol = match runmat_accelerate_api::provider()
        .map(|p| p.precision())
        .unwrap_or(runmat_accelerate_api::ProviderPrecision::F64)
    {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-6,
    };
    for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < tol,
            "value mismatch at index {idx}: {a} vs {b} (diff {diff})"
        );
    }
}

fn approx_eq_complex(lhs: &[(f64, f64)], rhs: &[(f64, f64)]) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, ((ar, ai), (br, bi))) in lhs.iter().zip(rhs.iter()).enumerate() {
        let dr = (ar - br).abs();
        let di = (ai - bi).abs();
        assert!(
            dr < 1e-9 && di < 1e-9,
            "complex mismatch at {idx}: ({ar},{ai}) vs ({br},{bi})"
        );
    }
}

#[path = "tests/host.rs"]
mod host;
#[path = "tests/provider.rs"]
mod provider;
