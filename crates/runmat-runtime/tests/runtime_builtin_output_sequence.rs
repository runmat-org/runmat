//! The runtime macro must publish the canonical sequence ABI, not a transient
//! language-value encoding of multiple outputs.

use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogIdentity};
use runmat_macros::runtime_builtin;
use runmat_runtime::builtin::{runtime_builtin_bindings_by_name, RuntimeBuiltinInvocation};
use runmat_runtime::execution::RootedValueSequence;
pub use runmat_runtime::RuntimeError;

// The attribute expansion refers to the receiving crate's execution owner.
pub mod execution {
    pub use runmat_runtime::execution::{RootedValueSequence, RootedValues};
}

pub mod context {
    pub use runmat_runtime::context::*;
}
use runmat_value::{Value, ValueSequence, ValueSequenceKind};

// Attribute expansion is relative to the crate receiving the macro. These
// reexports make this integration test exercise the real Runtime binding and
// legacy-adaptation owners rather than a hand-written stand-in ABI.
pub mod builtin {
    pub use runmat_runtime::builtin::*;
}

pub mod call {
    pub mod arguments {
        pub use runmat_runtime::call::arguments::adapt_legacy_builtin_result;
    }
}

pub type BuiltinResult<T> = Result<T, runmat_runtime::RuntimeError>;

const SEQUENCE_NAME: &str = "__runtime_builtin_output_sequence_fixture";
const LEGACY_NAME: &str = "__runtime_builtin_legacy_sequence_fixture";

#[runtime_builtin(
    name = "__runtime_builtin_output_sequence_fixture",
    binding_variant = "default",
    requested_outputs = true,
    output_sequence = true,
    builtin_path = "tests::runtime_builtin_output_sequence::sequence_fixture"
)]
async fn sequence_fixture(requested_outputs: Option<usize>) -> BuiltinResult<ValueSequence> {
    match requested_outputs {
        None => ValueSequence::single(Value::Num(7.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
        Some(0) => Ok(ValueSequence::empty()),
        Some(1) => ValueSequence::comma_separated(vec![Value::Num(11.0)])
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
        Some(count) => ValueSequence::comma_separated(
            (0..count).map(|index| Value::Num(index as f64)).collect(),
        )
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime),
    }
}

#[runtime_builtin(
    name = "__runtime_builtin_legacy_sequence_fixture",
    binding_variant = "default",
    requested_outputs = true,
    builtin_path = "tests::runtime_builtin_output_sequence::legacy_fixture"
)]
async fn legacy_fixture(requested_outputs: Option<usize>) -> BuiltinResult<Value> {
    Ok(match requested_outputs {
        Some(0) => Value::output_list(Vec::new()),
        Some(2) => Value::output_list(vec![Value::Num(2.0), Value::Num(3.0)]),
        _ => Value::Num(13.0),
    })
}

fn binding(name: &'static str) -> runmat_runtime::builtin::RuntimeBuiltinBinding {
    let bindings = runtime_builtin_bindings_by_name(name);
    let [binding] = bindings.as_slice() else {
        panic!("{name} must publish exactly one canonical binding: {bindings:?}");
    };
    assert_eq!(
        binding.identity,
        BuiltinBindingIdentity {
            builtin: BuiltinCatalogIdentity { name },
            variant: "default",
        }
    );
    *binding
}

fn invoke(name: &'static str, requested_outputs: Option<usize>) -> RootedValueSequence {
    let binding = binding(name);
    futures::executor::block_on((binding.implementation)(RuntimeBuiltinInvocation::new(
        &[],
        None,
        requested_outputs,
    )))
    .expect("fixture binding succeeds")
}

#[test]
fn output_sequence_handler_preserves_default_zero_one_and_many_carriers() {
    let cases = [
        (None, ValueSequenceKind::Single, vec![Value::Num(7.0)]),
        (Some(0), ValueSequenceKind::CommaSeparated, vec![]),
        (
            Some(1),
            ValueSequenceKind::CommaSeparated,
            vec![Value::Num(11.0)],
        ),
        (
            Some(3),
            ValueSequenceKind::CommaSeparated,
            vec![Value::Num(0.0), Value::Num(1.0), Value::Num(2.0)],
        ),
    ];
    for (requested_outputs, expected_kind, expected_values) in cases {
        let sequence = invoke(SEQUENCE_NAME, requested_outputs);
        assert_eq!(sequence.kind(), expected_kind);
        assert_eq!(sequence.as_slice(), expected_values);
    }
}

#[test]
fn default_canonical_handler_adapts_legacy_value_carriers_exactly_once() {
    let scalar = invoke(LEGACY_NAME, None);
    assert_eq!(scalar.kind(), ValueSequenceKind::Single);
    assert_eq!(scalar.as_slice(), [Value::Num(13.0)]);

    let empty = invoke(LEGACY_NAME, Some(0));
    assert_eq!(empty.kind(), ValueSequenceKind::CommaSeparated);
    assert!(empty.is_empty());

    let many = invoke(LEGACY_NAME, Some(2));
    assert_eq!(many.kind(), ValueSequenceKind::CommaSeparated);
    assert_eq!(many.as_slice(), [Value::Num(2.0), Value::Num(3.0)]);
    assert!(many
        .as_slice()
        .iter()
        .all(|value| !matches!(value, Value::OutputList(_))));
}
