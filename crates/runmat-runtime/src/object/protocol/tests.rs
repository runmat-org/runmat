use futures::executor::block_on;
use runmat_types::{ObjectIndexingContext, SequenceUse};
use runmat_value::{StructValue, Tensor, Value};

use super::*;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::sequence::{ResolveValueSequence, SequenceResolutionContext, ValueSequence};

fn single(sequence: ValueSequence) -> Value {
    sequence
        .resolve(
            SequenceUse::RequireSingle,
            SequenceResolutionContext::default(),
        )
        .expect("single sequence")
        .pop()
        .expect("one value")
}

fn read(path: ObjectSubscriptPath, base: Value) -> Value {
    single(
        block_on(read_subscript_path_sequence_with_access(
            base,
            path,
            SubscriptReadRequest {
                sequence_use: SequenceUse::RequireSingle,
                sequence_context: SequenceResolutionContext::default(),
                indexing_context: ObjectIndexingContext::Expression,
                access: ObjectAccessContext::default(),
            },
        ))
        .expect("path read"),
    )
}

#[test]
fn typed_dotted_invoke_distinguishes_callable_member_from_member_indexing() {
    let mut callable = StructValue::new();
    callable.insert("operation", Value::FunctionHandle("sqrt".into()));
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke("operation", vec![Value::Num(9.0)]).to_vec(),
    )
    .expect("dotted invoke path");
    assert_eq!(read(path, Value::Struct(callable)), Value::Num(3.0));

    let mut indexed = StructValue::new();
    indexed.insert(
        "samples",
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("tensor")),
    );
    let path = ObjectSubscriptPath::new(
        ObjectSubscript::dotted_invoke("samples", vec![Value::Num(2.0)]).to_vec(),
    )
    .expect("dotted invoke path");
    assert_eq!(read(path, Value::Struct(indexed)), Value::Num(20.0));
}

#[test]
fn ordinary_adjacent_member_and_parentheses_are_not_method_provenance() {
    let path = ObjectSubscriptPath::new(vec![
        ObjectSubscript::member("samples"),
        ObjectSubscript::parentheses(crate::object::indexing::ObjectIndexSelector::IndexValues {
            components: vec![Value::Num(1.0).into()],
        }),
    ])
    .expect("ordinary path");
    assert!(path
        .steps()
        .iter()
        .all(|step| step.origin() == crate::object::indexing::ObjectSubscriptOrigin::Ordinary));
}
