use super::*;
use futures::executor::block_on;
use runmat_execution::{
    DistributedObjectId, DistributedValueHandle, ExecutionScopeId, PoolHandle, PoolId,
};
use runmat_types::{
    DistributedOwner, DistributedValueId, DistributionScheme, LabCount, ParallelRegionId,
    ProgramFunctionId, RegionId, StructFact, ValueFact, ValueKindFact,
};
use runmat_value::{CellArray, ObjectInstance, StructArray, StructValue, Tensor};
use std::collections::BTreeMap;

fn run(value: Value) -> bool {
    match block_on(isstruct_builtin(value)).expect("isstruct") {
        Value::Bool(result) => result,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn recognizes_scalar_nonscalar_and_empty_structures_only() {
    let mut scalar = runmat_value::UnboundedStructValueBuilder::new();
    scalar.insert("value", Value::Num(19.0));
    let scalar = scalar.finish();
    assert!(run(Value::Struct(scalar.clone())));
    assert!(run(Value::StructArray(
        StructArray::new(vec![scalar.clone(), scalar], vec![1, 2]).unwrap()
    )));
    assert!(run(Value::StructArray(
        StructArray::empty(vec!["value".into()], vec![0, 2]).unwrap()
    )));
    assert!(!run(Value::Cell(
        CellArray::new(vec![Value::Struct(StructValue::new())], 1, 1).unwrap()
    )));
    assert!(!run(Value::Object(ObjectInstance::new("TestClass"))));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap()
    )));
}

#[test]
fn resident_and_distributed_classification_needs_no_payload() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        assert!(!run(Value::GpuTensor(handle)));
    });
    let scope_id = ExecutionScopeId::derive(&[b"isstruct"]);
    let function = ProgramFunctionId(7);
    let owner = ParallelRegionId(RegionId {
        function,
        ordinal: 2,
    });
    let value = Value::Distributed(
        Box::new(DistributedValueHandle {
            id: DistributedObjectId::derive(&[b"isstruct-value"]),
            contract: DistributedValueId {
                function,
                ordinal: 1,
            },
            owner: DistributedOwner::Region(owner),
            scope_id,
            generation: 1,
            pool: PoolHandle {
                id: PoolId::derive(&[b"isstruct-pool"]),
                scope_id,
                generation: 1,
            },
            partition_count: LabCount(2),
            value: ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                BTreeMap::new(),
                true,
            ))),
            global_shape: vec![2, 2],
            scheme: DistributionScheme::Replicated,
            materializable: true,
        })
        .into(),
    );
    assert!(run(value));
}

#[test]
fn rejects_more_than_one_output() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(isstruct_builtin(Value::Struct(StructValue::new()))).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:isstruct:TooManyOutputs"));
}
