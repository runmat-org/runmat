use super::*;
use futures::executor::block_on;
use runmat_execution::{
    DistributedObjectId, DistributedValueHandle, ExecutionScopeId, PoolHandle, PoolId,
};
use runmat_types::{
    CellFact, DistributedOwner, DistributedValueId, DistributionScheme, DynamicReason, LabCount,
    ParallelRegionId, ProgramFunctionId, RegionId, ValueFact, ValueKindFact,
};
use runmat_value::{CellArray, Tensor};

fn run(value: Value) -> bool {
    match block_on(iscell_builtin(value)).expect("iscell") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn recognizes_cell_container_identity_without_inspecting_contents() {
    let nested = CellArray::new(
        vec![
            Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        ],
        1,
        2,
    )
    .unwrap();
    assert!(run(Value::Cell(nested)));
    assert!(run(Value::Cell(CellArray::new(Vec::new(), 0, 2).unwrap())));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap()
    )));
}

#[test]
fn resident_numeric_values_are_not_cells() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        assert!(!run(Value::GpuTensor(handle)));
    });
}

#[test]
fn recognizes_distributed_cell_identity_without_materialization() {
    let scope_id = ExecutionScopeId::derive(&[b"iscell"]);
    let function = ProgramFunctionId(7);
    let owner = ParallelRegionId(RegionId {
        function,
        ordinal: 2,
    });
    let value = Value::Distributed(Box::new(DistributedValueHandle {
        id: DistributedObjectId::derive(&[b"iscell-value"]),
        contract: DistributedValueId {
            function,
            ordinal: 1,
        },
        owner: DistributedOwner::Region(owner),
        scope_id,
        generation: 1,
        pool: PoolHandle {
            id: PoolId::derive(&[b"iscell-pool"]),
            scope_id,
            generation: 1,
        },
        partition_count: LabCount(2),
        value: ValueFact::scalar(ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        })),
        global_shape: vec![2, 2],
        scheme: DistributionScheme::Replicated,
        materializable: true,
    }));
    assert!(run(value));
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(iscell_builtin(Value::Bool(true))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:iscell:TooManyOutputs"));
}
