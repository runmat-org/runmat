use super::*;
use futures::executor::block_on;
use runmat_execution::{
    DistributedObjectId, DistributedValueHandle, ExecutionScopeId, PoolHandle, PoolId,
};
use runmat_types::{
    DistributedOwner, DistributedValueId, DistributionScheme, LabCount, NumericClass,
    NumericDomain, NumericFact, ParallelRegionId, ProgramFunctionId, RegionId, ValueFact,
    ValueKindFact,
};

fn distributed_value(global_shape: Vec<u64>) -> Value {
    let scope_id = ExecutionScopeId::derive(&[b"shape-scalar-query"]);
    let function = ProgramFunctionId(9);
    let owner = ParallelRegionId(RegionId {
        function,
        ordinal: 1,
    });
    Value::Distributed(Box::new(DistributedValueHandle {
        id: DistributedObjectId::derive(&[b"shape-scalar-value"]),
        contract: DistributedValueId {
            function,
            ordinal: 2,
        },
        owner: DistributedOwner::Region(owner),
        scope_id,
        generation: 1,
        pool: PoolHandle {
            id: PoolId::derive(&[b"shape-scalar-pool"]),
            scope_id,
            generation: 1,
        },
        partition_count: LabCount(2),
        value: ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        global_shape,
        scheme: DistributionScheme::Replicated,
        materializable: true,
    }))
}

fn boundary(query: ShapeScalarQuery) -> ShapeScalarQueryBoundary {
    match query {
        ShapeScalarQuery::Length => ShapeScalarQueryBoundary::new(
            &runmat_builtins::LENGTH_CATALOG_ENTRY,
            &runmat_builtins::LENGTH_ERROR_INTERNAL,
            &runmat_builtins::LENGTH_ERROR_TOO_MANY_OUTPUTS,
            Some(&runmat_builtins::LENGTH_ERROR_RESULT_NOT_EXACT),
            Some(&runmat_builtins::LENGTH_ERROR_UNSUPPORTED_TABLE),
            query,
        ),
        ShapeScalarQuery::Rank => ShapeScalarQueryBoundary::new(
            &runmat_builtins::NDIMS_CATALOG_ENTRY,
            &runmat_builtins::NDIMS_ERROR_INTERNAL,
            &runmat_builtins::NDIMS_ERROR_TOO_MANY_OUTPUTS,
            None,
            None,
            query,
        ),
        ShapeScalarQuery::Height => ShapeScalarQueryBoundary::new(
            &runmat_builtins::HEIGHT_CATALOG_ENTRY,
            &runmat_builtins::HEIGHT_ERROR_INTERNAL,
            &runmat_builtins::HEIGHT_ERROR_TOO_MANY_OUTPUTS,
            Some(&runmat_builtins::HEIGHT_ERROR_RESULT_NOT_EXACT),
            None,
            query,
        ),
        ShapeScalarQuery::Width => ShapeScalarQueryBoundary::new(
            &runmat_builtins::WIDTH_CATALOG_ENTRY,
            &runmat_builtins::WIDTH_ERROR_INTERNAL,
            &runmat_builtins::WIDTH_ERROR_TOO_MANY_OUTPUTS,
            Some(&runmat_builtins::WIDTH_ERROR_RESULT_NOT_EXACT),
            None,
            query,
        ),
    }
}

#[test]
fn distributed_queries_use_validated_global_shape() {
    assert_eq!(
        block_on(boundary(ShapeScalarQuery::Length).execute(distributed_value(vec![8, 13])))
            .unwrap(),
        Value::Num(13.0)
    );
    assert_eq!(
        block_on(boundary(ShapeScalarQuery::Rank).execute(distributed_value(vec![8, 13, 4, 1])))
            .unwrap(),
        Value::Num(3.0)
    );
    assert_eq!(
        block_on(boundary(ShapeScalarQuery::Height).execute(distributed_value(vec![8, 13, 4])))
            .unwrap(),
        Value::Num(8.0)
    );
    assert_eq!(
        block_on(boundary(ShapeScalarQuery::Width).execute(distributed_value(vec![8, 13, 4])))
            .unwrap(),
        Value::Num(13.0)
    );
}

#[test]
fn length_rejects_an_inexact_distributed_extent() {
    let error = block_on(
        boundary(ShapeScalarQuery::Length).execute(distributed_value(vec![(1_u64 << 53) + 1, 2])),
    )
    .expect_err("inexact double");
    assert_eq!(
        error.identifier(),
        Some("RunMat:length:ResultNotExactDouble")
    );
}
