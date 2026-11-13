use super::*;
use crate::indexing::plan::{build_assignment_plan, build_index_plan};
use crate::indexing::selectors::SliceSelector;
use runmat_value::{StructValue, Tensor};

fn element(value: f64) -> StructValue {
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("value", Value::Num(value));
    structure.finish()
}

fn array(shape: Vec<usize>) -> Value {
    let count = shape.iter().product();
    Value::StructArray(
        StructArray::new(
            (0..count)
                .map(|index| element(index as f64 + 1.0))
                .collect(),
            shape,
        )
        .unwrap(),
    )
}

fn empty_rhs() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap())
}

#[test]
fn aggregate_deletion_context_preserves_geometry_aliases_and_retirement() {
    use crate::context::{
        host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
        RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct Managed(ManagedHostEffectLedger);
    impl RuntimeParallelService for Managed {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let make_runtime = |bytes| {
        let ledger = ManagedHostEffectLedger::new(bytes);
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
                );
        (context, ledger)
    };
    // No ambient activation: every final/scratch/shape owner must use context.
    let (context, ledger) = make_runtime(1 << 20);
    let empty = empty_rhs();
    let repeated = IndexPlan::new(vec![0, 0, 2], vec![1, 3], vec![3], 1, vec![1, 4]);
    let source = array(vec![1, 4]);
    let alias = source.clone();
    let selected = delete_with_plan_in_context(Some(&context), source, &repeated, &empty).unwrap();
    let Value::StructArray(result) = &selected else {
        panic!("structure output")
    };
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(
        result.get_linear(0).unwrap().fields.get("value"),
        Some(&Value::Num(2.))
    );
    assert_eq!(
        result.get_linear(1).unwrap().fields.get("value"),
        Some(&Value::Num(4.))
    );
    assert_eq!(shape(&alias), vec![1, 4]);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(selected);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let source = array(vec![2, 2, 2]);
    let nd = build_index_plan(
        &[
            SliceSelector::Colon,
            SliceSelector::Scalar(1),
            SliceSelector::Colon,
        ],
        3,
        &[2, 2, 2],
    )
    .unwrap();
    let selected = delete_with_plan_in_context(Some(&context), source, &nd, &empty).unwrap();
    let Value::StructArray(result) = &selected else {
        panic!("ND structure output")
    };
    assert_eq!(result.shape(), &[2, 1, 2]);
    for (index, value) in [3., 4., 7., 8.].into_iter().enumerate() {
        assert_eq!(
            result.get_linear(index).unwrap().fields.get("value"),
            Some(&Value::Num(value))
        );
    }
    drop(selected);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let cell = runmat_value::CellArray::new(
        (1..=4).map(|value| Value::Num(f64::from(value))).collect(),
        1,
        4,
    )
    .unwrap();
    let alias = cell.clone();
    let selected = crate::object::cell::delete_cell_with_plan_in_context(
        Some(&context),
        cell,
        &repeated,
        &empty,
    )
    .unwrap();
    let Value::Cell(result) = &selected else {
        panic!("cell output")
    };
    assert_eq!(&result.shape[..], &[1, 2]);
    assert_eq!(result.data(), &[Value::Num(2.), Value::Num(4.)]);
    assert_eq!(alias.data().len(), 4);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(selected);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let object = |tag| {
        let mut value = runmat_value::ObjectInstance::new("parallel.FevalFuture");
        value.replace_property_unbounded("Tag".into(), Value::Num(tag));
        Value::Object(value)
    };
    let objects = runmat_value::ObjectArray::new(
        "parallel.FevalFuture",
        (1..=4).map(|value| object(f64::from(value))).collect(),
        vec![1, 4],
    )
    .unwrap();
    let alias = objects.clone();
    assert!(!alias.class_name_owner().has_retained_claim());
    let physical_class_bytes =
        u64::try_from(alias.class_name_owner().requested_bytes().unwrap()).unwrap();
    let retained_class_bytes =
        crate::context::host_allocation::HostAllocationPlan::retained_exact_output_bytes(
            physical_class_bytes,
            1,
        )
        .unwrap();
    let selected = crate::indexing::object::delete_with_plan_in_context(
        Some(&context),
        Value::ObjectArray(objects),
        &repeated,
        &empty,
    )
    .unwrap();
    let Value::ObjectArray(result) = &selected else {
        panic!("object output")
    };
    assert_eq!(result.shape(), &[1, 2]);
    assert!(result
        .class_name_owner()
        .shares_allocation_with(alias.class_name_owner()));
    assert!(alias.class_name_owner().has_retained_claim());
    for (value, tag) in result.data().iter().zip([2., 4.]) {
        let Value::Object(value) = value else {
            panic!("object child")
        };
        assert_eq!(value.property("Tag"), Some(&Value::Num(tag)));
    }
    assert_eq!(alias.len(), 4);
    drop(selected);
    // The contextual successor promoted the original shared physical class
    // owner. Its physical backing and one exact claim-control allocation must
    // remain reserved and claimed until the unchanged source alias retires.
    assert_eq!(ledger.snapshot().live_claimed_bytes, retained_class_bytes);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained_class_bytes);
    assert_eq!(
        ledger.snapshot().free_bytes(),
        ledger.snapshot().capacity_bytes - retained_class_bytes
    );
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let one = IndexPlan::new(vec![0, 0], vec![1, 2], vec![2], 1, vec![1, 1]);
    let selected = crate::indexing::object::delete_with_plan_in_context(
        Some(&context),
        object(9.),
        &one,
        &empty,
    )
    .unwrap();
    assert!(
        matches!(&selected, Value::ObjectArray(array) if array.is_empty() && array.shape() == [1, 0])
    );
    drop(selected);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let (denied, ledger) = make_runtime(0);
    let source = array(vec![1, 4]);
    let alias = source.clone();
    assert!(delete_with_plan_in_context(Some(&denied), source, &repeated, &empty).is_err());
    assert_eq!(shape(&alias), vec![1, 4]);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    let none = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![1, 4]);
    assert_eq!(
        delete_with_plan_in_context(Some(&denied), alias.clone(), &none, &empty).unwrap(),
        alias
    );
    // Invalid user plans cannot wrap into an apparently valid empty scalar
    // or Cartesian selection, and semantic geometry gates still precede them.
    for dims in [1, 2] {
        let malformed = IndexPlan::new(vec![9], vec![1, 1], vec![1], dims, vec![1, 1]);
        let error = delete_with_plan_in_context(
            Some(&context),
            Value::Struct(element(1.)),
            &malformed,
            &empty,
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));
    }
    let invalid_geometry = IndexPlan::new(vec![99], vec![1, 1], vec![1], 1, vec![2, 2]);
    assert_eq!(
        delete_with_plan_in_context(Some(&context), array(vec![2, 2]), &invalid_geometry, &empty)
            .unwrap_err()
            .identifier(),
        Some("RunMat:UnsupportedStructDeletion")
    );
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let denied = denied.with_cancellation_token(cancellation.token());
    assert!(cancellation.cancel(runmat_execution::CancellationReason::User));
    assert!(denied.is_cancelled());
    let error = delete_with_plan_in_context(Some(&denied), array(vec![1, 4]), &repeated, &empty)
        .unwrap_err();
    let cause = error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>());
    assert!(matches!(
        cause,
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

fn shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Struct(_) => vec![1, 1],
        Value::StructArray(value) => value.shape().to_vec(),
        _ => panic!("expected structure value"),
    }
}

#[test]
fn scalar_linear_growth_preserves_schema_and_fills_holes() {
    let plan = build_assignment_plan(&[SliceSelector::Scalar(3)], 1, &[1, 1]).unwrap();
    let result = assign_with_plan(
        Value::Struct(element(1.0)),
        &plan,
        Value::Struct(element(3.0)),
        false,
    )
    .unwrap();
    assert_eq!(shape(&result), vec![1, 3]);
    let Value::StructArray(result) = result else {
        unreachable!()
    };
    assert!(
        matches!(result.get_linear(1).unwrap().fields.get("value"), Some(Value::Tensor(value)) if value.is_empty())
    );
    assert_eq!(
        result.get_linear(2).unwrap().fields.get("value"),
        Some(&Value::Num(3.0))
    );
}

#[test]
fn assignment_requires_the_exact_ordered_schema() {
    let mut reversed = runmat_value::UnboundedStructValueBuilder::new();
    reversed.insert("other", Value::Num(1.0));
    reversed.insert("value", Value::Num(2.0));
    let plan = build_index_plan(&[SliceSelector::Scalar(1)], 1, &[1, 2]).unwrap();
    let error = assign_with_plan(
        array(vec![1, 2]),
        &plan,
        Value::Struct(reversed.finish()),
        false,
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:DissimilarStructureAssignment")
    );
}

#[test]
fn linear_deletion_preserves_vector_orientation_and_deduplicates() {
    let row_plan = IndexPlan::new(vec![0, 0, 2], vec![1, 3], vec![3], 1, vec![1, 4]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![1, 4]), &row_plan, empty_rhs(), true).unwrap()),
        vec![1, 2]
    );
    let column_plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2], 1, vec![4, 1]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![4, 1]), &column_plan, empty_rhs(), true).unwrap()),
        vec![2, 1]
    );
    let all_row = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![1, 2]), &all_row, empty_rhs(), true).unwrap()),
        vec![1, 0]
    );
    let all_column = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2], 1, vec![2, 1]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 1]), &all_column, empty_rhs(), true).unwrap()),
        vec![0, 1]
    );
}

#[test]
fn scalar_and_empty_selection_deletion_are_normalized() {
    let scalar = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
    assert_eq!(
        shape(&assign_with_plan(Value::Struct(element(1.0)), &scalar, empty_rhs(), true).unwrap()),
        vec![0, 0]
    );
    let none = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![1, 2]);
    assert_eq!(
        shape(&assign_with_plan(array(vec![1, 2]), &none, empty_rhs(), true).unwrap()),
        vec![1, 2]
    );
}

#[test]
fn subscript_deletion_supports_one_dimension_and_rejects_two_partial_dimensions() {
    let row = build_index_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Colon],
        2,
        &[2, 3],
    )
    .unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 3]), &row, empty_rhs(), true).unwrap()),
        vec![1, 3]
    );
    let column = build_index_plan(
        &[SliceSelector::Colon, SliceSelector::Scalar(2)],
        2,
        &[2, 3],
    )
    .unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 3]), &column, empty_rhs(), true).unwrap()),
        vec![2, 2]
    );
    let invalid = build_index_plan(
        &[SliceSelector::Scalar(1), SliceSelector::Scalar(1)],
        2,
        &[2, 2],
    )
    .unwrap();
    assert_eq!(
        assign_with_plan(array(vec![2, 2]), &invalid, empty_rhs(), true)
            .unwrap_err()
            .identifier(),
        Some("RunMat:UnsupportedStructDeletion")
    );
}

#[test]
fn full_and_nd_dimension_deletion_preserve_typed_empty_and_shape() {
    let full = build_index_plan(&[SliceSelector::Colon, SliceSelector::Colon], 2, &[2, 2]).unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 2]), &full, empty_rhs(), true).unwrap()),
        vec![0, 0]
    );
    let nd = build_index_plan(
        &[
            SliceSelector::Colon,
            SliceSelector::Scalar(1),
            SliceSelector::Colon,
        ],
        3,
        &[2, 2, 2],
    )
    .unwrap();
    assert_eq!(
        shape(&assign_with_plan(array(vec![2, 2, 2]), &nd, empty_rhs(), true).unwrap()),
        vec![2, 1, 2]
    );
}

#[test]
fn deletion_rejects_nonempty_rhs() {
    let plan = build_index_plan(&[SliceSelector::Scalar(1)], 1, &[1, 2]).unwrap();
    assert_eq!(
        assign_with_plan(array(vec![1, 2]), &plan, Value::Num(0.0), true)
            .unwrap_err()
            .identifier(),
        Some("RunMat:InvalidStructDeletionRhs")
    );
}
