use super::*;
use crate::indexing::{
    plan::IndexPlan,
    read_slice::read_sparse_slice_from_plan_in_context,
    write_slice::{assign_sparse_with_plan_host_in_context, delete_sparse_with_plan_in_context},
};
use runmat_value::{IntValue, IntegerStorage, SparseTensor, Tensor};

#[test]
fn sparse_indexing_keeps_native_classes_and_exact_claims_through_clones() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let ledger = ManagedHostEffectLedger::new(131_072);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    context.set_runmat_extensions_enabled(true);
    let plan = IndexPlan::new(vec![0, 2, 4], vec![3, 1], vec![3], 1, vec![3, 2]);
    let sources = [
        SparseTensor::new_integer(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            IntegerStorage::U64(vec![u64::MAX, 0, 42]),
        )
        .unwrap(),
        SparseTensor::new_f32(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            vec![f32::from_bits(0x7fc01234), -0.0, 7.0],
        )
        .unwrap(),
        SparseTensor::new_complex_f32(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            vec![(f32::from_bits(0x7fc01234), -0.0), (0.0, 0.0), (7.0, 2.0)],
        )
        .unwrap(),
        SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1]).unwrap(),
    ];
    for source in &sources {
        let Value::SparseTensor(output) =
            read_sparse_slice_from_plan_in_context(Some(&context), source, &plan).unwrap()
        else {
            panic!("sparse result")
        };
        assert_eq!(output.shape(), [3, 1]);
        assert_eq!(output.class_name(), source.class_name());
        if let Some(values) = output.integer_storage() {
            assert_eq!(values, &IntegerStorage::U64(vec![u64::MAX, 0, 42]));
        }
        if let Some(values) = output.as_f32_slice() {
            assert_eq!(values[0].to_bits(), 0x7fc01234);
            assert_eq!(output.nnz(), 2);
        }
        if let Some(values) = output.as_complex_f32_slice() {
            assert_eq!(values[0].0.to_bits(), 0x7fc01234);
            assert_eq!(values[0].1.to_bits(), (-0.0f32).to_bits());
            assert_eq!(output.nnz(), 3);
        }
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        let retained = output.clone();
        drop(output);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(retained);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn sparse_merge_growth_delete_and_scalar_share_existing_finite_owner() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let ledger = ManagedHostEffectLedger::new(131_072);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    context.set_runmat_extensions_enabled(true);
    let source =
        SparseTensor::new_integer(3, 2, vec![0, 0, 0], vec![], IntegerStorage::U64(vec![]))
            .unwrap();
    let alias = source.clone();
    let rhs = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX, 0]), vec![1, 3]).unwrap(),
    );
    let plan = IndexPlan::new(vec![1, 1, 3], vec![1, 3], vec![3], 1, vec![3, 2]);
    let Value::SparseTensor(output) =
        assign_sparse_with_plan_host_in_context(Some(&context), source, &plan, &rhs).unwrap()
    else {
        panic!("sparse merge")
    };
    assert_eq!(
        output.numeric_value_at(0),
        Some(NumericScalar::U64(u64::MAX))
    );
    assert_eq!(output.nnz(), 1);
    assert_eq!(alias.nnz(), 0);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let source = SparseTensor::new_integer(
        1,
        2,
        vec![0, 1, 1],
        vec![0],
        IntegerStorage::U64(vec![u64::MAX]),
    )
    .unwrap();
    let alias = source.clone();
    let plan = IndexPlan::new(vec![], vec![0, 1], vec![0], 1, vec![1, 4]);
    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
    let Value::SparseTensor(grown) =
        assign_sparse_with_plan_host_in_context(Some(&context), source, &plan, &empty).unwrap()
    else {
        panic!("sparse growth")
    };
    assert_eq!(grown.shape(), [1, 4]);
    assert!(grown.row_indices.shares_allocation_with(&alias.row_indices));
    assert!(grown
        .numeric_host_buffer()
        .unwrap()
        .shares_allocation_with(alias.numeric_host_buffer().unwrap()));
    assert!(!grown.col_ptrs.shares_allocation_with(&alias.col_ptrs));
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(grown);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let source = SparseTensor::new_integer(
        3,
        2,
        vec![0, 2, 3],
        vec![0, 2, 1],
        IntegerStorage::U64(vec![1, u64::MAX, 42]),
    )
    .unwrap();
    let deletion = IndexPlan::new(vec![0, 3], vec![1, 2], vec![1, 2], 2, vec![3, 2]);
    let Value::SparseTensor(output) =
        delete_sparse_with_plan_in_context(Some(&context), source, &deletion, &empty).unwrap()
    else {
        panic!("sparse deletion")
    };
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, 42]))
    );
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let source = SparseTensor::zeros_f32(1, 1);
    let witness = Value::Int(IntValue::U64((1u64 << 63) + (1u64 << 39) + 1));
    let Value::SparseTensor(output) = futures::executor::block_on(
        crate::indexing::write_linear::assign_sparse_scalar_in_context(
            Some(&context),
            source,
            &[1],
            &witness,
            false,
        ),
    )
    .unwrap() else {
        panic!("single scalar")
    };
    assert_eq!(output.as_f32_slice().unwrap()[0].to_bits(), 0x5f000001);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn sparse_finite_denial_and_cancel_leave_original_csc_untouched() {
    let source = SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 4.0]).unwrap();
    let plan = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![2, 2]);
    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1]).unwrap());
    for cancel in [false, true] {
        let ledger = ManagedHostEffectLedger::new(if cancel { 131_072 } else { 0 });
        let cancellation = CancellationSource::new();
        if cancel {
            cancellation.cancel(CancellationReason::User);
        }
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        assert!(read_sparse_slice_from_plan_in_context(Some(&context), &source, &plan).is_err());
        assert!(assign_sparse_with_plan_host_in_context(
            Some(&context),
            source.clone(),
            &plan,
            &Value::Num(9.0)
        )
        .is_err());
        let deletion = IndexPlan::new(vec![0, 2], vec![1, 2], vec![1, 2], 2, vec![2, 2]);
        assert!(delete_sparse_with_plan_in_context(
            Some(&context),
            source.clone(),
            &deletion,
            &empty
        )
        .is_err());
        assert!(futures::executor::block_on(
            crate::indexing::write_linear::assign_sparse_scalar_in_context(
                Some(&context),
                source.clone(),
                &[1],
                &Value::Num(9.0),
                false
            )
        )
        .is_err());
        assert_eq!(source.as_f64_slice(), Some(&[3.0, 4.0][..]));
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn sparse_matrix_selection_keeps_duplicate_row_order_and_colon_cost_sparse() {
    let ledger = ManagedHostEffectLedger::new(131_072);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let source =
        SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0]).unwrap();
    let rows = Value::Tensor(Tensor::new(vec![3.0, 1.0, 3.0], vec![1, 3]).unwrap());
    let cols = Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap());
    let Value::SparseTensor(output) =
        futures::executor::block_on(crate::indexing::read_slice::read_sparse_slice_in_context(
            Some(&context),
            &source,
            2,
            0,
            0,
            &[rows, cols],
        ))
        .unwrap()
    else {
        panic!("sparse matrix")
    };
    assert_eq!(output.shape(), [3, 2]);
    assert_eq!(output.col_ptrs.as_ref(), &[0, 0, 3]);
    assert_eq!(output.row_indices.as_ref(), &[0, 1, 2]);
    assert_eq!(output.as_f64_slice(), Some(&[30.0, 10.0, 30.0][..]));
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let source =
        SparseTensor::new(1_000_000_000, 1, vec![0, 1], vec![999_999_999], vec![2.0]).unwrap();
    let Value::SparseTensor(output) =
        futures::executor::block_on(crate::indexing::read_slice::read_sparse_slice_in_context(
            Some(&context),
            &source,
            2,
            1,
            0,
            &[Value::Num(1.0)],
        ))
        .unwrap()
    else {
        panic!("sparse colon")
    };
    assert_eq!(output.shape(), [1_000_000_000, 1]);
    assert_eq!(output.row_indices.as_ref(), &[999_999_999]);
    assert_eq!(output.as_f64_slice(), Some(&[2.0][..]));
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    use crate::indexing::read_slice::{
        read_sparse_slice_with_policy_in_context, try_read_sparse_slice_host_in_context,
        SparseReadZeroPolicy,
    };
    // Both actual Native entry capabilities must retain the sparse colon cost.
    let policy = SparseReadZeroPolicy::PreserveTypedComponents;
    let host = try_read_sparse_slice_host_in_context(
        &context,
        &source,
        2,
        1,
        0,
        &[Value::Num(1.0)],
        policy,
    )
    .unwrap()
    .expect("pure host sparse selectors");
    let resumed = futures::executor::block_on(read_sparse_slice_with_policy_in_context(
        Some(&context),
        &source,
        2,
        1,
        0,
        &[Value::Num(1.0)],
        policy,
    ))
    .unwrap();
    assert_eq!(host, resumed);
    let Value::SparseTensor(host_sparse) = &host else {
        panic!("sparse Native result");
    };
    assert_eq!(host_sparse.shape(), [1_000_000_000, 1]);
    assert_eq!(host_sparse.nnz(), 1);
    drop((host, resumed));
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let sources = [
        SparseTensor::new_integer(
            2,
            1,
            vec![0, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![0, u64::MAX]),
        )
        .unwrap(),
        SparseTensor::new_complex_f32(2, 1, vec![0, 2], vec![0, 1], vec![(-0.0, 0.0), (7.0, 2.0)])
            .unwrap(),
        SparseTensor::new_f32(2, 1, vec![0, 2], vec![0, 1], vec![-0.0, 7.0]).unwrap(),
    ];
    let column = [Value::Num(1.0)];
    for source in sources {
        for (dims, colon_mask, numeric, indices, shape, lengths) in [
            (1, 1, &[][..], vec![0, 1], vec![2, 1], vec![2]),
            (2, 1, column.as_slice(), vec![0, 1], vec![2, 1], vec![2, 1]),
            (1, 0, column.as_slice(), vec![0], vec![1, 1], vec![1]),
        ] {
            let plan = IndexPlan::new(indices, shape, lengths, dims, vec![2, 1]);
            let expected =
                read_sparse_slice_from_plan_in_context(Some(&context), &source, &plan).unwrap();
            let host = try_read_sparse_slice_host_in_context(
                &context, &source, dims, colon_mask, 0, numeric, policy,
            )
            .unwrap()
            .expect("host selector capability");
            let resumed = futures::executor::block_on(read_sparse_slice_with_policy_in_context(
                Some(&context),
                &source,
                dims,
                colon_mask,
                0,
                numeric,
                policy,
            ))
            .unwrap();
            assert_eq!(host, expected, "host physical component policy");
            assert_eq!(resumed, expected, "resumed physical component policy");
            let Value::SparseTensor(output) = &host else {
                panic!("sparse component result");
            };
            assert_eq!(output.class_name(), source.class_name());
            assert_eq!(
                output.nnz(),
                if source.is_complex() || source.integer_storage().is_some() {
                    plan.indices.len()
                } else {
                    usize::from(plan.indices.contains(&1))
                }
            );
            if let Some(values) = output.as_complex_f32_slice() {
                assert_eq!(values[0].0.to_bits(), (-0.0f32).to_bits());
            }
            drop((expected, host, resumed));
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
}
