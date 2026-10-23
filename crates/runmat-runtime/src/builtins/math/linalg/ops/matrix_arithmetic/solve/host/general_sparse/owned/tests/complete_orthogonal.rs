use super::*;

#[test]
fn sparse_complete_solve_geometry_and_claim_lifetime() {
    let cases = [
        (
            2,
            3,
            vec![0, 1, 2, 4],
            vec![0, 1, 0, 1],
            vec![1., 1., 1., 1.],
            vec![1., 1.],
            vec![1. / 3., 1. / 3., 2. / 3.],
        ),
        // Proven zero triangular diagonal selects COD, not a broad retry.
        (
            2,
            2,
            vec![0, 1, 2],
            vec![0, 0],
            vec![1., 2.],
            vec![5., 3.],
            vec![1., 2.],
        ),
        (
            0,
            3,
            vec![0, 0, 0, 0],
            vec![],
            vec![],
            vec![],
            vec![0., 0., 0.],
        ),
        (2, 0, vec![0], vec![], vec![], vec![1., 2.], vec![]),
    ];
    for (m, n, p, r, a, b, expected) in cases {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let context = context(&ledger, Rc::new(InlineCpu::default()));
        let lhs = Value::SparseTensor(SparseTensor::new(m, n, p, r, a).unwrap());
        let rhs = Value::Tensor(Tensor::new(b, vec![m, 1]).unwrap());
        let output = solve(&lhs, &rhs, &context).unwrap();
        let Value::Tensor(tensor) = &output else {
            panic!("real result required")
        };
        assert_eq!(tensor.shape(), &[n, 1]);
        for (&actual, expected) in tensor.as_f64_slice().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[n, 1]).unwrap()
        );
        let retained = output.clone();
        drop(output);
        assert_eq!(
            ledger.snapshot().live_claimed_bytes,
            HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[n, 1]).unwrap()
        );
        drop(retained);
        empty(&ledger);
    }
}

#[test]
fn sparse_qr_finite_capacity_and_cancel_errors_leave_no_claims() {
    let lhs = Value::SparseTensor(
        SparseTensor::new(
            2,
            3,
            vec![0, 1, 2, 4],
            vec![0, 1, 0, 1],
            vec![1., 1., 1., 1.],
        )
        .unwrap(),
    );
    let rhs = Value::Tensor(Tensor::new(vec![1., 1.], vec![2, 1]).unwrap());
    let ledger = ManagedHostEffectLedger::new(1);
    let context = context(&ledger, Rc::new(InlineCpu::default()));
    let error = solve(&lhs, &rhs, &context).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    empty(&ledger);
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let cancel = runmat_execution::cancellation::CancellationSource::new();
    let context =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), cancel.token())
            .with_service_ports(
                RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
            );
    cancel.cancel(runmat_execution::CancellationReason::User);
    let error = solve(&lhs, &rhs, &context).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    empty(&ledger);
}

#[test]
fn sparse_qr_original_issuer_and_nonfinite_rhs_remain_terminal() {
    use crate::builtins::math::linalg::ops::sparse_factor::{
        SparseCompleteQr, SparseCompleteQrError,
    };
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let runtime = context(&ledger, Rc::new(InlineCpu::default()));
    let sparse = SparseTensor::new(
        2,
        3,
        vec![0, 1, 2, 4],
        vec![0, 1, 0, 1],
        vec![1., 1., 1., 1.],
    )
    .unwrap();
    let factor = block_on(SparseCompleteQr::<f64>::factor(&sparse, &runtime)).unwrap();
    let substitute_ledger = ManagedHostEffectLedger::new(2_000_000);
    let substitute = context(&substitute_ledger, Rc::new(InlineCpu::default()));
    assert!(matches!(
        factor.check_owner(&substitute),
        Err(HostAllocationError::StaleAuthority)
    ));
    let mut writes = 0;
    let error = block_on(factor.solve_with_output(
        &[f64::NAN, 1.],
        1,
        &runtime,
        std::convert::identity,
        |_, _| {
            writes += 1;
            Ok(())
        },
    ))
    .unwrap_err();
    assert!(matches!(error, SparseCompleteQrError::NonFiniteInput));
    assert_eq!(writes, 0, "nonfinite RHS cannot publish a coordinate");
    drop(factor);
    empty(&ledger);
    empty(&substitute_ledger);
}
