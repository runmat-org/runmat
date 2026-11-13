use crate::builtins::math::linalg::ops::iterative_preconditioner::PreparedMatrixFactor;
use crate::builtins::math::linalg::ops::matrix_arithmetic::solve::{
    evaluate_with_context, SolveOrientation,
};
use crate::context::host_allocation::HostAllocationPlan;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use crate::execution::{
    ExecutionServiceError, OwnedCpuJob, RuntimeBlockingCpuService, RuntimeExecutionService,
};
use futures::executor::block_on;
use runmat_value::{ComplexElement, ComplexTensor, NumericDType, SparseTensor, Tensor, Value};
use std::rc::Rc;

mod fixtures;
use fixtures::context;

fn empty(ledger: &ManagedHostEffectLedger) {
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

mod cases;
use cases::Class;
mod cancellation;

#[test]
fn lower_upper_native_classes_publish_only_exact_payload_and_shape_without_lu() {
    for lower in [false, true] {
        for class in [
            Class::RealF32,
            Class::RealF64,
            Class::ComplexF32,
            Class::ComplexF64,
        ] {
            let ledger = ManagedHostEffectLedger::new(2_000_000);
            let context = context(&ledger);
            let (lhs, rhs) = class.inputs(lower);
            let result = block_on(evaluate_with_context(
                SolveOrientation::Left,
                &lhs,
                &rhs,
                Some(&context),
            ))
            .unwrap();
            class.verify(&result);
            assert_eq!(ledger.snapshot().live_claimed_bytes, class.quote());
            assert_eq!(ledger.snapshot().live_reserved_bytes, class.quote());
            let clone = result.clone();
            drop(result);
            class.verify(&clone);
            assert_eq!(ledger.snapshot().live_claimed_bytes, class.quote());
            drop(clone);
            empty(&ledger);
        }
    }
}

#[test]
fn iterative_triangle_remains_borrowed_and_reuses_caller_output_without_allocation() {
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let context = context(&ledger);
    let (matrix, _) = Class::RealF64.inputs(true);
    let mut factor = block_on(PreparedMatrixFactor::<f64>::prepare(
        &matrix,
        2,
        Some(&context),
    ))
    .unwrap();
    empty(&ledger);
    for _ in 0..3 {
        let mut output = [0.0; 2];
        block_on(factor.solve_into(&[2.0, 11.0], &mut output, Some(&context))).unwrap();
        assert_eq!(output, [1.0, 2.0]);
        empty(&ledger);
    }
}

#[test]
fn exact_complex_mutable_view_preserves_native_bits_and_final_claims() {
    use super::super::super::managed_output::ManagedSolveOutput;
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let context = context(&ledger);
    let mut output = ManagedSolveOutput::<ComplexElement<f32>>::new(&[2, 2], &context).unwrap();
    let nan = f32::from_bits(0x7fc0_1234);
    {
        let view = output.as_mut_slice().unwrap();
        view[0] = ComplexElement(-0.0, nan);
        view[1] = ComplexElement(3.0, -4.0);
    }
    let value = output.finish(&context).unwrap();
    let Value::ComplexTensor(tensor) = value else {
        panic!("native class changed")
    };
    assert_eq!(
        tensor.as_f32_slice().unwrap()[0].0.to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(tensor.as_f32_slice().unwrap()[0].1.to_bits(), nan.to_bits());
    assert_eq!(tensor.as_f32_slice().unwrap()[1], ComplexElement(3.0, -4.0));
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        Class::ComplexF32.quote()
    );
    drop(tensor);
    empty(&ledger);
}

#[test]
fn invalid_triangular_diagonal_never_allocates_a_result_or_factors() {
    use crate::builtins::math::linalg::ops::triangular_operator::{
        Diagonal, Triangle, TriangularLinearOperator, TriangularOperatorError,
    };
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let context = context(&ledger);
    for (matrix, row, expected) in [
        (
            SparseTensor::new(2, 2, vec![0, 1, 1], vec![1], vec![1.0]).unwrap(),
            [1.0, 0.0],
            [1.0, 0.0],
        ),
        (
            SparseTensor::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![0.0, 1.0, 2.0]).unwrap(),
            [1.0, 2.0],
            [0.2, 0.4],
        ),
    ] {
        let lhs = Value::SparseTensor(matrix);
        // Invalid for direct triangular substitution, but a valid singular
        // least-squares system for the public complete-orthogonal route.
        let invalid = match block_on(TriangularLinearOperator::borrow_cooperatively(
            &lhs,
            Triangle::Lower,
            Diagonal::Explicit,
            &context,
        )) {
            Err(error) => error,
            Ok(_) => panic!("direct triangular substitution rejects zero/missing diagonal"),
        };
        assert!(matches!(
            invalid,
            TriangularOperatorError::MissingDiagonal(_)
                | TriangularOperatorError::SingularDiagonal(_)
        ));
        let error = super::super::map_operator(invalid, &context);
        assert_eq!(
            error.identifier(),
            runmat_builtins::MLDIVIDE_ERROR_INVALID_INPUT.identifier
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        empty(&ledger);
        let rhs = Value::Tensor(Tensor::new(vec![1.0; 2], vec![2, 1]).unwrap());
        let result = block_on(evaluate_with_context(
            SolveOrientation::Left,
            &lhs,
            &rhs,
            Some(&context),
        ))
        .unwrap();
        let Value::Tensor(tensor) = &result else {
            panic!("real minimum-norm solution");
        };
        assert_eq!(tensor.shape(), [2, 1]);
        let x = tensor.as_f64_slice().unwrap();
        for (&actual, expected) in x.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        // A*x=[0;1]: residual is orthogonal to range(A); x is orthogonal
        // to null(A), independently certifying the minimum-norm LS result.
        assert!((row[0] * x[0] + row[1] * x[1] - 1.0).abs() < 1e-12);
        assert!((-row[1] * x[0] + row[0] * x[1]).abs() < 1e-12);
        let quote = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[2, 1]).unwrap();
        assert_eq!(ledger.snapshot().live_claimed_bytes, quote);
        assert_eq!(ledger.snapshot().live_reserved_bytes, quote);
        drop(result);
        empty(&ledger);
    }
    {
        let matrix =
            SparseTensor::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, f64::NAN, 2.0])
                .unwrap();
        let lhs = Value::SparseTensor(matrix);
        let rhs = Value::Tensor(Tensor::new(vec![1.0; 2], vec![2, 1]).unwrap());
        let error = block_on(evaluate_with_context(
            SolveOrientation::Left,
            &lhs,
            &rhs,
            Some(&context),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::MLDIVIDE_ERROR_INVALID_INPUT.identifier
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        empty(&ledger);
    }
}
