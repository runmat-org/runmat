use super::*;
use futures::executor::block_on;
use runmat_value::Tensor;
use std::rc::Rc;

use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};

struct FiniteHost(ManagedHostEffectLedger);

impl RuntimeParallelService for FiniteHost {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

fn finite_runtime(ledger: ManagedHostEffectLedger) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(FiniteHost(ledger))),
        )
}

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("test tensor"))
}

fn bounds(value: Value) -> Vec<f64> {
    let Value::Tensor(tensor) = value else {
        panic!("expected interval tensor")
    };
    assert_eq!(tensor.shape()[1], 2);
    tensor.materialize_f64()
}

#[test]
fn orthogonal_jacobian_and_independent_covariance_agree() {
    // Independent two-parameter orthogonal design: J'J = diag(4,4),
    // SSE = 4, df = 2, and parameter variances are both 1/2.
    let beta = tensor(vec![1.0, 2.0], 2, 1);
    let residual = tensor(vec![1.0, -1.0, 1.0, -1.0], 4, 1);
    let jacobian = tensor(vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0], 4, 2);
    let covariance = tensor(vec![0.5, 0.0, 0.0, 0.5], 2, 2);
    let from_j = bounds(
        block_on(nlparci_builtin(
            beta.clone(),
            residual.clone(),
            vec![Value::from("Jacobian"), jacobian],
        ))
        .expect("Jacobian interval"),
    );
    let from_cov = bounds(
        block_on(nlparci_builtin(
            beta,
            residual,
            vec![Value::from("Covar"), covariance],
        ))
        .expect("covariance interval"),
    );
    let half = 4.302_652_729_749_462 * 0.5_f64.sqrt();
    for (actual, expected) in from_j
        .iter()
        .zip([1.0 - half, 2.0 - half, 1.0 + half, 2.0 + half])
    {
        assert!((actual - expected).abs() < 1.0e-8, "{actual} vs {expected}");
    }
    for (left, right) in from_j.iter().zip(from_cov) {
        assert!((left - right).abs() < 1.0e-9);
    }
}

#[test]
fn nan_rows_are_omitted_and_alpha_changes_width() {
    let beta = Value::Num(2.0);
    let residual = tensor(vec![1.0, -1.0, f64::NAN, 0.0], 4, 1);
    let jacobian = tensor(vec![1.0, 1.0, 1.0, 1.0], 4, 1);
    let ci95 = bounds(
        block_on(nlparci_builtin(
            beta.clone(),
            residual.clone(),
            vec![Value::from("Jacobian"), jacobian.clone()],
        ))
        .expect("default interval"),
    );
    let ci90 = bounds(
        block_on(nlparci_builtin(
            beta,
            residual,
            vec![
                Value::from("Jacobian"),
                jacobian,
                Value::from("Alpha"),
                Value::Num(0.1),
            ],
        ))
        .expect("alpha interval"),
    );
    assert!(ci90[1] - ci90[0] < ci95[1] - ci95[0]);
    assert!((ci95[0] + ci95[1] - 4.0).abs() < 1.0e-12);

    let very_narrow_tail = bounds(
        block_on(nlparci_builtin(
            Value::Num(2.0),
            tensor(vec![1.0, -1.0, 0.0], 3, 1),
            vec![
                Value::from("Jacobian"),
                tensor(vec![1.0, 1.0, 1.0], 3, 1),
                Value::from("Alpha"),
                Value::Num(1.0e-17),
            ],
        ))
        .expect("small positive Alpha"),
    );
    assert!(very_narrow_tail[0].is_finite() && very_narrow_tail[1].is_finite());
    assert!(very_narrow_tail[1] - very_narrow_tail[0] > ci95[1] - ci95[0]);
}

#[test]
fn invalid_rank_dof_alpha_and_classes_have_stable_identifiers() {
    let beta = tensor(vec![1.0, 2.0], 2, 1);
    let residual = tensor(vec![1.0, -1.0, 0.0], 3, 1);
    let rank_deficient = tensor(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 3, 2);
    let err = block_on(nlparci_builtin(
        beta.clone(),
        residual.clone(),
        vec![Value::from("Jacobian"), rank_deficient],
    ))
    .expect_err("rank deficient Jacobian");
    assert_eq!(err.identifier(), Some("RunMat:nlparci:NumericalFailure"));

    let covariance = tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    let err = block_on(nlparci_builtin(
        beta.clone(),
        residual.clone(),
        vec![
            Value::from("Covar"),
            covariance.clone(),
            Value::from("Alpha"),
            Value::Num(1.0),
        ],
    ))
    .expect_err("invalid alpha");
    assert_eq!(err.identifier(), Some("RunMat:nlparci:InvalidArgument"));

    let err = block_on(nlparci_builtin(
        beta,
        residual,
        vec![
            Value::from("Covar"),
            covariance,
            Value::from("Alpha"),
            Value::Int(runmat_value::IntValue::I32(1)),
        ],
    ))
    .expect_err("typed integer alpha is not documented");
    assert_eq!(err.identifier(), Some("RunMat:nlparci:InvalidArgument"));
}

#[test]
fn zero_requested_outputs_preserve_input_and_numerical_errors() {
    let cases = [
        (
            Value::Num(f64::INFINITY),
            tensor(vec![1.0, -1.0], 2, 1),
            Value::from("Covar"),
            Value::Num(1.0),
            "RunMat:nlparci:InvalidArgument",
        ),
        (
            Value::Num(1.0),
            tensor(vec![1.0, f64::INFINITY], 2, 1),
            Value::from("Covar"),
            Value::Num(1.0),
            "RunMat:nlparci:InvalidArgument",
        ),
        (
            Value::Num(1.0),
            tensor(vec![1.0, f64::NAN], 2, 1),
            Value::from("Covar"),
            Value::Num(1.0),
            "RunMat:nlparci:NumericalFailure",
        ),
        (
            tensor(vec![1.0, 2.0], 2, 1),
            tensor(vec![1.0, -1.0, 0.0], 3, 1),
            Value::from("Jacobian"),
            tensor(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 3, 2),
            "RunMat:nlparci:NumericalFailure",
        ),
        (
            Value::Num(1.0),
            tensor(vec![1.0, -1.0], 2, 1),
            Value::from("Covar"),
            Value::Num(-1.0),
            "RunMat:nlparci:InvalidArgument",
        ),
    ];
    for (beta, residual, method, source, identifier) in cases {
        for requested in [0, 1] {
            let _outputs = crate::output_count::push_output_count(Some(requested))
                .expect("output-count scope");
            let error = block_on(nlparci_in_runtime(
                None,
                beta.clone(),
                residual.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    method.clone(),
                    source.clone(),
                ]),
            ))
            .expect_err("invalid call must reject independent of output count");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }
}

#[test]
fn single_bounds_and_insufficient_complete_rows_keep_their_contracts() {
    let beta = Value::Tensor(
        Tensor::from_numeric_storage(runmat_value::NumericStorage::F32(vec![2.0]), vec![1, 1])
            .expect("single beta"),
    );
    let residual = tensor(vec![1.0, -1.0, 0.0], 3, 1);
    let covariance = tensor(vec![1.0 / 3.0], 1, 1);
    let output = block_on(nlparci_builtin(
        beta.clone(),
        residual.clone(),
        vec![Value::from("Covar"), covariance.clone()],
    ))
    .expect("single interval");
    let Value::Tensor(output) = output else {
        panic!("expected tensor")
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(output.shape(), [1, 2]);

    let missing = tensor(vec![1.0, f64::NAN, f64::NAN], 3, 1);
    let err = block_on(nlparci_builtin(
        beta,
        missing,
        vec![Value::from("Covar"), covariance],
    ))
    .expect_err("zero residual degrees of freedom");
    assert_eq!(err.identifier(), Some("RunMat:nlparci:NumericalFailure"));
}

#[test]
fn cancelled_invocation_does_not_enter_interval_work() {
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(nlparci_in_runtime(
        Some(&runtime),
        Value::Num(1.0),
        tensor(vec![1.0, -1.0], 2, 1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Covar"),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("cancelled before materialization");
    let cancelled = crate::execution::ExecutionServiceError::Cancelled.into_runtime_error();
    assert_eq!(error.identifier(), cancelled.identifier());
}

#[test]
fn finite_host_precharge_and_retained_interval_claim_share_one_owner() {
    let beta = tensor(vec![1.0, 2.0], 2, 1);
    let residual = tensor(vec![1.0, -1.0, 1.0, -1.0], 4, 1);
    let jacobian = tensor(vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0], 4, 2);
    let ledger = ManagedHostEffectLedger::new_with_epoch(65_536, 3);
    let runtime = finite_runtime(ledger.clone());
    let output = block_on(nlparci_in_runtime(
        Some(&runtime),
        beta.clone(),
        residual.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Jacobian"),
            jacobian.clone(),
        ]),
    ))
    .expect("finite admitted interval");
    let Value::Tensor(ref interval_tensor) = output else {
        panic!("expected retained interval tensor")
    };
    assert_eq!(interval_tensor.shape(), [2, 2]);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // Num beta/Covar/Alpha are borrowed scalar inputs. The exact finite
    // reservation needs no hidden one-element Tensor conversions.
    let scalar_beta = Value::Num(1.0);
    let scalar_residual = tensor(vec![1.0, -1.0], 2, 1);
    let scalar_covar = Value::Num(0.25);
    let scalar_alpha = Value::Num(0.05);
    let scalar_plan = allocation::plan(
        &scalar_beta,
        &scalar_residual,
        &scalar_covar,
        Some(&scalar_alpha),
        admission::Method::Covariance,
        true,
    )
    .unwrap();
    let exact_capacity = scalar_plan.scratch_peak_bytes() + scalar_plan.retained_output_bytes();
    let scalar_ledger = ManagedHostEffectLedger::new_with_epoch(exact_capacity, 3);
    let scalar_runtime = finite_runtime(scalar_ledger.clone());
    let scalar_output = block_on(nlparci_in_runtime(
        Some(&scalar_runtime),
        scalar_beta,
        scalar_residual,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Covar"),
            scalar_covar,
            Value::from("Alpha"),
            scalar_alpha,
        ]),
    ))
    .expect("borrowed host scalars fit their checked finite envelope");
    assert!(matches!(&scalar_output, Value::Tensor(tensor) if tensor.shape() == [1, 2]));
    drop(scalar_output);
    assert_eq!(scalar_ledger.snapshot().live_reserved_bytes, 0);

    let zero_plan = allocation::plan(
        &Value::Num(1.0),
        &tensor(vec![1.0, -1.0], 2, 1),
        &Value::Num(0.25),
        None,
        admission::Method::Covariance,
        false,
    )
    .unwrap();
    assert_eq!(
        zero_plan.scratch_peak_bytes(),
        scalar_plan.scratch_peak_bytes()
    );
    assert_eq!(zero_plan.retained_output_bytes(), 0);
    let zero_ledger = ManagedHostEffectLedger::new_with_epoch(zero_plan.scratch_peak_bytes(), 3);
    let zero_runtime = finite_runtime(zero_ledger.clone());
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let zero_output = block_on(nlparci_in_runtime(
            Some(&zero_runtime),
            Value::Num(1.0),
            tensor(vec![1.0, -1.0], 2, 1),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Covar"),
                Value::Num(0.25),
            ]),
        ))
        .expect("zero outputs still perform the admitted interval calculation");
        assert!(matches!(zero_output, Value::OutputList(values) if values.is_empty()));
    }
    assert_eq!(zero_ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(zero_ledger.snapshot().live_reserved_bytes, 0);

    let insufficient = ManagedHostEffectLedger::new_with_epoch(1, 3);
    let runtime = finite_runtime(insufficient.clone());
    let error = block_on(nlparci_in_runtime(
        Some(&runtime),
        beta,
        residual,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Jacobian"),
            jacobian,
        ]),
    ))
    .expect_err("finite envelope must reject before numerical work");
    assert_eq!(error.identifier(), Some("RunMat:nlparci:HostAllocation"));
    assert_eq!(insufficient.snapshot().live_reserved_bytes, 0);
}
