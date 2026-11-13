use super::*;
use futures::executor::block_on;
#[derive(Default)]
struct Transfers {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocations: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::TestAllocation>>,
    downloads: std::sync::atomic::AtomicUsize,
}
impl Transfers {
    fn value(
        &self,
        value: f64,
        intent: Option<runmat_accelerate_api::GpuHandleProvenance>,
    ) -> Value {
        let allocation = self
            .owner
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(vec![value]),
                shape: vec![1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        let handle = match intent {
            Some(intent) => allocation.handle.clone().with_provenance(intent).unwrap(),
            None => allocation.handle.clone(),
        };
        self.allocations.lock().unwrap().push(allocation);
        Value::GpuTensor(handle)
    }
}
impl runmat_accelerate_api::AccelProvider for Transfers {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "NNLS input admission witness".into()
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("fixture owns allocations directly")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("typed download required")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            let allocations = self.allocations.lock().unwrap();
            let allocation = allocations
                .iter()
                .find(|a| a.handle.allocation_identity() == handle.allocation_identity())
                .unwrap();
            let data = allocation.probe.with_data(handle, |data| {
                let runmat_accelerate_api::HostNumericDataView::F64(values) = data else {
                    panic!("double fixture")
                };
                runmat_accelerate_api::HostNumericDataOwned::F64(values.to_vec())
            })?;
            Ok(runmat_accelerate_api::HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }
}

#[test]
fn automatic_data_and_controls_are_transparent_but_explicit_and_unproven_refuse_before_transfer() {
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    use std::sync::atomic::Ordering;
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(Transfers::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let runtime = context();
    runtime.set_runmat_extensions_enabled(false);
    let call = |args| {
        block_on(execute(
            Some(&runtime),
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    };
    for data in [
        provider.value(2.0, Some(Explicit)),
        provider.value(2.0, None),
    ] {
        assert!(call(vec![
            provider.value(1.0, Some(Automatic)),
            data,
            options([("TolX", provider.value(1e-10, Some(Automatic)))])
        ])
        .is_err());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    }
    assert!(call(vec![
        provider.value(1.0, Some(Automatic)),
        Value::Num(2.0),
        options([("TolX", provider.value(1e-10, Some(Explicit)))])
    ])
    .is_err());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let result = run(
        &runtime,
        vec![
            provider.value(1.0, Some(Automatic)),
            provider.value(2.0, Some(Automatic)),
            options([("TolX", provider.value(1e-10, Some(Automatic)))]),
        ],
    );
    assert_eq!(values(&result[3]), vec![1.0]);
    assert_eq!(values(&result[0]), vec![2.0]);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 3);
}
#[test]
fn finite_workspace_policy_rejects_before_argument_work_or_capacity() {
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
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
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1_000_000);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
    );
    let before = ledger.snapshot();
    let error = block_on(lsqnonneg_builtin_registered(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let after = ledger.snapshot();
    assert_eq!(before.live_claimed_bytes, after.live_claimed_bytes);
    assert_eq!(before.live_reserved_bytes, after.live_reserved_bytes);
}
fn matrix(data: Vec<f64>, m: usize, n: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![m, n]).unwrap())
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(v) => vec![*v],
        Value::Tensor(t) => t.materialize_f64(),
        _ => panic!("numeric output"),
    }
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(value) = value else {
        panic!("struct");
    };
    &value.fields[name]
}
fn options(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut out = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        out.insert(name, value);
    }
    Value::Struct(out.finish())
}
fn context() -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
}
fn run(runtime: &crate::context::RuntimeContext, args: Vec<Value>) -> Vec<Value> {
    let _outputs = crate::output_count::push_output_count(Some(6)).unwrap();
    let value = block_on(execute(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap();
    let Value::OutputList(values) = value else {
        panic!("six outputs");
    };
    values
        .try_into_owned()
        .unwrap()
        .try_into_unbounded_vec()
        .unwrap()
}

#[test]
fn normal_public_fit_has_original_residual_and_nonpositive_dual() {
    let runtime = context();
    // Four-decimal observation data, not analytic square-root constants.
    let c = vec![
        0.0372,
        0.6861,
        0.6233,
        0.6344,
        0.2869,
        7071.0 / 10000.0,
        0.6245,
        0.6170,
    ];
    let d = vec![0.8587, 0.1781, 0.0747, 0.8405];
    let expected = c[4..].iter().zip(&d).map(|(a, b)| a * b).sum::<f64>()
        / c[4..].iter().map(|a| a * a).sum::<f64>();
    let result = run(
        &runtime,
        vec![matrix(c.clone(), 4, 2), matrix(d.clone(), 4, 1)],
    );
    let x = values(&result[0]);
    assert_eq!(x[0], 0.0);
    assert!((x[1] - expected).abs() < 1e-10);
    assert_eq!(values(&result[3]), vec![1.0]);
    let r = values(&result[2]);
    for row in 0..4 {
        assert!((r[row] - (d[row] - c[row] * x[0] - c[row + 4] * x[1])).abs() < 1e-12);
    }
    assert!((values(&result[1])[0] - r.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-12);
    let dual = values(&result[5]);
    assert!((dual[0] + 0.1506).abs() < 1e-4);
    assert!(dual[1].abs() < 1e-12);
    assert_eq!(values(field(&result[4], "iterations")), vec![1.0]);
}
#[test]
fn feasible_segment_removal_and_iteration_limits_are_actual_transactions() {
    let runtime = context();
    let args = |limit: Option<f64>| {
        let mut a = vec![
            matrix(vec![1.0, 0.0, 2.0, 1.0], 2, 2),
            matrix(vec![1.0, -0.5], 2, 1),
        ];
        if let Some(limit) = limit {
            a.push(options([
                ("MaxIter", Value::Num(limit)),
                ("Display", Value::from("off")),
            ]));
        }
        a
    };
    let result = run(&runtime, args(None));
    assert_eq!(values(&result[3]), vec![1.0]);
    assert_eq!(values(field(&result[4], "iterations")), vec![3.0]);
    assert!((values(&result[0])[0] - 1.0).abs() < 1e-12);
    assert_eq!(values(&result[0])[1], 0.0);
    assert!((values(&result[1])[0] - 0.25).abs() < 1e-12);
    for (limit, expected) in [
        (0.0, vec![0.0, 0.0]),
        (1.0, vec![0.0, 0.3]),
        (2.0, vec![0.75, 0.0]),
    ] {
        let result = run(&runtime, args(Some(limit)));
        assert_eq!(values(&result[3]), vec![0.0]);
        assert_eq!(values(field(&result[4], "iterations")), vec![limit]);
        for (actual, expected) in values(&result[0]).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
#[test]
fn problem_sparse_rank_deficient_wide_and_zero_models() {
    let runtime = context();
    let c = Value::SparseTensor(
        runmat_value::SparseTensor::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 1.0, 2.0],
        )
        .unwrap(),
    );
    let result = run(
        &runtime,
        vec![options([
            ("C", c),
            ("d", matrix(vec![3.0, 6.0], 1, 2)),
            ("solver", Value::from("lsqnonneg")),
            ("options", options([("Display", Value::from("off"))])),
        ])],
    );
    assert_eq!(values(&result[3]), vec![1.0]);
    let x = values(&result[0]);
    assert!((x.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    assert!(x.iter().all(|v| *v >= 0.0));
    let wide = run(&runtime, vec![matrix(vec![1.0; 3], 1, 3), Value::Num(3.0)]);
    assert_eq!(values(&wide[3]), vec![1.0]);
    assert!((values(&wide[0]).iter().sum::<f64>() - 3.0).abs() < 1e-12);
    let zero = run(
        &runtime,
        vec![matrix(vec![0.0; 6], 2, 3), matrix(vec![2.0, -3.0], 2, 1)],
    );
    assert_eq!(values(&zero[0]), vec![0.0; 3]);
    assert_eq!(values(&zero[2]), vec![2.0, -3.0]);
    // The scaled squared norm can round even when the residual is exact.
    assert!((values(&zero[1])[0] - 13.0).abs() <= 4.0 * f64::EPSILON * 13.0);
    assert_eq!(values(&zero[5]), vec![0.0; 3]);
    assert_eq!(values(&zero[3]), vec![1.0]);
}
#[test]
fn unsupported_options_and_numeric_range_never_manufacture_success() {
    let runtime = context();
    let call = |args| {
        block_on(execute(
            Some(&runtime),
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    };
    for opts in [
        options([("Algorithm", Value::from("interior-point"))]),
        options([("TolFun", Value::Num(1e-8))]),
        options([("TolX", Value::Num(0.0))]),
        options([("MaxIter", Value::Num(-1.0))]),
        options([("Display", Value::from("iter"))]),
        options([("unknown", empty())]),
    ] {
        assert!(call(vec![Value::Num(1.0), Value::Num(2.0), opts]).is_err());
    }
    // Known empty optimset fields are unset, not ignored nonempty requests.
    assert!(call(vec![
        Value::Num(1.0),
        Value::Num(2.0),
        options([("TolFun", empty()), ("OutputFcn", empty())])
    ])
    .is_ok());
    let error = call(vec![
        matrix(vec![1e300, 0.0, 0.0, 1e-300], 2, 2),
        matrix(vec![1.0, 1.0], 2, 1),
    ])
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:lsqnonneg:NumericalFailure")
    );
    assert!(call(vec![
        matrix(vec![1.0, f64::NAN], 2, 1),
        matrix(vec![1.0, 2.0], 2, 1)
    ])
    .is_err());
    assert!(call(vec![Value::Num(1.0), matrix(vec![1.0, 2.0], 2, 1)]).is_err());
}
#[test]
fn context_class_and_original_cancellation_ownership_are_preserved() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;
    let missing = block_on(execute(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(2.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        missing.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
    assert_eq!(missing.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let strict = context();
    strict.set_runmat_extensions_enabled(false);
    let enabled = context();
    enabled.set_runmat_extensions_enabled(true);
    {
        let _foreign = enabled.enter();
        for input in [
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::Tensor(
                Tensor::new_with_dtype(vec![1.0], vec![1, 1], runmat_value::NumericDType::F32)
                    .unwrap(),
            ),
        ] {
            assert!(block_on(execute(
                Some(&strict),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    input,
                    Value::Num(2.0)
                ])
            ))
            .is_err());
        }
        let tol = Value::Tensor(
            Tensor::new_with_dtype(vec![1e-8], vec![1, 1], runmat_value::NumericDType::F32)
                .unwrap(),
        );
        assert_eq!(
            values(
                &run(
                    &strict,
                    vec![Value::Num(1.0), Value::Num(2.0), options([("TolX", tol)])]
                )[3]
            ),
            vec![1.0]
        );
    }
    {
        let _foreign = strict.enter();
        assert_eq!(
            values(
                &run(
                    &enabled,
                    vec![
                        Value::Int(runmat_value::IntValue::I32(1)),
                        Value::Num(2.0),
                        options([("MaxIter", Value::Int(runmat_value::IntValue::I32(3)))])
                    ]
                )[0]
            ),
            vec![2.0]
        );
    }
    let original_source = CancellationSource::new();
    let foreign_source = CancellationSource::new();
    let original = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        original_source.token(),
    );
    original.set_error_namespace("OriginalNNLS");
    let foreign = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        foreign_source.token(),
    );
    foreign_source.cancel(CancellationReason::User);
    let a = runmat_value::SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
    {
        let _foreign = foreign.enter();
        let answer = block_on(
            async {
    let matrix=crate::builtins::math::linalg::ops::sparse_factor::least_squares::PassiveMatrix::borrow(
        &a,&original,runmat_builtins::LSQNONNEG_CATALOG_ENTRY.identity).await?;
    matrix.solve(&[0],&[2.0]).await
},
        )
        .unwrap();
        assert_eq!(answer.rank, 1);
        assert_eq!(answer.values, vec![2.0]);
    }
    original_source.cancel(CancellationReason::User);
    let error = block_on(
        async {
    let matrix=crate::builtins::math::linalg::ops::sparse_factor::least_squares::PassiveMatrix::borrow(
        &a,&original,runmat_builtins::LSQNONNEG_CATALOG_ENTRY.identity).await?;
    matrix.solve(&[0],&[2.0]).await
},
    )
    .err()
    .unwrap();
    assert_eq!(error.identifier(), Some("OriginalNNLS:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
}
#[test]
fn medium_dense_and_sparse_rank_deficient_actual_factor_paths() {
    let runtime = context();
    let (m, n) = (96, 24);
    let mut c = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            c[col * m + row] = if row == col {
                1.0
            } else {
                ((row + 1) * (col + 3) % 23) as f64 / 500.0
            };
        }
    }
    let expected = (0..n).map(|i| 1.0 + i as f64 / 10.0).collect::<Vec<_>>();
    let mut d = vec![0.0; m];
    for col in 0..n {
        for row in 0..m {
            d[row] += c[col * m + row] * expected[col];
        }
    }
    let started = runmat_time::Instant::now();
    let result = run(&runtime, vec![matrix(c, m, n), matrix(d, m, 1)]);
    eprintln!(
        "NNLS dense96x24 owned passive RRQR: {:?}",
        started.elapsed()
    );
    assert_eq!(values(&result[3]), vec![1.0]);
    assert!(values(&result[1])[0] < 1e-16);
    for (actual, expected) in values(&result[0]).iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-8);
    }
    let (m, n) = (192, 48);
    let mut rows = (0..n - 1).collect::<Vec<_>>();
    rows.push(0);
    let sparse = Value::SparseTensor(
        runmat_value::SparseTensor::new(m, n, (0..=n).collect::<Vec<_>>(), rows, vec![1.0; n])
            .unwrap(),
    );
    let d = (0..m)
        .map(|r| if r < n - 1 { (r + 1) as f64 } else { 0.0 })
        .collect::<Vec<_>>();
    let started = runmat_time::Instant::now();
    let result = run(&runtime, vec![sparse, matrix(d, m, 1)]);
    eprintln!(
        "NNLS sparse192x48 rank-deficient owned passive RRQR: {:?}",
        started.elapsed()
    );
    assert_eq!(values(&result[3]), vec![1.0]);
    assert!(values(&result[1])[0] < 1e-16);
    assert!(values(&result[0]).iter().all(|x| *x >= 0.0));
}
