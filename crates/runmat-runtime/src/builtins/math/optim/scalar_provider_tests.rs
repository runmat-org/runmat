use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_value::{IntValue, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct RejectDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

struct CancelAfterDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
    cancellation: runmat_execution::cancellation::CancellationSource,
}

impl AccelProvider for CancelAfterDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "scalar optimization cancellation fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("the scalar solver must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("the scalar solver must use the typed numeric download")
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        let cancellation = self.cancellation.clone();
        Box::pin(async move {
            cancellation.cancel(runmat_execution::CancellationReason::User);
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![0.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
        })
    }
}

#[test]
fn cancellation_after_first_bound_download_prevents_second_download_and_callback() {
    use std::rc::Rc;

    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let provider = Box::leak(Box::new(CancelAfterDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
        cancellation,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let mut lower = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![0.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    lower.handle = lower
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let mut upper = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    upper.handle = upper
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let error = block_on(super::fminbnd::fminbnd_builtin_registered(
        Some(runtime),
        Value::FunctionHandle("sin".into()),
        Value::GpuTensor(lower.handle.clone()),
        Value::GpuTensor(upper.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(crate::runtime_error::error_has_control_cancellation_origin(
        &error
    ));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(lower.probe.retirements(), 0);
    assert_eq!(upper.probe.retirements(), 0);
    lower.handle.release().unwrap();
    upper.handle.release().unwrap();
}

impl AccelProvider for RejectDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "scalar optimization admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("admission must not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(std::io::Error::other("scalar optimization fixture rejected transfer").into())
        })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(std::io::Error::other("scalar optimization fixture rejected transfer").into())
        })
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn canonical_nonlinear_family_preflights_outputs_and_preserves_failed_transfer_causes() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(RejectDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let mut resident = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![3.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    resident.handle = resident
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    // This explicit invocation owns its policy; the ambient thread guard
    // does not change a fresh RuntimeContext's compatibility profile.
    runtime.set_runmat_extensions_enabled(true);
    for (entry, maximum, output_error) in [
        (
            &runmat_builtins::FMINUNC_CATALOG_ENTRY,
            6,
            &runmat_builtins::FMINUNC_ERROR_INVALID_ARGUMENT,
        ),
        (
            &runmat_builtins::FSOLVE_CATALOG_ENTRY,
            5,
            &runmat_builtins::FSOLVE_ERROR_INVALID_ARGUMENT,
        ),
        (
            &runmat_builtins::LSQNONLIN_CATALOG_ENTRY,
            7,
            &runmat_builtins::LSQNONLIN_ERROR_TOO_MANY_OUTPUTS,
        ),
        (
            &runmat_builtins::LSQCURVEFIT_CATALOG_ENTRY,
            7,
            &runmat_builtins::LSQCURVEFIT_ERROR_TOO_MANY_OUTPUTS,
        ),
    ] {
        let mut args = vec![
            Value::FunctionHandle("sin".into()),
            Value::GpuTensor(resident.handle.clone()),
        ];
        if entry.identity == runmat_builtins::LSQCURVEFIT_CATALOG_ENTRY.identity {
            args.extend([Value::Num(1.0), Value::Num(1.0)]);
        }
        let before = provider.downloads.load(Ordering::SeqCst);
        let error = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            entry.identity.name,
            &args,
            maximum + 1,
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            output_error.identifier,
            "{}",
            entry.identity.name
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before);
        let failed = block_on(crate::call_builtin_async_with_outputs_in_context(
            &runtime,
            entry.identity.name,
            &args,
            1,
        ))
        .unwrap_err();
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            before + 1,
            "{}: {failed:?}",
            entry.identity.name
        );
        assert_eq!(failed.identifier(), Some("RunMat:gather:DownloadFailed"));
        assert_eq!(failed.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let mut source: &(dyn std::error::Error + 'static) = &failed;
        while !source.is::<std::io::Error>() {
            source = source
                .source()
                .expect("original transfer cause must survive solver mapping");
        }
        assert_eq!(
            source.to_string(),
            "scalar optimization fixture rejected transfer"
        );
        assert_eq!(resident.probe.retirements(), 0);
    }
    resident.handle.release().unwrap();
}

#[test]
fn host_visible_scalar_errors_precede_resident_transfer_and_preserve_borrowed_lease() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(RejectDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let mut real = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![0.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    real.handle = real
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let mut complex = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![0.0, 0.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    complex.handle = complex
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    let mut unavailable_upper = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    unavailable_upper.handle = unavailable_upper
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        .unwrap();
    unavailable_upper.probe.mark_unavailable();
    let function = || Value::FunctionHandle("sin".into());
    let resident = || Value::GpuTensor(real.handle.clone());

    let bad_upper_lease = block_on(super::fminbnd::fminbnd_builtin(
        Value::FunctionHandle("sin".into()),
        resident(),
        Value::GpuTensor(unavailable_upper.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        bad_upper_lease.gpu_gather_retry(),
        crate::GpuGatherRetry::Never
    );
    assert!(bad_upper_lease.source.is_some());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(real.probe.retirements(), 0);

    let invalid_callback = block_on(super::fminbnd::fminbnd_builtin(
        Value::Num(3.0),
        resident(),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        invalid_callback.identifier(),
        Some("RunMat:FevalFunctionValueUnsupported")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(real.probe.retirements(), 0);

    let invalid_callback_shape = block_on(super::fzero::fzero_builtin(
        Value::StringArray(
            runmat_value::StringArray::new_2d(vec!["sin".into(), "cos".into()], 1, 2).unwrap(),
        ),
        resident(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        invalid_callback_shape.identifier(),
        Some("RunMat:FevalHandleShapeInvalid")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(real.probe.retirements(), 0);

    let bad_upper = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Complex(1.0, 1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(bad_upper.identifier(), Some("RunMat:fminbnd:InvalidInput"));
    assert_eq!(bad_upper.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let wide_upper = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(wide_upper.identifier(), Some("RunMat:fminbnd:InvalidInput"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    let quad_bad_tolerance = block_on(super::quad::quad_builtin(
        function(),
        resident(),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-1.0)]),
    ))
    .unwrap_err();
    assert_eq!(
        quad_bad_tolerance.identifier(),
        Some("RunMat:quad:InvalidArgument")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(real.probe.retirements(), 0);

    let quad_bad_later_bound = block_on(super::quad::quad_builtin(
        function(),
        Value::GpuTensor(unavailable_upper.handle.clone()),
        Value::Complex(1.0, 1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        quad_bad_later_bound.identifier(),
        Some("RunMat:quad:InvalidInput")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(unavailable_upper.probe.retirements(), 0);

    let bad_initial = block_on(super::fzero::fzero_builtin(
        function(),
        Value::GpuTensor(complex.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(bad_initial.identifier(), Some("RunMat:fzero:InvalidInput"));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

    for (field, expected) in [
        ("TolX", "option TolX must be a real numeric scalar"),
        ("Display", "option Display must be a string"),
    ] {
        let mut opaque = runmat_value::UnboundedStructValueBuilder::new();
        opaque.insert("unselected_private_property", resident());
        let mut settings = runmat_value::UnboundedStructValueBuilder::new();
        settings.insert(field, Value::Struct(opaque.finish()));
        let bad_control = block_on(super::fminbnd::fminbnd_builtin(
            function(),
            resident(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
                settings.finish(),
            )]),
        ))
        .unwrap_err();
        assert_eq!(
            bad_control.identifier(),
            Some("RunMat:fminbnd:InvalidArgument")
        );
        assert!(bad_control.message().contains(expected));
        assert!(!bad_control
            .message()
            .contains("unselected_private_property"));
    }
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(real.probe.retirements(), 0);

    let mut opaque = runmat_value::UnboundedStructValueBuilder::new();
    opaque.insert("unselected_private_property", resident());
    let options = Value::Cell(
        runmat_value::CellArray::new(vec![Value::Struct(opaque.finish())], 1, 1).unwrap(),
    );
    let bad_options = [
        block_on(super::fminbnd::fminbnd_builtin(
            function(),
            resident(),
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![options.clone()]),
        ))
        .unwrap_err(),
        block_on(super::fzero::fzero_builtin(
            function(),
            resident(),
            crate::builtins::common::test_support::builtin_arguments(vec![options]),
        ))
        .unwrap_err(),
    ];
    for (error, identifier) in bad_options.iter().zip([
        "RunMat:fminbnd:InvalidArgument",
        "RunMat:fzero:InvalidArgument",
    ]) {
        assert_eq!(error.identifier(), Some(identifier));
        assert!(error.message().contains("options must be a struct"));
        assert!(!error.message().contains("unselected_private_property"));
    }
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(real.probe.retirements(), 0);

    let _outputs = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let bad_outputs = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        bad_outputs.identifier(),
        Some("RunMat:fminbnd:TooManyOutputs")
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    drop(_outputs);

    // A primary-double caller owns no numeric/single extension authority.
    // Metadata-only refusal precedes provider lookup as well as transfer.
    {
        use super::common::constrained_numeric::{self as numeric, Geometry, Numeric, Policy};
        let strict = Policy {
            owner: runmat_builtins::FMINCON_CATALOG_ENTRY.identity,
            invalid: Some(&runmat_builtins::FMINCON_ERROR_INVALID_ARGUMENT),
            numeric: None,
            single: None,
            resident: None,
            sparse: None,
        };
        let mut single = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(vec![0.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        single.handle = single
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        let mut opaque = runmat_value::UnboundedStructValueBuilder::new();
        opaque.insert("resident", resident());
        let _absent = ThreadProviderGuard::set(None);
        for value in [
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Bool(false),
            Value::GpuTensor(single.handle.clone()),
            Value::Struct(opaque.finish()),
        ] {
            let error =
                numeric::preflight(&value, "primary value", Geometry::Scalar, &strict, None)
                    .unwrap_err();
            assert_eq!(error.identifier(), strict.invalid.unwrap().identifier);
        }
        let extended = Policy {
            invalid: None,
            numeric: Some(&runmat_builtins::FMINCON_NUMERIC_EXTENSION),
            single: Some(&runmat_builtins::FMINCON_SINGLE_EXTENSION),
            resident: Some(&runmat_builtins::FMINCON_RESIDENT_EXTENSION),
            ..strict
        };
        let error = numeric::preflight(
            &Value::GpuTensor(single.handle.clone()),
            "extended value",
            Geometry::Scalar,
            &extended,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::IS_UNDERLYING_TYPE_ERROR_PROVIDER_PAYLOAD.identifier
        );
        let mut cooperation = super::common::ScalarCooperation::for_builtin(strict.owner, None);
        assert!(numeric::preflight(
            &Value::Num(f64::NAN),
            "raw trial",
            Geometry::Scalar,
            &strict,
            None
        )
        .unwrap());
        assert!(
            matches!(block_on(numeric::materialize_raw(Value::Num(f64::NAN),
            "raw trial", &strict, &mut cooperation)).unwrap(),
            Numeric::Real { values, .. } if values[0].is_nan())
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(single.probe.retirements(), 0);
        single.handle.release().unwrap();
    }

    let failed_transfer = block_on(super::fminbnd::fminbnd_builtin(
        function(),
        resident(),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(
        failed_transfer.identifier(),
        Some("RunMat:gather:DownloadFailed")
    );
    assert_eq!(
        failed_transfer.gpu_gather_retry,
        crate::GpuGatherRetry::Never
    );
    let mut source: &(dyn std::error::Error + 'static) = &failed_transfer;
    while !source.is::<std::io::Error>() {
        source = source.source().expect("original provider cause retained");
    }
    assert_eq!(
        source.to_string(),
        "scalar optimization fixture rejected transfer"
    );
    assert_eq!(real.probe.retirements(), 0);
    assert_eq!(complex.probe.retirements(), 0);
    real.handle.release().unwrap();
    complex.handle.release().unwrap();
}
