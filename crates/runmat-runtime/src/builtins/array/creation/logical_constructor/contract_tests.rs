//! Shared applicability: these witnesses cover all four constant constructors.
use super::*;
use runmat_accelerate_api::{
    allocation::WeakGpuTensorHandle,
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NumericElementType,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct GuardProvider {
    forbid_hooks: bool,
    alias: bool,
    calls: AtomicUsize,
    fill_mode: AtomicUsize,
    uploads: AtomicUsize,
    owner: TestAllocationOwner,
    borrowed: std::sync::Mutex<Option<WeakGpuTensorHandle>>,
    rejected: std::sync::Mutex<Vec<RetirementProbe>>,
}

impl GuardProvider {
    fn new(forbid_hooks: bool, alias: bool) -> &'static Self {
        Box::leak(Box::new(Self {
            forbid_hooks,
            alias,
            calls: AtomicUsize::new(0),
            fill_mode: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            owner: TestAllocationOwner::new(),
            borrowed: std::sync::Mutex::new(None),
            rejected: std::sync::Mutex::new(Vec::new()),
        }))
    }
    fn result(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        assert!(
            !self.forbid_hooks,
            "prototype owner rejection must precede provider calls"
        );
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.alias {
            let handle = self
                .borrowed
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .acquire()
                .expect("borrowed prototype remains live");
            return Ok(handle.with_view(handle.view().clone())?);
        }
        // A real numeric allocation is an invalid logical-constructor result.
        let allocation = self.owner.allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.rejected.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}

impl AccelProvider for GuardProvider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("unexpected download")
    }
    fn device_info(&self) -> String {
        "constructor contract fixture".into()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn logical_fill(
        &self,
        shape: &[usize],
        _: bool,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        match self.fill_mode.load(Ordering::SeqCst) {
            1 => return Ok(None),
            2 => {
                return Err(
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "injected device allocation failure"
                    ))
                    .with_unproven_effects(),
                )
            }
            3 => {
                let output = self.result(shape).map_err(|source| {
                    runmat_accelerate_api::ProviderExecutionError::from(source)
                        .with_unproven_effects()
                })?;
                drop(output);
                return Err(runmat_accelerate_api::ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "injected unsupported after constructor allocation",
                    ),
                )
                .after_effect());
            }
            _ => {}
        }
        self.result(shape).map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn upload_logical_raw(
        &self,
        host: &runmat_accelerate_api::HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let allocation = self
            .owner
            .allocate_logical(host.data.to_vec(), host.shape.to_vec())
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?;
        self.rejected.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn zeros(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        self.result(shape)
    }
    fn ones(
        &self,
        shape: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = self.result(shape);
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn fill(
        &self,
        shape: &[usize],
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = self.result(shape);
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn fill_like(
        &self,
        handle: &GpuTensorHandle,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<GpuTensorHandle> = self.result(handle.shape());
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}

fn call(name: &str, args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match name {
        "true" => block_on(true_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        )),
        "false" => block_on(false_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        )),
        "inf" => block_on(crate::builtins::array::creation::inf::inf_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        )),
        "nan" => block_on(crate::builtins::array::creation::nan::nan_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        )),
        _ => unreachable!(),
    }
}

#[test]
fn all_constructors_reject_unknown_prototype_owner_before_provider_calls() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(true, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in ["true", "false", "inf", "nan"] {
        let prototype = TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
            .handle;
        let error = call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("unknown prototype owner");
        assert!(error.message().contains("provider"), "{name}: {error}");
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(provider.rejected.lock().unwrap().is_empty());
}

#[test]
fn all_constructors_reject_extra_outputs_before_provider_calls() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(true, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for name in ["true", "false", "inf", "nan"] {
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let prototype = allocation.handle;
        let error = call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("extra output");
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(provider.rejected.lock().unwrap().is_empty());
}

#[test]
fn all_constructors_accept_discarded_scalar_output() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    for name in ["true", "false", "inf", "nan"] {
        call(name, vec![]).expect("discarded output");
    }
}

#[test]
fn logical_fill_only_no_attempt_uses_owned_logical_upload() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(false, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let prototype = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    provider.fill_mode.store(1, Ordering::SeqCst);
    for (name, bit) in [("true", 1u8), ("false", 0u8)] {
        let result = call(
            name,
            vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::from("like"),
                Value::GpuTensor(prototype.handle.clone()),
            ],
        )
        .unwrap();
        let Value::GpuTensor(handle) = result else {
            panic!("resident logical output")
        };
        assert!(handle.view().is_logical());
        assert_eq!(handle.shape(), &[2, 3]);
        let probe = provider.rejected.lock().unwrap().last().unwrap().clone();
        probe
            .with_data(&handle, |data| {
                let runmat_accelerate_api::HostNumericDataView::U8(values) = data else {
                    panic!("canonical logical bytes")
                };
                assert_eq!(values, &[bit; 6]);
            })
            .unwrap();
        assert_eq!(probe.retirements(), 0);
        drop(handle);
        assert_eq!(probe.retirements(), 1);
    }
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    provider.fill_mode.store(2, Ordering::SeqCst);
    let error = call(
        "true",
        vec![
            Value::from("like"),
            Value::GpuTensor(prototype.handle.clone()),
        ],
    )
    .unwrap_err();
    assert!(error
        .message()
        .contains("injected device allocation failure"));
    assert_eq!(
        provider.uploads.load(Ordering::SeqCst),
        2,
        "real device failure must not trigger upload fallback"
    );
    assert_eq!(prototype.probe.retirements(), 0);
    provider.fill_mode.store(3, Ordering::SeqCst);
    for name in ["true", "false"] {
        let error = call(
            name,
            vec![
                Value::from("like"),
                Value::GpuTensor(prototype.handle.clone()),
            ],
        )
        .expect_err("raw Unsupported after allocation is terminal");
        assert!(error
            .message()
            .contains("injected unsupported after constructor allocation"));
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let mut source = std::error::Error::source(&error);
        let mut after_effect = false;
        while let Some(cause) = source {
            if let Some(provider) =
                cause.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
            {
                after_effect |= provider.is_after_effect();
            }
            source = cause.source();
        }
        assert!(after_effect, "allocation phase remains in the cause chain");
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        assert_eq!(prototype.probe.retirements(), 0);
        assert!(provider
            .rejected
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
    }
}

#[test]
fn logical_constructors_reject_numeric_producer_before_adoption() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(false, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in ["true", "false"] {
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let prototype = allocation.handle;
        let error = call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("numeric producer cannot become logical");
        assert!(error.message().contains("invalid constructor result"));
        assert_eq!(allocation.probe.retirements(), 1);
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let rejected = provider.rejected.lock().unwrap();
    assert_eq!(rejected.len(), 2);
    assert!(rejected.iter().all(|probe| probe.retirements() == 1));
}

#[test]
fn all_constructors_preserve_borrowed_alias_metadata_and_storage() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(false, true);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in ["true", "false", "inf", "nan"] {
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let prototype = allocation.handle;
        *provider.borrowed.lock().unwrap() = Some(prototype.downgrade());
        let retained = prototype.clone();
        call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("borrowed alias");
        assert!(!runmat_accelerate_api::handle_is_logical(&retained));
        assert_eq!(allocation.probe.retirements(), 0);
        drop(retained);
        assert_eq!(allocation.probe.retirements(), 1);
    }
    assert!(provider.rejected.lock().unwrap().is_empty());
}

#[test]
fn logical_constructors_upload_native_logical_when_f32_provider_has_no_fill_hooks() {
    test_support::with_native_logical_test_provider(|provider| {
        assert!(provider.ones(&[2, 3]).unwrap().is_none());
        assert!(provider.zeros(&[2, 3]).is_err());
        let host = Tensor::from_f32(vec![7.0], vec![1, 1]).expect("single prototype");
        let prototype = crate::builtins::common::gpu_helpers::upload_tensor(provider, &host)
            .expect("native single prototype upload");
        for (name, expected) in [("true", 1.0), ("false", 0.0)] {
            let output = call(
                name,
                vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from("like"),
                    Value::GpuTensor(prototype.clone()),
                ],
            )
            .expect("single-only upload fallback");
            let Value::GpuTensor(handle) = output else {
                panic!("expected resident logical output");
            };
            assert_eq!(handle.shape(), vec![2, 3]);
            assert_eq!(
                handle.physical().element_type(),
                runmat_accelerate_api::NumericElementType::U8
            );
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            assert!(
                crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(
                        owner, provider
                    ))
            );
            let lease = handle.allocation_lease().downgrade();
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("logical gather");
            assert_eq!(gathered.materialize_f64(), vec![expected; 6]);
            assert!(lease.acquire().is_none());
        }
        assert!(!runmat_accelerate_api::handle_is_logical(&prototype));
    });
}
