//! Shared applicability: these witnesses cover all four constant constructors.
use super::*;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct GuardProvider {
    forbid_hooks: bool,
    alias: bool,
    calls: AtomicUsize,
    frees: AtomicUsize,
}

impl GuardProvider {
    fn new(forbid_hooks: bool, alias: bool) -> &'static Self {
        Box::leak(Box::new(Self {
            forbid_hooks,
            alias,
            calls: AtomicUsize::new(0),
            frees: AtomicUsize::new(0),
        }))
    }
    fn result(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        assert!(
            !self.forbid_hooks,
            "prototype owner rejection must precede provider calls"
        );
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut handle = GpuTensorHandle::new(
            shape.to_vec(),
            self.device_id(),
            if self.alias { 100 } else { 200 },
        );
        handle.descriptor.storage = Some(GpuTensorStorage::Real);
        // Deliberately absent durable element type, regardless of hint registries.
        handle.descriptor.element_type = None;
        Ok(handle)
    }
}

impl AccelProvider for GuardProvider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("unexpected download")
    }
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn device_info(&self) -> String {
        "constructor contract fixture".into()
    }
    fn device_id(&self) -> u32 {
        u32::MAX - 37
    }
    fn zeros(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        self.result(shape)
    }
    fn ones(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        self.result(shape)
    }
    fn fill(&self, shape: &[usize], _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.result(shape)
    }
    fn fill_like(&self, handle: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.result(&handle.shape)
    }
}

fn call(name: &str, args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match name {
        "true" => block_on(true_builtin(args)),
        "false" => block_on(false_builtin(args)),
        "inf" => block_on(crate::builtins::array::creation::inf::inf_builtin(args)),
        "nan" => block_on(crate::builtins::array::creation::nan::nan_builtin(args)),
        _ => unreachable!(),
    }
}

#[test]
fn all_constructors_reject_unknown_prototype_owner_before_provider_calls() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(true, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in ["true", "false", "inf", "nan"] {
        let prototype = GpuTensorHandle::new(vec![1, 1], u32::MAX, 100);
        let error = call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("unknown prototype owner");
        assert!(error.message().contains("provider"), "{name}: {error}");
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.frees.load(Ordering::SeqCst), 0);
}

#[test]
fn all_constructors_reject_extra_outputs_before_provider_calls() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(true, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _outputs = crate::output_count::push_output_count(Some(2));
    for name in ["true", "false", "inf", "nan"] {
        let prototype = GpuTensorHandle::new(vec![1, 1], provider.device_id(), 100);
        let error = call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("extra output");
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(provider.frees.load(Ordering::SeqCst), 0);
}

#[test]
fn all_constructors_accept_discarded_scalar_output() {
    let _outputs = crate::output_count::push_output_count(Some(0));
    for name in ["true", "false", "inf", "nan"] {
        call(name, vec![]).expect("discarded output");
    }
}

#[test]
fn logical_constructors_reject_missing_physical_element_type_before_relabelling() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(false, false);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in ["true", "false"] {
        let prototype = GpuTensorHandle::new(vec![1, 1], provider.device_id(), 100);
        let error = call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("missing durable element type");
        assert!(error.message().contains("invalid constructor result"));
        let rejected = GpuTensorHandle::new(vec![1, 1], provider.device_id(), 200);
        assert!(!runmat_accelerate_api::handle_is_logical(&rejected));
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.frees.load(Ordering::SeqCst), 2);
}

#[test]
fn logical_constructors_preserve_borrowed_alias_metadata_and_storage() {
    let _lock = test_support::accel_test_lock();
    let provider = GuardProvider::new(false, true);
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    for name in ["true", "false"] {
        let prototype = GpuTensorHandle::new(vec![1, 1], provider.device_id(), 100);
        let retained = prototype.clone();
        call(name, vec![Value::from("like"), Value::GpuTensor(prototype)])
            .expect_err("borrowed alias");
        assert!(!runmat_accelerate_api::handle_is_logical(&retained));
    }
    assert_eq!(provider.frees.load(Ordering::SeqCst), 0);
}

#[test]
fn logical_constructors_upload_native_single_when_f32_provider_has_no_fill_hooks() {
    test_support::with_f32_test_provider(|provider| {
        assert!(provider.ones(&[2, 3]).is_err());
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
            assert_eq!(handle.shape, vec![2, 3]);
            assert_eq!(
                handle.descriptor.element_type,
                Some(runmat_accelerate_api::NumericElementType::F32)
            );
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            assert!(
                crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("logical gather");
            assert_eq!(gathered.materialize_f64(), vec![expected; 6]);
        }
        assert!(!runmat_accelerate_api::handle_is_logical(&prototype));
    });
}
