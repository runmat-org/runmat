use super::*;
use runmat_accelerate_api::AccelProvider;

struct PreferenceDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: std::sync::atomic::AtomicUsize,
    abandoned: std::sync::atomic::AtomicUsize,
    pending: bool,
}

struct PendingPreferenceDownload<'a>(&'a std::sync::atomic::AtomicUsize);

impl Drop for PendingPreferenceDownload<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl PreferenceDownloadProvider {
    fn new(pending: bool) -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: std::sync::atomic::AtomicUsize::new(0),
            abandoned: std::sync::atomic::AtomicUsize::new(0),
            pending,
        }
    }

    fn resident(&self) -> Value {
        Value::GpuTensor(
            self.upload(&runmat_accelerate_api::HostTensorView {
                data: &[7.0],
                shape: &[1, 1],
            })
            .expect("authentic resident input"),
        )
    }

    fn downloads(&self) -> usize {
        self.downloads.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl runmat_accelerate_api::AccelProvider for PreferenceDownloadProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.inner.precision()
    }

    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView<'_>,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.pending {
            Box::pin(async move {
                let _active = PendingPreferenceDownload(&self.abandoned);
                std::future::pending().await
            })
        } else {
            self.inner.download_numeric(handle)
        }
    }
}

fn poll_preference_once<F: std::future::Future>(
    future: std::pin::Pin<&mut F>,
) -> std::task::Poll<F::Output> {
    let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
    std::future::Future::poll(future, &mut context)
}

#[test]
fn nested_resident_default_rejects_before_service_or_provider_effects() {
    let default = Value::Cell(
        CellArray::new(vec![unregistered_resident_value()], 1, 1).expect("nested default"),
    );
    let error = run(getpref_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("nestedDefault".into()),
            Value::String("value".into()),
            default,
        ]),
    ))
    .expect_err("default intent rejects before service lookup or gather");
    assert!(error.message().contains("provider-resident default"));
}

#[test]
fn failed_nested_setpref_gather_leaves_the_application_store_unchanged() {
    let runtime = preference_context();
    let group = Value::String("nestedSetprefFailure".into());
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group.clone(),
            Value::String("value".into()),
            Value::Cell(
                CellArray::new(vec![unregistered_resident_value()], 1, 1).expect("nested value"),
            ),
        ]),
    ))
    .expect_err("unregistered allocation cannot materialize");
    assert!(error.message().contains("provider") || error.message().contains("allocation"));
    assert_eq!(
        run(ispref_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![group])
        ))
        .expect("no partial setting"),
        Value::Bool(false)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_nested_host_sibling_rejects_before_resident_download() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(PreferenceDownloadProvider::new(false)));
    unsafe { runmat_accelerate_api::register_provider(owner) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let runtime = preference_context();
    let value = Value::Cell(
        CellArray::new(
            vec![owner.resident(), Value::FunctionHandle("local".into())],
            1,
            2,
        )
        .expect("nested preference"),
    );
    let args = [Value::from("invalidSibling"), Value::from("payload"), value];
    let error = run(
        crate::dispatcher::call_builtin_async_with_outputs_in_context(
            &runtime, "setpref", &args, 0,
        ),
    )
    .expect_err("host callable must fail before resident sibling download");
    assert!(error.message().contains("callable"));
    assert_eq!(owner.downloads(), 0);
    assert_eq!(
        run(ispref_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "invalidSibling"
            )]),
        ))
        .expect("no application-store mutation"),
        Value::Bool(false)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn managed_host_sibling_is_not_admitted_beside_a_resident_value() {
    runmat_gc::gc_test_context(|| {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let owner = Box::leak(Box::new(PreferenceDownloadProvider::new(false)));
        unsafe { runmat_accelerate_api::register_provider(owner) };
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let rooted = runmat_gc::gc_allocate_rooted(Value::Num(3.0)).expect("managed target");
        let managed = Value::HandleObject(runmat_value::HandleRef {
            class_name: "PreferenceInput".into(),
            target: rooted.handle(),
            valid: true,
        });
        let payload = Value::Cell(
            CellArray::new(vec![owner.resident(), managed], 1, 2).expect("nested input"),
        );
        let runtime = preference_context();
        let args = [
            Value::from("managedSibling"),
            Value::from("payload"),
            payload,
        ];
        let error = run(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime, "setpref", &args, 0,
            ),
        )
        .expect_err("managed handle is not durable even beside a resident value");
        assert!(error.message().contains("handle"));
        assert_eq!(owner.downloads(), 0);
        assert_eq!(
            run(ispref_builtin(
                Some(runtime),
                crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                    "managedSibling"
                )]),
            ))
            .expect("no mutation"),
            Value::Bool(false)
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn plain_object_array_resident_properties_are_materialized_before_durable_commit() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(PreferenceDownloadProvider::new(false)));
    unsafe { runmat_accelerate_api::register_provider(owner) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let runtime = preference_context();
    let mut object = runmat_value::ObjectInstance::new("PreferencePayload");
    object.replace_property_unbounded("sample".into(), owner.resident());
    let array =
        runmat_value::ObjectArray::from_objects("PreferencePayload", vec![object], vec![1, 1])
            .expect("plain object array");
    let args = [
        Value::from("residentObjectArray"),
        Value::from("payload"),
        Value::ObjectArray(array),
    ];
    run(
        crate::dispatcher::call_builtin_async_with_outputs_in_context(
            &runtime, "setpref", &args, 0,
        ),
    )
    .expect("explicit resident materialization");
    assert_eq!(owner.downloads(), 1);
    let output = run(getpref_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("residentObjectArray"),
            Value::from("payload"),
        ]),
    ))
    .expect("committed host object array");
    let Value::ObjectArray(array) = output else {
        panic!("object array retained");
    };
    let Value::Object(object) = &array.data()[0] else {
        panic!("plain object retained");
    };
    let Value::Tensor(sample) = &object.properties()["sample"] else {
        panic!("resident property became a host tensor");
    };
    assert_eq!(sample.as_f64_slice(), Some(&[7.0][..]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn direct_dispatch_abandonment_drops_pending_preference_download_without_mutation() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(PreferenceDownloadProvider::new(true)));
    unsafe { runmat_accelerate_api::register_provider(owner) };
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let runtime = preference_context();
    let args = [
        Value::from("pendingPreference"),
        Value::from("payload"),
        owner.resident(),
    ];
    let mut call = Box::pin(
        crate::dispatcher::call_builtin_async_with_outputs_in_context(
            &runtime, "setpref", &args, 0,
        ),
    );
    assert!(poll_preference_once(call.as_mut()).is_pending());
    assert_eq!(owner.downloads(), 1);
    drop(call);
    assert_eq!(
        owner.abandoned.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "abandoned dispatch dropped the in-flight provider operation"
    );
    assert_eq!(
        run(ispref_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "pendingPreference"
            )]),
        ))
        .expect("abandoned call did not mutate the store"),
        Value::Bool(false)
    );
}

#[test]
fn invalid_multi_value_shape_rejects_before_resident_download_or_store_mutation() {
    let runtime = preference_context();
    let group = Value::String("runmatMultiPreferenceAdmission".into());
    let names = Value::StringArray(
        runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2])
            .expect("names shape"),
    );
    let wrong_shape = Value::Cell(
        CellArray::new_with_shape(
            vec![unregistered_resident_value(), Value::Num(2.0)],
            vec![2, 1],
        )
        .expect("wrong shape"),
    );
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group.clone(),
            names.clone(),
            wrong_shape,
        ]),
    ))
    .expect_err("shape must reject before any gather");
    assert!(error.message().contains("matching shape"));
    assert_eq!(
        run(ispref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![group.clone()])
        ))
        .expect("group query"),
        Value::Bool(false)
    );
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group,
            names,
            Value::Num(1.0),
        ]),
    ))
    .expect_err("scalar cannot fill multiple settings");
    assert!(error.message().contains("matching shape"));
}

#[test]
fn getpref_rejects_resident_controls_and_defaults_before_provider_access() {
    let runtime = preference_context();
    let resident = || unregistered_resident_value();
    let error = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            resident(),
            Value::String("name".into()),
        ]),
    ))
    .expect_err("resident group");
    assert!(error.message().contains("group must be"));

    let error = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefResident".into()),
            resident(),
        ]),
    ))
    .expect_err("resident preference name");
    assert!(error.message().contains("preference must be"));

    let error = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefResident".into()),
            Value::String("value".into()),
            resident(),
        ]),
    ))
    .expect_err("resident default");
    assert!(error.message().contains("provider-resident default"));
}
