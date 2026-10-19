use super::*;
use crate::*;

struct Provider(u32);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0
    }
    fn device_info(&self) -> String {
        "worker registry fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused") })
    }
}
static FIRST: Provider = Provider(91);
static SECOND: Provider = Provider(92);

#[test]
fn provider_instance_requires_namespace_and_facade_not_vtable_identity() {
    fn erase(provider: &Provider) -> &dyn AccelProvider {
        provider
    }
    let first = Provider(next_device_id());
    let distinct = Provider(first.device_id());
    let other_namespace = Provider(next_device_id());
    let direct: &dyn AccelProvider = &first;
    assert!(same_provider_instance(direct, erase(&first)));
    assert!(same_provider_instance(erase(&first), direct));
    // Deliberately do not register these duplicate-namespace objects: the
    // registry forbids ambiguous live owners. Identity still rejects them.
    assert!(!same_provider_instance(&first, &distinct));
    assert!(!same_provider_instance(&first, &other_namespace));
}

#[test]
fn same_data_address_does_not_override_a_different_namespace() {
    #[repr(C)]
    struct Facade {
        inner: Provider,
        namespace: u32,
    }
    impl AccelProvider for Facade {
        fn device_id(&self) -> u32 {
            self.namespace
        }
        fn device_info(&self) -> String {
            "outer fixture facade".into()
        }
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(host)
        }
        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.inner.download(handle)
        }
    }
    let facade = Facade {
        inner: Provider(next_device_id()),
        namespace: next_device_id(),
    };
    let outer: &dyn AccelProvider = &facade;
    let inner: &dyn AccelProvider = &facade.inner;
    assert!(std::ptr::addr_eq(outer, inner));
    assert!(!same_provider_instance(outer, inner));
}

#[test]
fn worker_registry_state_and_default_reset_do_not_cross_owners() {
    let mut first = Registry::default();
    let mut second = Registry::default();
    first.default = Some(&FIRST);
    first.devices.insert(91, &FIRST);
    second.default = Some(&SECOND);
    second.devices.insert(92, &SECOND);
    first.default = None;
    assert!(
        first.devices.contains_key(&91),
        "selection reset preserves live owner addressability"
    );
    assert!(!first.devices.contains_key(&92));
    assert!(!second.devices.contains_key(&91));
    assert_eq!(second.default.unwrap().device_id(), 92);
}

#[test]
fn nested_thread_selection_restores_without_mutating_registry() {
    let previous = crate::current_thread_provider();
    {
        let _first = ThreadProviderGuard::set(Some(&FIRST));
        assert_eq!(crate::current_thread_provider().unwrap().device_id(), 91);
        {
            let _second = ThreadProviderGuard::set(Some(&SECOND));
            assert_eq!(crate::current_thread_provider().unwrap().device_id(), 92);
        }
        assert_eq!(crate::current_thread_provider().unwrap().device_id(), 91);
    }
    assert_eq!(
        crate::current_thread_provider().map(AccelProvider::device_id),
        previous.map(AccelProvider::device_id)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_authorities_remain_send_sync_and_thread_selection_is_isolated() {
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<GpuTensorHandle>();
    require_send_sync::<std::sync::Arc<dyn AccelProvider>>();
    let _selected = ThreadProviderGuard::set(Some(&FIRST));
    std::thread::spawn(|| {
        assert!(crate::current_thread_provider().is_none());
        let _selected = ThreadProviderGuard::set(Some(&SECOND));
        assert_eq!(crate::current_thread_provider().unwrap().device_id(), 92);
    })
    .join()
    .unwrap();
    assert_eq!(crate::current_thread_provider().unwrap().device_id(), 91);
}

#[cfg(target_arch = "wasm32")]
#[test]
fn browser_provider_can_be_thread_affine_without_unsafe_markers() {
    struct Local(std::rc::Rc<()>);
    impl AccelProvider for Local {
        fn device_info(&self) -> String {
            format!("local refs {}", std::rc::Rc::strong_count(&self.0))
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("unused")
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("unused") })
        }
    }
    let provider = Local(std::rc::Rc::new(()));
    assert!(provider.device_info().contains("local refs"));
}
