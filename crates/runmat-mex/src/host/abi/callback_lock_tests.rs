//! Host callbacks can inspect their invocation and register a nested service.
use super::*;
use std::cell::Cell;
use std::rc::Weak;

struct Probe {
    host: Weak<MexCallState>,
    calls: Cell<usize>,
}

impl Drop for Probe {
    fn drop(&mut self) {
        if let Some(host) = self.host.upgrade() {
            assert!(
                host.inner.try_lock().is_ok(),
                "service retirement holds the MEX arena lock"
            );
        }
    }
}

impl Probe {
    fn check(&self) {
        let host = self.host.upgrade().unwrap();
        assert!(
            host.inner.try_lock().is_ok(),
            "callback holds the MEX arena lock"
        );
        // A nested module registers its own origin service. The registry's
        // RefCell borrow must also have ended before calling into this host.
        let (guard, _) = LocalBoundaryServiceGuard::register(Rc::new(Self {
            host: Rc::downgrade(&host),
            calls: Cell::new(0),
        }));
        drop(guard);
        self.calls.set(self.calls.get() + 1);
    }
}

impl MexBoundaryHostServices for Probe {
    fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
        self.check();
        Ok(())
    }
    fn call(
        &self,
        _: &str,
        arguments: Vec<MxArray>,
        outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        self.check();
        assert_eq!(outputs, arguments.len());
        Ok(arguments)
    }
    fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
        self.check();
        Ok(None)
    }
    fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
        self.check();
        Ok(())
    }
    fn get_object_property(
        &self,
        object: MxArray,
        _: usize,
        _: &str,
    ) -> Result<MxArray, MexDiagnostic> {
        self.check();
        Ok(object)
    }
    fn set_object_property(
        &self,
        object: MxArray,
        _: usize,
        _: &str,
        _: MxArray,
    ) -> Result<MxArray, MexDiagnostic> {
        self.check();
        Ok(object)
    }
}

#[test]
fn synchronous_host_callbacks_release_arena_and_registry_borrows() {
    let host = Rc::new(MexCallState::new(MxApiMode::InterleavedComplex));
    let services = Rc::new(Probe {
        host: Rc::downgrade(&host),
        calls: Cell::new(0),
    });
    let (guard, slot) = LocalBoundaryServiceGuard::register(services.clone());
    {
        let mut state = host.lock().unwrap();
        state.local_service_guard = Some(guard);
        state.services = slot;
    }
    let api = host.host_api();
    let name = CString::new("fixture").unwrap();
    let workspace = CString::new("base").unwrap();
    let input = unsafe { create_double_scalar(api.host, 7.0) };
    let object = host
        .lock()
        .unwrap()
        .mx
        .allocate(MxArray::handle(crate::MxHandleToken {
            resource: 1,
            generation: 1,
            class_name: runmat_types::ClassIdentity::new("FixtureHandle").unwrap(),
        }));
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { eval(api.host, name.as_ptr()) }, 0);
    assert_eq!(
        unsafe {
            call(
                api.host,
                name.as_ptr(),
                1,
                &mut output,
                1,
                &input.cast_const(),
            )
        },
        0
    );
    assert!(!output.is_null());
    assert!(unsafe { get_variable(api.host, workspace.as_ptr(), name.as_ptr()) }.is_null());
    assert_eq!(
        unsafe { put_variable(api.host, workspace.as_ptr(), name.as_ptr(), input) },
        0
    );
    assert!(!unsafe { get_property(api.host, object, 0, name.as_ptr()) }.is_null());
    let property_value = unsafe { create_double_scalar(api.host, 9.0) };
    assert_eq!(
        unsafe { set_property(api.host, object, 0, name.as_ptr(), property_value) },
        0
    );
    assert_eq!(services.calls.get(), 6);
    assert!(host.lock().unwrap().error.is_none());
    drop(services);
    host.set_services(Rc::new(crate::UnavailableMexHostServices));
    host.finish_call();
}
