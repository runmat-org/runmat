use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::{self, ThreadId};

use runmat_extension_abi::{
    RunMatBufferLease, RunMatBufferLeaseHandle, RunMatExtensionCapabilities, RunMatHostVTable,
    RunMatValueHandle, RUNMAT_EXTENSION_ABI_VERSION,
};
use runmat_value::Value;

use super::abi;
use super::buffer::BufferLease;

#[derive(Debug)]
struct ValueEntry {
    generation: u64,
    retain_count: usize,
    value: Value,
}

#[derive(Debug)]
struct OriginState {
    next_resource: u64,
    values: BTreeMap<u64, ValueEntry>,
}

#[derive(Debug)]
struct LeaseState {
    next_resource: u64,
    leases: BTreeMap<u64, (u64, Box<BufferLease>)>,
}

/// C-facing context whose thread-safe lease registry is independent from the
/// origin-thread RunMat value registry.
#[derive(Debug)]
pub(super) struct ExtensionHostContext {
    origin_thread: ThreadId,
    origin_state: AtomicUsize,
    leases: Mutex<LeaseState>,
}

/// Runtime-owned registry behind the public extension host vtable.
///
/// RunMat values remain on their originating runtime thread. A successful
/// buffer borrow creates a separate read-only lease containing only canonical,
/// thread-safe host-buffer owners. Extensions may read that view and release
/// its lease from a worker thread without moving a general `Value` or a GC
/// handle across threads.
#[derive(Debug)]
pub struct ExtensionHost {
    origin: RefCell<OriginState>,
    context: ExtensionHostContext,
    _pinned: PhantomPinned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BufferBorrowError {
    AffinityViolation,
    StaleHandle,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ValueAccessError {
    AffinityViolation,
    StaleHandle,
}

impl ExtensionHost {
    pub fn new() -> Pin<Box<Self>> {
        let host = Box::pin(Self {
            origin: RefCell::new(OriginState {
                next_resource: 1,
                values: BTreeMap::new(),
            }),
            context: ExtensionHostContext {
                origin_thread: thread::current().id(),
                origin_state: AtomicUsize::new(0),
                leases: Mutex::new(LeaseState {
                    next_resource: 1,
                    leases: BTreeMap::new(),
                }),
            },
            _pinned: PhantomPinned,
        });
        let state = std::ptr::from_ref(&host.as_ref().get_ref().origin) as usize;
        host.context.origin_state.store(state, Ordering::Release);
        host
    }

    pub fn register_value(self: Pin<&Self>, value: Value) -> RunMatValueHandle {
        assert_eq!(
            thread::current().id(),
            self.context.origin_thread,
            "extension values must be registered on their runtime thread"
        );
        let mut state = self.origin.borrow_mut();
        let resource = state.take_resource();
        let generation = 1;
        state.values.insert(
            resource,
            ValueEntry {
                generation,
                retain_count: 1,
                value,
            },
        );
        RunMatValueHandle {
            resource,
            generation,
        }
    }

    pub fn host_vtable(self: Pin<&Self>) -> RunMatHostVTable {
        RunMatHostVTable {
            abi_version: RUNMAT_EXTENSION_ABI_VERSION,
            struct_size: std::mem::size_of::<RunMatHostVTable>(),
            capabilities: RunMatExtensionCapabilities::READ_VALUE
                | RunMatExtensionCapabilities::BUFFER_LEASES,
            context: std::ptr::from_ref(&self.context).cast_mut().cast(),
            retain_value: Some(abi::retain_value),
            release_value: Some(abi::release_value),
            value_kind: Some(abi::value_kind),
            borrow_buffer: None,
            invoke_callback: None,
            is_cancelled: None,
            borrow_buffer_lease: Some(abi::borrow_buffer_lease),
            release_buffer: Some(abi::release_buffer),
        }
    }
}

impl Drop for ExtensionHost {
    fn drop(&mut self) {
        self.context.origin_state.store(0, Ordering::Release);
    }
}

impl ExtensionHostContext {
    pub(super) fn with_value<R>(
        &self,
        handle: RunMatValueHandle,
        operation: impl FnOnce(&Value) -> R,
    ) -> Result<R, ValueAccessError> {
        let state = self.origin_state()?;
        let state = state.borrow();
        state
            .value(handle)
            .map(operation)
            .ok_or(ValueAccessError::StaleHandle)
    }

    pub(super) fn retain(&self, handle: RunMatValueHandle) -> Result<(), ValueAccessError> {
        let state = self.origin_state()?;
        if state.borrow_mut().retain(handle) {
            Ok(())
        } else {
            Err(ValueAccessError::StaleHandle)
        }
    }

    pub(super) fn release_value(&self, handle: RunMatValueHandle) -> Result<(), ValueAccessError> {
        let state = self.origin_state()?;
        if state.borrow_mut().release_value(handle) {
            Ok(())
        } else {
            Err(ValueAccessError::StaleHandle)
        }
    }

    pub(super) fn borrow_buffer(
        &self,
        handle: RunMatValueHandle,
    ) -> Result<RunMatBufferLease, BufferBorrowError> {
        let lease = self
            .with_value(handle, BufferLease::from_value)
            .map_err(|error| match error {
                ValueAccessError::AffinityViolation => BufferBorrowError::AffinityViolation,
                ValueAccessError::StaleHandle => BufferBorrowError::StaleHandle,
            })?
            .map_err(|()| BufferBorrowError::Unsupported)?;
        let mut state = self
            .leases
            .lock()
            .map_err(|_| BufferBorrowError::Unsupported)?;
        let resource = state.take_resource();
        let generation = 1;
        state.leases.insert(resource, (generation, Box::new(lease)));
        let (_, lease) = state.leases.get(&resource).expect("inserted lease");
        Ok(RunMatBufferLease {
            view: lease.view(),
            handle: RunMatBufferLeaseHandle {
                resource,
                generation,
            },
        })
    }

    pub(super) fn release_buffer(&self, handle: RunMatBufferLeaseHandle) -> bool {
        let Ok(mut state) = self.leases.lock() else {
            return false;
        };
        if state
            .leases
            .get(&handle.resource)
            .is_none_or(|(generation, _)| *generation != handle.generation)
        {
            return false;
        }
        state.leases.remove(&handle.resource);
        true
    }

    fn origin_state(&self) -> Result<&RefCell<OriginState>, ValueAccessError> {
        if thread::current().id() != self.origin_thread {
            return Err(ValueAccessError::AffinityViolation);
        }
        let address = self.origin_state.load(Ordering::Acquire);
        if address == 0 {
            return Err(ValueAccessError::StaleHandle);
        }
        // SAFETY: `ExtensionHost` pins this `RefCell`, publishes its address
        // after construction, clears it before destruction, and permits access
        // only from the owning thread.
        Ok(unsafe { &*(address as *const RefCell<OriginState>) })
    }
}

impl OriginState {
    fn value(&self, handle: RunMatValueHandle) -> Option<&Value> {
        self.values
            .get(&handle.resource)
            .filter(|entry| entry.generation == handle.generation)
            .map(|entry| &entry.value)
    }

    fn retain(&mut self, handle: RunMatValueHandle) -> bool {
        let Some(entry) = self.values.get_mut(&handle.resource) else {
            return false;
        };
        if entry.generation != handle.generation {
            return false;
        }
        let Some(retain_count) = entry.retain_count.checked_add(1) else {
            return false;
        };
        entry.retain_count = retain_count;
        true
    }

    fn release_value(&mut self, handle: RunMatValueHandle) -> bool {
        let Some(entry) = self.values.get_mut(&handle.resource) else {
            return false;
        };
        if entry.generation != handle.generation || entry.retain_count == 0 {
            return false;
        }
        entry.retain_count -= 1;
        if entry.retain_count == 0 {
            self.values.remove(&handle.resource);
        }
        true
    }

    fn take_resource(&mut self) -> u64 {
        let resource = self.next_resource;
        self.next_resource = self
            .next_resource
            .checked_add(1)
            .expect("extension value identity exhausted");
        resource
    }
}

impl LeaseState {
    fn take_resource(&mut self) -> u64 {
        let resource = self.next_resource;
        self.next_resource = self
            .next_resource
            .checked_add(1)
            .expect("extension buffer-lease identity exhausted");
        resource
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_extension_abi::{RunMatBufferLease, RunMatStatusCode, RunMatValueKind};

    #[test]
    fn buffer_lease_retains_the_canonical_allocation_independently() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BufferLease>();
        assert_send_sync::<ExtensionHostContext>();

        let tensor = runmat_value::Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        // SAFETY: the address is observed only while a Tensor or lease owns it.
        let expected = unsafe { tensor.host_buffer().foreign_data_pointer() }
            .cast::<u8>()
            .cast_const();
        let host = ExtensionHost::new();
        let handle = host.as_ref().register_value(Value::Tensor(tensor));
        let table = host.as_ref().host_vtable();

        let mut kind = RunMatValueKind::Unknown;
        let status = unsafe { table.value_kind.unwrap()(table.context, handle, &mut kind) };
        assert_eq!(status, RunMatStatusCode::Ok);
        assert_eq!(kind, RunMatValueKind::Dense);

        let mut lease = RunMatBufferLease {
            view: runmat_extension_abi::RunMatBufferView {
                data: std::ptr::null(),
                byte_length: 0,
                shape: std::ptr::null(),
                rank: 0,
                element_type: runmat_extension_abi::RunMatElementType::Unknown,
                flags: 0,
            },
            handle: RunMatBufferLeaseHandle::INVALID,
        };
        let status =
            unsafe { table.borrow_buffer_lease.unwrap()(table.context, handle, &mut lease) };
        assert_eq!(status, RunMatStatusCode::Ok);
        assert_eq!(lease.view.data, expected);
        assert_eq!(lease.view.byte_length, 2 * std::mem::size_of::<f64>());
        assert_eq!(lease.view.rank, 2);
        // SAFETY: the live lease guarantees two readable double values.
        assert_eq!(
            unsafe { std::slice::from_raw_parts(lease.view.data.cast::<f64>(), 2) },
            &[3.0, 4.0]
        );

        assert_eq!(
            unsafe { table.release_value.unwrap()(table.context, handle) },
            RunMatStatusCode::Ok
        );
        // SAFETY: releasing the value does not end the independent lease.
        assert_eq!(unsafe { *lease.view.data.cast::<f64>() }, 3.0);

        let context = table.context as usize;
        let lease_handle = lease.handle;
        let release = table.release_buffer.unwrap();
        let status = std::thread::spawn(move || unsafe {
            release(context as *mut std::ffi::c_void, lease_handle)
        })
        .join()
        .unwrap();
        assert_eq!(status, RunMatStatusCode::Ok);
        assert_eq!(
            unsafe { table.release_buffer.unwrap()(table.context, lease.handle) },
            RunMatStatusCode::StaleHandle
        );
    }

    #[test]
    fn value_callbacks_reject_worker_threads_without_touching_runtime_values() {
        let host = ExtensionHost::new();
        let handle = host.as_ref().register_value(Value::Num(1.0));
        let table = host.as_ref().host_vtable();
        let context = table.context as usize;
        let kind = table.value_kind.unwrap();
        let status = std::thread::spawn(move || {
            let mut value_kind = RunMatValueKind::Unknown;
            unsafe { kind(context as *mut std::ffi::c_void, handle, &mut value_kind) }
        })
        .join()
        .unwrap();
        assert_eq!(status, RunMatStatusCode::AffinityViolation);
    }
}
