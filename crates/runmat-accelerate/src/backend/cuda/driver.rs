use std::ffi::{c_char, c_void, CStr};

use anyhow::{Context, Result};
use libloading::Library;

pub(super) type DevicePointer = u64;
type Device = i32;
type ContextHandle = *mut c_void;
type ResultCode = i32;

const SUCCESS: ResultCode = 0;
const MEMORY_TYPE_DEVICE: i32 = 2;

type Init = unsafe extern "C" fn(u32) -> ResultCode;
type DeviceGet = unsafe extern "C" fn(*mut Device, i32) -> ResultCode;
type DeviceGetName = unsafe extern "C" fn(*mut c_char, i32, Device) -> ResultCode;
type PrimaryContextRetain = unsafe extern "C" fn(*mut ContextHandle, Device) -> ResultCode;
type PrimaryContextRelease = unsafe extern "C" fn(Device) -> ResultCode;
type ContextPush = unsafe extern "C" fn(ContextHandle) -> ResultCode;
type ContextPop = unsafe extern "C" fn(*mut ContextHandle) -> ResultCode;
type MemoryAllocate = unsafe extern "C" fn(*mut DevicePointer, usize) -> ResultCode;
type MemoryFree = unsafe extern "C" fn(DevicePointer) -> ResultCode;
type CopyHostToDevice = unsafe extern "C" fn(DevicePointer, *const c_void, usize) -> ResultCode;
type CopyDeviceToHost = unsafe extern "C" fn(*mut c_void, DevicePointer, usize) -> ResultCode;
type CopyDeviceToDevice = unsafe extern "C" fn(DevicePointer, DevicePointer, usize) -> ResultCode;
type MemorySet = unsafe extern "C" fn(DevicePointer, u8, usize) -> ResultCode;
type Copy2d = unsafe extern "C" fn(*const Copy2dParameters) -> ResultCode;
type Synchronize = unsafe extern "C" fn() -> ResultCode;
type ErrorString = unsafe extern "C" fn(ResultCode, *mut *const c_char) -> ResultCode;

#[repr(C)]
#[derive(Clone, Copy)]
struct Copy2dParameters {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: i32,
    src_host: *const c_void,
    src_device: DevicePointer,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: i32,
    dst_host: *mut c_void,
    dst_device: DevicePointer,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

pub(super) struct CudaDriver {
    _library: Library,
    init: Init,
    device_get: DeviceGet,
    device_get_name: DeviceGetName,
    primary_context_retain: PrimaryContextRetain,
    primary_context_release: PrimaryContextRelease,
    context_push: ContextPush,
    context_pop: ContextPop,
    memory_allocate: MemoryAllocate,
    memory_free: MemoryFree,
    copy_host_to_device: CopyHostToDevice,
    copy_device_to_host: CopyDeviceToHost,
    copy_device_to_device: CopyDeviceToDevice,
    memory_set: MemorySet,
    copy_2d: Copy2d,
    synchronize: Synchronize,
    error_string: ErrorString,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    pub(super) fn load() -> Result<Option<Self>> {
        let mut last_error = None;
        for candidate in library_candidates() {
            // SAFETY: CUDA's driver library is process-global; every resolved
            // symbol is retained together with the owning library handle.
            let library = match unsafe { Library::new(candidate) } {
                Ok(library) => library,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            // SAFETY: names and signatures below are from the stable CUDA
            // driver ABI. The library remains in the returned structure.
            let loaded = unsafe { Self::from_library(library) };
            return loaded.map(Some);
        }
        let _ = last_error;
        Ok(None)
    }

    unsafe fn from_library(library: Library) -> Result<Self> {
        macro_rules! symbol {
            ($name:literal, $type:ty) => {{
                // SAFETY: the caller established the CUDA driver library and
                // the requested symbol uses its documented ABI signature.
                *unsafe { library.get::<$type>(concat!($name, "\0").as_bytes()) }
                    .with_context(|| concat!("missing CUDA driver symbol ", $name))?
            }};
        }
        // Both spellings are exported across the supported driver range. The
        // `_v2` form is preferred, while the unsuffixed symbol preserves
        // compatibility with older drivers that implement the same signature.
        let primary_context_release = unsafe {
            library
                .get::<PrimaryContextRelease>(b"cuDevicePrimaryCtxRelease_v2\0")
                .or_else(|_| library.get::<PrimaryContextRelease>(b"cuDevicePrimaryCtxRelease\0"))
        }
        .map(|symbol| *symbol)
        .context("missing CUDA driver symbol cuDevicePrimaryCtxRelease")?;
        Ok(Self {
            init: symbol!("cuInit", Init),
            device_get: symbol!("cuDeviceGet", DeviceGet),
            device_get_name: symbol!("cuDeviceGetName", DeviceGetName),
            primary_context_retain: symbol!("cuDevicePrimaryCtxRetain", PrimaryContextRetain),
            primary_context_release,
            context_push: symbol!("cuCtxPushCurrent_v2", ContextPush),
            context_pop: symbol!("cuCtxPopCurrent_v2", ContextPop),
            memory_allocate: symbol!("cuMemAlloc_v2", MemoryAllocate),
            memory_free: symbol!("cuMemFree_v2", MemoryFree),
            copy_host_to_device: symbol!("cuMemcpyHtoD_v2", CopyHostToDevice),
            copy_device_to_host: symbol!("cuMemcpyDtoH_v2", CopyDeviceToHost),
            copy_device_to_device: symbol!("cuMemcpyDtoD_v2", CopyDeviceToDevice),
            memory_set: symbol!("cuMemsetD8_v2", MemorySet),
            copy_2d: symbol!("cuMemcpy2D_v2", Copy2d),
            synchronize: symbol!("cuCtxSynchronize", Synchronize),
            error_string: symbol!("cuGetErrorString", ErrorString),
            _library: library,
        })
    }

    pub(super) fn initialize(&self, ordinal: i32) -> Result<(i32, usize, String)> {
        self.call("cuInit", unsafe { (self.init)(0) })?;
        let mut device = 0;
        self.call("cuDeviceGet", unsafe {
            (self.device_get)(&mut device, ordinal)
        })?;
        let mut context = std::ptr::null_mut();
        self.call("cuDevicePrimaryCtxRetain", unsafe {
            (self.primary_context_retain)(&mut context, device)
        })?;
        let mut name = [0_i8; 256];
        self.call("cuDeviceGetName", unsafe {
            (self.device_get_name)(name.as_mut_ptr(), name.len() as i32, device)
        })?;
        // SAFETY: a successful CUDA call writes one NUL-terminated name.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok((device, context as usize, name))
    }

    pub(super) fn release_primary_context(&self, device: i32) {
        let _ = self.call("cuDevicePrimaryCtxRelease", unsafe {
            (self.primary_context_release)(device)
        });
    }

    pub(super) fn push_context(&self, context: usize) -> Result<()> {
        self.call("cuCtxPushCurrent", unsafe {
            (self.context_push)(context as ContextHandle)
        })
    }

    pub(super) fn pop_context(&self) -> Result<()> {
        let mut popped = std::ptr::null_mut();
        self.call("cuCtxPopCurrent", unsafe {
            (self.context_pop)(&mut popped)
        })
    }

    pub(super) fn allocate(&self, bytes: usize) -> Result<DevicePointer> {
        if bytes == 0 {
            return Ok(0);
        }
        let mut pointer = 0;
        self.call("cuMemAlloc", unsafe {
            (self.memory_allocate)(&mut pointer, bytes)
        })?;
        Ok(pointer)
    }

    pub(super) fn free(&self, pointer: DevicePointer) -> Result<()> {
        if pointer == 0 {
            return Ok(());
        }
        self.call("cuMemFree", unsafe { (self.memory_free)(pointer) })
    }

    pub(super) fn zero(&self, pointer: DevicePointer, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.call("cuMemsetD8", unsafe {
            (self.memory_set)(pointer, 0, bytes)
        })
    }

    pub(super) fn upload(&self, pointer: DevicePointer, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        self.call("cuMemcpyHtoD", unsafe {
            (self.copy_host_to_device)(pointer, data.as_ptr().cast(), data.len())
        })
    }

    pub(super) fn download(&self, pointer: DevicePointer, data: &mut [u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        self.call("cuMemcpyDtoH", unsafe {
            (self.copy_device_to_host)(data.as_mut_ptr().cast(), pointer, data.len())
        })
    }

    pub(super) fn copy(
        &self,
        destination: DevicePointer,
        source: DevicePointer,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.call("cuMemcpyDtoD", unsafe {
            (self.copy_device_to_device)(destination, source, bytes)
        })
    }

    pub(super) fn copy_strided(
        &self,
        destination: DevicePointer,
        destination_pitch: usize,
        source: DevicePointer,
        source_pitch: usize,
        element_bytes: usize,
        elements: usize,
    ) -> Result<()> {
        if elements == 0 {
            return Ok(());
        }
        let parameters = Copy2dParameters {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: MEMORY_TYPE_DEVICE,
            src_host: std::ptr::null(),
            src_device: source,
            src_array: std::ptr::null_mut(),
            src_pitch: source_pitch,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: MEMORY_TYPE_DEVICE,
            dst_host: std::ptr::null_mut(),
            dst_device: destination,
            dst_array: std::ptr::null_mut(),
            dst_pitch: destination_pitch,
            width_in_bytes: element_bytes,
            height: elements,
        };
        self.call("cuMemcpy2D", unsafe { (self.copy_2d)(&parameters) })
    }

    pub(super) fn synchronize(&self) -> Result<()> {
        self.call("cuCtxSynchronize", unsafe { (self.synchronize)() })
    }

    fn call(&self, operation: &str, code: ResultCode) -> Result<()> {
        if code == SUCCESS {
            return Ok(());
        }
        let mut message = std::ptr::null();
        let text = if unsafe { (self.error_string)(code, &mut message) } == SUCCESS
            && !message.is_null()
        {
            // SAFETY: the driver owns a static NUL-terminated error string.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA error {code}")
        };
        Err(anyhow::anyhow!("{operation} failed: {text}"))
    }
}

fn library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["nvcuda.dll"]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    }
}
