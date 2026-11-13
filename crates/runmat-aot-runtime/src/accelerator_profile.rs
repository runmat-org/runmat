//! Exact target-built backend features; archive packers read these bytes from
//! the target object rather than inferring support from the build host.

use runmat_execution::host::{NativeAcceleratorProfile, NATIVE_ACCELERATOR_PROFILE_BYTES};

pub(crate) const PROFILE: NativeAcceleratorProfile = NativeAcceleratorProfile {
    wgpu: cfg!(feature = "wgpu"),
    cuda: cfg!(feature = "cuda"),
};

#[used]
#[no_mangle]
pub static RUNMAT_AOT_ACCELERATOR_PROFILE_V1: [u8; NATIVE_ACCELERATOR_PROFILE_BYTES] =
    PROFILE.canonical_bytes();
