//! The memory this host admits for one meshing request.
//!
//! The browser host grows its wasm memory on demand up to the module's maximum
//! (`-sMAXIMUM_MEMORY` in scripts/runtime/build-exact-cad-wasm.sh). A meshing request may use
//! what the module can still grow into: that maximum minus the memory already in use when the
//! request starts. The host measures both inside the module at request time, so the ceiling
//! follows the build's maximum and whatever an earlier CAD or meshing call left resident. When
//! the run brings an admitted memory budget smaller than that headroom, the budget wins.
//!
//! A native parity build has no wasm memory to measure. It meshes under the portable fixture
//! ceiling ([`MeshingExecutionCeilings::PORTABLE_DEFAULT`]), which the meshing contract defines
//! for a host whose capacity is unknown.
use runmat_meshing_core::MeshingExecutionCeilings;

/// This module's memory, measured in the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    /// The most memory the module can grow to.
    pub maximum_bytes: u64,
    /// Memory in use now: static data, the stack and live heap allocations. Freed heap blocks
    /// below the break are reusable, so they do not count.
    pub in_use_bytes: u64,
}

impl HostMemory {
    /// The memory a request can still allocate.
    pub fn headroom_bytes(&self) -> u64 {
        self.maximum_bytes.saturating_sub(self.in_use_bytes)
    }

    /// Measures this module's memory; `None` in a native build.
    pub fn measure() -> Option<Self> {
        #[cfg(target_os = "emscripten")]
        {
            #[repr(C)]
            struct MallInfo {
                arena: usize,
                ordblks: usize,
                smblks: usize,
                hblks: usize,
                hblkhd: usize,
                usmblks: usize,
                fsmblks: usize,
                uordblks: usize,
                fordblks: usize,
                keepcost: usize,
            }
            unsafe extern "C" {
                fn emscripten_get_heap_max() -> usize;
                fn sbrk(increment: isize) -> *mut std::ffi::c_void;
                fn mallinfo() -> MallInfo;
            }
            // SAFETY: these emscripten libc calls read allocator state; `sbrk(0)` does not move
            // the break.
            let (maximum, brk, free) = unsafe {
                (
                    emscripten_get_heap_max(),
                    sbrk(0) as usize,
                    mallinfo().fordblks,
                )
            };
            Some(Self {
                maximum_bytes: maximum as u64,
                in_use_bytes: brk.saturating_sub(free) as u64,
            })
        }
        #[cfg(not(target_os = "emscripten"))]
        {
            None
        }
    }
}

/// The memory ceiling for one meshing request: the measured headroom, lowered to the run's
/// admitted memory budget when the run brings a smaller one.
pub fn meshing_memory_bytes(measured: Option<HostMemory>, admitted_bytes: Option<u64>) -> u64 {
    let host = measured.map_or(
        MeshingExecutionCeilings::PORTABLE_DEFAULT.maximum_memory_bytes,
        |memory| memory.headroom_bytes(),
    );
    admitted_bytes.map_or(host, |admitted| admitted.min(host))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn ceiling_is_the_measured_headroom() {
        // Two hosts with different maxima and baselines get different ceilings: the ceiling
        // follows the measurement, with no fixed value behind it.
        for (maximum, in_use) in [
            (4 * GIB - (64 << 10), 40 << 20),
            (4 * GIB - (64 << 10), 900 << 20),
            (2 * GIB, 100 << 20),
            (512 << 20, 300 << 20),
        ] {
            let memory = HostMemory {
                maximum_bytes: maximum,
                in_use_bytes: in_use,
            };
            assert_eq!(meshing_memory_bytes(Some(memory), None), maximum - in_use);
        }
    }

    #[test]
    fn a_smaller_admitted_budget_wins() {
        let memory = HostMemory {
            maximum_bytes: 4 * GIB,
            in_use_bytes: GIB,
        };
        assert_eq!(meshing_memory_bytes(Some(memory), Some(GIB)), GIB);
        assert_eq!(meshing_memory_bytes(Some(memory), Some(8 * GIB)), 3 * GIB);
    }

    #[test]
    fn a_full_module_admits_nothing() {
        let memory = HostMemory {
            maximum_bytes: GIB,
            in_use_bytes: 2 * GIB,
        };
        assert_eq!(memory.headroom_bytes(), 0);
    }

    #[test]
    fn a_native_build_has_no_wasm_memory_to_measure() {
        #[cfg(not(target_os = "emscripten"))]
        assert_eq!(HostMemory::measure(), None);
        assert_eq!(
            meshing_memory_bytes(None, None),
            MeshingExecutionCeilings::PORTABLE_DEFAULT.maximum_memory_bytes
        );
    }
}
