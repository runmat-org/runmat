use std::mem::{align_of, size_of};

use super::{
    NativeCall, NativeCancellation, NativeDeoptimization, NativeException, NativeExit, NativeFrame,
    NativeHostStatus, NativeHostVTable, NativeResumeState, NativeRoot, NativeRootSet,
    NativeSafepoint, NativeSiteOutcome, NativeSiteOutcomeKind, NativeSitePhase, NativeSiteRequest,
    NativeSourceLocation, NativeSourceMapEntry, NativeSourceMapView, NativeSuspension, NativeUtf8,
    NativeValueRef,
};

pub const NATIVE_ABI_SCHEMA_VERSION: u16 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeAbiVersion {
    pub major: u16,
    pub minor: u16,
}

impl NativeAbiVersion {
    pub const fn encoded(self) -> u32 {
        ((self.major as u32) << 16) | self.minor as u32
    }
}

pub const NATIVE_ABI_VERSION: NativeAbiVersion = NativeAbiVersion { major: 1, minor: 3 };

/// Target-independent identity used by executable/runtime compatibility keys.
pub fn native_abi_contract_fingerprint() -> runmat_execution::Digest {
    runmat_execution::Digest::sha256(
        b"runmat-native-abi-v4\0opaque-values\0generation-roots\0explicit-host-table\0typed-host-status\0out-parameter-exits\0no-cross-boundary-unwind\0exact-site-resume\0transactional-exit\0path-free-sources\0typed-native-site-dispatch\0no-obsolete-slow-call-slot",
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeTypeLayout {
    pub size: usize,
    pub alignment: usize,
}

impl NativeTypeLayout {
    fn of<T>() -> Self {
        Self {
            size: size_of::<T>(),
            alignment: align_of::<T>(),
        }
    }
}

/// Complete target layout used by native-code cache keys and compatibility
/// checks. Portable executable identity uses the contract fingerprint above;
/// native objects additionally bind this target-specific fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeAbiLayout {
    pub pointer_width: u16,
    pub host_status: NativeTypeLayout,
    pub value: NativeTypeLayout,
    pub root: NativeTypeLayout,
    pub root_set: NativeTypeLayout,
    pub source: NativeTypeLayout,
    pub utf8: NativeTypeLayout,
    pub source_map_entry: NativeTypeLayout,
    pub source_map_view: NativeTypeLayout,
    pub resume: NativeTypeLayout,
    pub frame: NativeTypeLayout,
    pub call: NativeTypeLayout,
    pub host: NativeTypeLayout,
    pub exception: NativeTypeLayout,
    pub cancellation: NativeTypeLayout,
    pub suspension: NativeTypeLayout,
    pub deoptimization: NativeTypeLayout,
    pub exit: NativeTypeLayout,
    pub safepoint: NativeTypeLayout,
    pub site_phase: NativeTypeLayout,
    pub site_request: NativeTypeLayout,
    pub site_outcome_kind: NativeTypeLayout,
    pub site_outcome: NativeTypeLayout,
}

pub fn native_abi_layout() -> NativeAbiLayout {
    NativeAbiLayout {
        pointer_width: usize::BITS as u16,
        host_status: NativeTypeLayout::of::<NativeHostStatus>(),
        value: NativeTypeLayout::of::<NativeValueRef>(),
        root: NativeTypeLayout::of::<NativeRoot>(),
        root_set: NativeTypeLayout::of::<NativeRootSet>(),
        source: NativeTypeLayout::of::<NativeSourceLocation>(),
        utf8: NativeTypeLayout::of::<NativeUtf8>(),
        source_map_entry: NativeTypeLayout::of::<NativeSourceMapEntry>(),
        source_map_view: NativeTypeLayout::of::<NativeSourceMapView>(),
        resume: NativeTypeLayout::of::<NativeResumeState>(),
        frame: NativeTypeLayout::of::<NativeFrame>(),
        call: NativeTypeLayout::of::<NativeCall>(),
        host: NativeTypeLayout::of::<NativeHostVTable>(),
        exception: NativeTypeLayout::of::<NativeException>(),
        cancellation: NativeTypeLayout::of::<NativeCancellation>(),
        suspension: NativeTypeLayout::of::<NativeSuspension>(),
        deoptimization: NativeTypeLayout::of::<NativeDeoptimization>(),
        exit: NativeTypeLayout::of::<NativeExit>(),
        safepoint: NativeTypeLayout::of::<NativeSafepoint>(),
        site_phase: NativeTypeLayout::of::<NativeSitePhase>(),
        site_request: NativeTypeLayout::of::<NativeSiteRequest>(),
        site_outcome_kind: NativeTypeLayout::of::<NativeSiteOutcomeKind>(),
        site_outcome: NativeTypeLayout::of::<NativeSiteOutcome>(),
    }
}

pub fn native_abi_layout_fingerprint() -> runmat_execution::Digest {
    let layout = native_abi_layout();
    runmat_execution::Digest::sha256(format!(
        "runmat-native-layout-v1\0{}\0{:?}",
        std::env::consts::ARCH,
        layout
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn version_and_contract_identity_are_stable() {
        assert_eq!(NATIVE_ABI_VERSION.encoded(), 0x0001_0003);
        assert_eq!(
            native_abi_contract_fingerprint().to_string(),
            "sha256:f550a4e40b7d2f75c72231767205f88b68c73b1d7ae89b057bbbdcf4de67b225"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn layouts_are_c_compatible_and_offsets_are_explicit() {
        let layout = native_abi_layout();
        assert_eq!(layout.pointer_width, usize::BITS as u16);
        assert_eq!(
            layout.host_status,
            NativeTypeLayout {
                size: 4,
                alignment: 4
            }
        );
        assert_eq!(
            layout.value,
            NativeTypeLayout {
                size: 16,
                alignment: 8
            }
        );
        assert_eq!(
            layout.root,
            NativeTypeLayout {
                size: 24,
                alignment: 8
            }
        );
        assert_eq!(
            layout.source,
            NativeTypeLayout {
                size: 24,
                alignment: 8
            }
        );
        assert_eq!(std::mem::offset_of!(NativeFrame, caller), 16);
        assert_eq!(std::mem::offset_of!(NativeCall, host), 16);
        assert_eq!(std::mem::offset_of!(NativeExit, exception), 16);
        assert_eq!(std::mem::offset_of!(NativeResumeState, phase), 12);
        assert_eq!(std::mem::offset_of!(NativeResumeState, ordinal), 16);
        assert_eq!(std::mem::offset_of!(NativeResumeState, bytecode_pc), 32);
        assert_eq!(
            layout.site_request,
            NativeTypeLayout {
                size: 24,
                alignment: 4
            }
        );
        assert_eq!(
            layout.site_outcome,
            NativeTypeLayout {
                size: 16,
                alignment: 4
            }
        );
        assert!(
            std::mem::offset_of!(NativeHostVTable, execute_site)
                > std::mem::offset_of!(NativeHostVTable, source_lookup)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn target_layout_fingerprint_is_deterministic() {
        assert_eq!(
            native_abi_layout_fingerprint(),
            native_abi_layout_fingerprint()
        );
        assert_ne!(
            native_abi_layout_fingerprint(),
            native_abi_contract_fingerprint()
        );
    }
}
