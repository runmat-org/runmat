//! This module is compiled FOR the runtime target. The build-host packer reads
//! the exported bytes from its object/archive; it never executes foreign code.
use super::{
    native_abi_layout, NativeTypeLayout, NATIVE_ABI_CONTRACT_PREIMAGE, NATIVE_ABI_SCHEMA_VERSION,
    NATIVE_ABI_VERSION,
};
use runmat_execution::host::{
    NativeAbiProbe, NativeArchitecture, NativeEndianness, NativeEnvironment, NativeOperatingSystem,
    NativeProbeContract, NativeProbeTarget, NativeProbeTypeLayout,
};

const fn portable(layout: NativeTypeLayout) -> NativeProbeTypeLayout {
    assert!(layout.size <= u32::MAX as usize && layout.alignment <= u32::MAX as usize);
    NativeProbeTypeLayout {
        size: layout.size as u32,
        alignment: layout.alignment as u32,
    }
}

/// Unsupported native hosts report unavailable facts rather than guessing a
/// supported target or preventing interpreter-only builds from compiling.
pub const fn try_native_abi_probe() -> Option<NativeAbiProbe> {
    if !cfg!(target_pointer_width = "64") || !cfg!(target_endian = "little") {
        return None;
    }
    let architecture = if cfg!(target_arch = "x86_64") {
        NativeArchitecture::X86_64
    } else if cfg!(target_arch = "aarch64") {
        NativeArchitecture::Aarch64
    } else {
        return None;
    };
    let (operating_system, environment) = if cfg!(target_os = "linux") {
        let environment = if cfg!(target_env = "gnu") {
            NativeEnvironment::Gnu
        } else if cfg!(target_env = "musl") {
            NativeEnvironment::Musl
        } else {
            return None;
        };
        (NativeOperatingSystem::Linux, environment)
    } else if cfg!(target_os = "macos") {
        (NativeOperatingSystem::Macos, NativeEnvironment::Darwin)
    } else if cfg!(all(target_os = "windows", target_env = "msvc")) {
        (NativeOperatingSystem::Windows, NativeEnvironment::Msvc)
    } else {
        return None;
    };
    let endianness = NativeEndianness::Little;
    let layout = native_abi_layout();
    Some(NativeAbiProbe::from_target_layout(
        NativeProbeTarget {
            architecture,
            operating_system,
            environment,
            endianness,
            pointer_width: layout.pointer_width,
        },
        NativeProbeContract {
            schema_version: NATIVE_ABI_SCHEMA_VERSION,
            encoded_version: NATIVE_ABI_VERSION.encoded(),
            preimage: NATIVE_ABI_CONTRACT_PREIMAGE,
        },
        [
            portable(layout.host_status),
            portable(layout.value),
            portable(layout.root),
            portable(layout.root_set),
            portable(layout.source),
            portable(layout.utf8),
            portable(layout.source_map_entry),
            portable(layout.source_map_view),
            portable(layout.resume),
            portable(layout.frame),
            portable(layout.call),
            portable(layout.host),
            portable(layout.exception),
            portable(layout.cancellation),
            portable(layout.suspension),
            portable(layout.deoptimization),
            portable(layout.exit),
            portable(layout.safepoint),
            portable(layout.site_phase),
            portable(layout.site_request),
            portable(layout.site_outcome_kind),
            portable(layout.site_outcome),
        ],
        layout.field_offsets,
    ))
}

/// For target-built producers that explicitly require a supported native ABI.
/// Graceful host capability discovery must use `try_native_abi_probe` instead.
pub const fn native_abi_probe() -> NativeAbiProbe {
    match try_native_abi_probe() {
        Some(probe) => probe,
        None => panic!("target has no supported native ABI probe"),
    }
}

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    target_pointer_width = "64",
    target_endian = "little",
    any(
        all(target_os = "linux", any(target_env = "gnu", target_env = "musl")),
        target_os = "macos",
        all(target_os = "windows", target_env = "msvc")
    )
))]
mod emitted {
    use super::*;
    use runmat_execution::host::NATIVE_ABI_PROBE_BYTES;

    /// No pointers, relocations, padding, or host process state occur in this symbol.
    #[used]
    #[no_mangle]
    pub static RUNMAT_NATIVE_ABI_PROBE_V1: [u8; NATIVE_ABI_PROBE_BYTES] =
        native_abi_probe().canonical_bytes();

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn target_compiler_record_matches_runtime_binding_and_rejects_changed_layout() {
            let probe = NativeAbiProbe::from_canonical_bytes(&RUNMAT_NATIVE_ABI_PROBE_V1).unwrap();
            assert_eq!(probe, native_abi_probe());
            let binding = probe.binding().unwrap();
            let encoded = serde_json::to_vec(&probe).unwrap();
            assert_eq!(
                serde_json::from_slice::<NativeAbiProbe>(&encoded).unwrap(),
                probe
            );
            let mut extended = RUNMAT_NATIVE_ABI_PROBE_V1.to_vec();
            extended.push(0);
            assert!(serde_json::from_slice::<NativeAbiProbe>(
                &serde_json::to_vec(&extended).unwrap()
            )
            .is_err());
            assert_eq!(
                binding.contract_fingerprint,
                crate::native::native_abi_contract_fingerprint()
            );
            assert_eq!(
                binding.layout_fingerprint,
                crate::native::native_abi_layout_fingerprint()
            );
            let mut changed = RUNMAT_NATIVE_ABI_PROBE_V1;
            changed[8] += 1;
            assert!(NativeAbiProbe::from_canonical_bytes(&changed).is_err());
            let mut changed = probe.clone();
            changed.offsets[0] = changed
                .layout(runmat_execution::host::NativeAbiType::Call)
                .size;
            assert!(changed.validate().is_err());
            let mut changed = probe.clone();
            changed.offsets.swap(0, 1);
            assert_ne!(
                changed.binding().unwrap(),
                binding,
                "field offsets participate in identity"
            );
            let mut changed = RUNMAT_NATIVE_ABI_PROBE_V1;
            changed[26 + runmat_execution::host::NATIVE_ABI_CONTRACT_CAPACITY - 1] = 1;
            assert!(
                NativeAbiProbe::from_canonical_bytes(&changed).is_err(),
                "contract padding must be canonical zero bytes"
            );
        }
    }
}
