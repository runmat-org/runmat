#[cfg(test)]
mod tests {
    use core::mem::{align_of, offset_of, size_of};

    use crate::*;

    #[test]
    fn stable_handle_and_version_layouts_are_exact() {
        assert_eq!(size_of::<RunMatAbiVersion>(), 4);
        assert_eq!(align_of::<RunMatAbiVersion>(), 2);
        assert_eq!(size_of::<RunMatForeignHandle>(), 24);
        assert_eq!(offset_of!(RunMatForeignHandle, host), 0);
        assert_eq!(offset_of!(RunMatForeignHandle, resource), 8);
        assert_eq!(offset_of!(RunMatForeignHandle, generation), 16);
        assert_eq!(size_of::<RunMatValueHandle>(), 16);
        assert_eq!(size_of::<RunMatBufferLeaseHandle>(), 16);
    }

    #[test]
    fn every_vtable_starts_with_version_and_size() {
        assert_eq!(offset_of!(RunMatHostVTable, abi_version), 0);
        assert_eq!(offset_of!(RunMatExtensionVTable, abi_version), 0);
        assert!(offset_of!(RunMatHostVTable, struct_size) >= size_of::<RunMatAbiVersion>());
        assert!(offset_of!(RunMatExtensionVTable, struct_size) >= size_of::<RunMatAbiVersion>());
    }

    #[test]
    fn abi_1_1_host_callbacks_extend_the_complete_1_0_prefix() {
        assert!(
            offset_of!(RunMatHostVTable, borrow_buffer_lease)
                > offset_of!(RunMatHostVTable, is_cancelled)
        );
        assert!(
            offset_of!(RunMatHostVTable, release_buffer)
                > offset_of!(RunMatHostVTable, borrow_buffer_lease)
        );
    }

    #[test]
    fn public_header_names_the_canonical_version_and_query_symbol() {
        assert!(RUNMAT_EXTENSION_HEADER.contains("RUNMAT_EXTENSION_ABI_MAJOR 1"));
        assert!(RUNMAT_EXTENSION_HEADER.contains("RUNMAT_EXTENSION_ABI_MINOR 1"));
        assert!(RUNMAT_EXTENSION_HEADER.contains("borrow_buffer_lease"));
        assert!(RUNMAT_EXTENSION_HEADER.contains("release_buffer"));
        assert!(RUNMAT_EXTENSION_HEADER.contains("RUNMAT_EXTENSION_CAP_BUFFER_LEASES"));
        assert!(RUNMAT_EXTENSION_HEADER.contains(RUNMAT_EXTENSION_QUERY_SYMBOL));
    }
}
