use super::*;
use windows_sys::Win32::Security::Authorization::SetNamedSecurityInfoW;
use windows_sys::Win32::Security::{
    GetSecurityDescriptorDacl, PROTECTED_DACL_SECURITY_INFORMATION,
};

#[test]
fn creation_is_owner_private_and_world_readable_keys_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty-private-fixture");
    let temporary = create_file_new(&path).unwrap();
    verify_file(&temporary).unwrap();
    // Widen an EMPTY fixture, never a key, to exercise rejecting permissive
    // imported files. The production writer never does this mutation.
    let sddl: Vec<u16> = "D:P(A;;FA;;;OW)(A;;GR;;;WD)\0".encode_utf16().collect();
    let mut descriptor = null_mut();
    // SAFETY: valid terminated SDDL and output pointer; freed below.
    assert_ne!(
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                1,
                &mut descriptor,
                null_mut(),
            )
        },
        0
    );
    let descriptor = LocalAllocation(descriptor);
    let mut present = 0;
    let mut defaulted = 0;
    let mut dacl = null_mut();
    // SAFETY: descriptor is valid and all output pointers are writable.
    assert_ne!(
        unsafe { GetSecurityDescriptorDacl(descriptor.0, &mut present, &mut dacl, &mut defaulted) },
        0
    );
    assert_ne!(present, 0);
    let encoded = wide(path.as_os_str()).unwrap();
    // SAFETY: live fixture path and ACL owned by descriptor for call duration.
    assert_eq!(
        unsafe {
            SetNamedSecurityInfoW(
                encoded.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                dacl,
                std::ptr::null(),
            )
        },
        0
    );
    assert!(verify_file(&temporary).is_err());

    // The node state leaf is protected before any credential is stored. The
    // directory's Administrators compatibility never changes the file policy.
    let state = directory.path().join("state");
    let guard = directory::open(&state, directory::Mode::CreateMissing).unwrap();
    verify(&guard, PrivateObject::Directory).unwrap();
    let secret_path = state.join("empty-credential-fixture");
    let file = create_file_new(&secret_path).unwrap();
    verify_file(&file).unwrap();
    assert!(create_file_new(&secret_path).is_err());
    drop(file);
    let encoded = wide(state.as_os_str()).unwrap();
    // Widen only this EMPTY fixture directory; the actual store never repairs
    // or silently replaces a pre-existing wide directory ACL.
    assert_eq!(
        unsafe {
            SetNamedSecurityInfoW(
                encoded.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                dacl,
                std::ptr::null(),
            )
        },
        0
    );
    assert!(directory::open(&state, directory::Mode::Existing).is_err());
}
