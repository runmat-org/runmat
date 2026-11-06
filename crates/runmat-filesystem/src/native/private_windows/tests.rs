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
    let encoded = wide(&path).unwrap();
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
    let nested = state.join("jobs/attempt");
    crate::NativeFsProvider
        .create_private_directory_durable(&nested)
        .unwrap();
    crate::NativeFsProvider
        .create_private_directory_durable(&nested)
        .unwrap();
    for path in [state.join("jobs"), nested] {
        let handle = directory::open(&path, directory::Mode::Existing).unwrap();
        verify(&handle, PrivateObject::Directory).unwrap();
    }
    let secret_path = state.join("empty-credential-fixture");
    let file = create_file_new(&secret_path).unwrap();
    verify_file(&file).unwrap();
    assert!(create_file_new(&secret_path).is_err());
    drop(file);
    let encoded = wide(&state).unwrap();
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
    assert!(crate::NativeFsProvider
        .create_private_directory_durable(&state)
        .is_err());
}

#[test]
fn private_publication_conserves_the_validated_winner_and_its_acl() {
    let root = tempfile::tempdir().unwrap();
    for depth in [0, 6] {
        let mut parent = root.path().to_owned();
        for index in 0..depth {
            parent.push(format!("segment-{index}-{}", "x".repeat(40)));
        }
        let path = parent.join("state/jobs/credential");
        if depth != 0 {
            assert!(path.as_os_str().encode_wide().count() > 260);
        }
        let provider = crate::NativeFsProvider;
        assert!(matches!(
            provider
                .atomic_create_private_new(&path, b"original", 8)
                .unwrap(),
            crate::AtomicCreateResult::Published {
                durability: crate::PublicationDurability::Durable
            }
        ));
        verify_file(&open_file(&path).unwrap()).unwrap();
        assert!(matches!(
            provider.atomic_create_private_new(&path, b"replacement", 8).unwrap(),
            crate::AtomicCreateResult::Existing(bytes) if bytes == b"original"
        ));
        assert!(provider
            .atomic_create_private_new(&path, b"replacement", 7)
            .is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
    }
}

#[test]
fn private_api_paths_preserve_disk_and_share_identity() {
    for (input, expected) in [
        (r"C:\state\head.json", r"\\?\C:\state\head.json"),
        ("C:/state/head.json", r"\\?\C:\state\head.json"),
        (
            r"\\server\share\state\head.json",
            r"\\?\UNC\server\share\state\head.json",
        ),
        (r"\\?\C:\state\head.json", r"\\?\C:\state\head.json"),
        (
            r"\\?\UNC\server\share\state\head.json",
            r"\\?\UNC\server\share\state\head.json",
        ),
    ] {
        let encoded = wide(Path::new(input)).unwrap();
        assert_eq!(encoded.last(), Some(&0));
        assert_eq!(
            &encoded[..encoded.len() - 1],
            expected.encode_utf16().collect::<Vec<_>>()
        );
    }
    for invalid in [
        r"state\head.json",
        r"C:\state\..\head.json",
        "C:\\state\\bad\0name",
        r"\\.\PhysicalDrive0",
    ] {
        assert_eq!(
            wide(Path::new(invalid)).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
}
