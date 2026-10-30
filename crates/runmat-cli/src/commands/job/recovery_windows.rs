//! Windows private recovery files: restrict access in CreateFileW itself, not
//! after creating a file with an inherited (possibly public) directory ACL.
use std::ffi::c_void;
use std::fs::File;
use std::os::windows::{
    ffi::OsStrExt as _,
    io::{AsRawHandle as _, FromRawHandle as _},
};
use std::path::Path;
use std::ptr::null_mut;

use anyhow::{bail, Context as _, Result};
use windows_sys::Win32::Foundation::{
    LocalFree, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, ConvertStringSidToSidW, GetSecurityInfo,
    SE_FILE_OBJECT,
};
use windows_sys::Win32::Security::{
    EqualSid, GetAce, ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, DACL_SECURITY_INFORMATION,
    OWNER_SECURITY_INFORMATION, SECURITY_ATTRIBUTES,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
};

// These are the documented Win32 ACE_HEADER.AceType values. windows-sys exposes
// ACCESS_ALLOWED_ACE but not these macros in its Security module.
const ALLOW_ACE: u8 = 0;
const DENY_ACE: u8 = 1;

struct LocalAllocation(*mut c_void);
impl Drop for LocalAllocation {
    fn drop(&mut self) {
        // SAFETY: each allocation comes from an API documented to use LocalFree.
        unsafe {
            LocalFree(self.0);
        }
    }
}

fn wide(value: &std::ffi::OsStr) -> Result<Vec<u16>> {
    let mut value: Vec<u16> = value.encode_wide().collect();
    if value.contains(&0) {
        bail!("Windows recovery path contains a NUL");
    }
    value.push(0);
    Ok(value)
}

pub(super) fn create_temporary(parent: &Path) -> Result<tempfile::NamedTempFile> {
    let parent = parent
        .canonicalize()
        .context("resolve private recovery output directory")?;
    let sddl: Vec<u16> = "D:P(A;;FA;;;OW)(A;;FA;;;SY)\0".encode_utf16().collect();
    let mut descriptor = null_mut();
    // SAFETY: input is terminated UTF-16 and output is a valid writable pointer.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            1,
            &mut descriptor,
            null_mut(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error())
            .context("create private recovery security descriptor");
    }
    let descriptor = LocalAllocation(descriptor);
    let attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor.0,
        bInheritHandle: 0,
    };
    // CREATE_NEW rejects collisions/symlinks; the fresh name isn't relied on
    // for access control. The protected DACL is present before the file exists.
    let path = parent.join(format!(".runmat-recovery-{}.tmp", uuid::Uuid::new_v4()));
    let encoded = wide(path.as_os_str())?;
    // SAFETY: attributes/descriptor and terminated path remain valid for call.
    let handle = unsafe {
        CreateFileW(
            encoded.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
            &attributes,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(std::io::Error::last_os_error()).context("create owner-private recovery file");
    }
    // SAFETY: CreateFileW returned one uniquely owned valid handle.
    let file = unsafe { File::from_raw_handle(handle) };
    let temporary =
        tempfile::NamedTempFile::from_parts(file, tempfile::TempPath::try_from_path(path)?);
    verify_file(temporary.as_file())?;
    Ok(temporary)
}

fn sid(value: &str) -> Result<LocalAllocation> {
    let value: Vec<u16> = value.encode_utf16().chain(Some(0)).collect();
    let mut pointer = null_mut();
    // SAFETY: terminated string and valid output pointer.
    if unsafe { ConvertStringSidToSidW(value.as_ptr(), &mut pointer) } == 0 {
        return Err(std::io::Error::last_os_error()).context("parse recovery ACL principal");
    }
    Ok(LocalAllocation(pointer))
}

pub(super) fn verify_file(file: &File) -> Result<()> {
    let mut owner = null_mut();
    let mut dacl: *mut ACL = null_mut();
    let mut descriptor = null_mut();
    // SAFETY: handle is live, every requested output has valid storage. Returned
    // owner/ACL pointers remain owned by descriptor until LocalFree below.
    let status = unsafe {
        GetSecurityInfo(
            file.as_raw_handle(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | OWNER_SECURITY_INFORMATION,
            &mut owner,
            null_mut(),
            &mut dacl,
            null_mut(),
            &mut descriptor,
        )
    };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status as i32))
            .context("inspect recovery file ACL");
    }
    let _descriptor = LocalAllocation(descriptor);
    if owner.is_null() || dacl.is_null() {
        bail!("recovery file must have an owner and a restrictive DACL");
    }
    let system = sid("S-1-5-18")?;
    let owner_rights = sid("S-1-3-4")?;
    // SAFETY: GetSecurityInfo provides a valid ACL structure.
    for index in 0..u32::from(unsafe { (*dacl).AceCount }) {
        let mut ace = null_mut();
        // SAFETY: index is bounded by AceCount and output pointer is valid.
        if unsafe { GetAce(dacl, index, &mut ace) } == 0 {
            return Err(std::io::Error::last_os_error()).context("inspect recovery file ACE");
        }
        // SAFETY: successful GetAce returns a valid ACE beginning with ACE_HEADER.
        let header = unsafe { &*(ace as *const ACE_HEADER) };
        match header.AceType {
            DENY_ACE => {}
            ALLOW_ACE => {
                if usize::from(header.AceSize) < std::mem::size_of::<ACCESS_ALLOWED_ACE>() {
                    bail!("recovery file has a malformed access-control entry");
                }
                // SAFETY: type and minimum size checked; SidStart is inline SID.
                let principal =
                    unsafe { std::ptr::addr_of!((*(ace as *const ACCESS_ALLOWED_ACE)).SidStart) }
                        as *mut c_void;
                // SAFETY: owner and converted principals are valid SIDs, and
                // Windows supplied this ACE's validated SID in the returned ACL.
                let allowed = unsafe {
                    EqualSid(principal, owner) != 0
                        || EqualSid(principal, owner_rights.0) != 0
                        || EqualSid(principal, system.0) != 0
                };
                if !allowed {
                    bail!("recovery file grants access beyond its owner and SYSTEM");
                }
            }
            _ => bail!("recovery file has an unsupported access-control entry"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Security::Authorization::SetNamedSecurityInfoW;
    use windows_sys::Win32::Security::{
        GetSecurityDescriptorDacl, PROTECTED_DACL_SECURITY_INFORMATION,
    };

    #[test]
    fn creation_is_owner_private_and_world_readable_keys_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let temporary = create_temporary(directory.path()).unwrap();
        verify_file(temporary.as_file()).unwrap();
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
            unsafe {
                GetSecurityDescriptorDacl(descriptor.0, &mut present, &mut dacl, &mut defaulted)
            },
            0
        );
        assert_ne!(present, 0);
        let path = wide(temporary.path().as_os_str()).unwrap();
        // SAFETY: live fixture path and ACL owned by descriptor for call duration.
        assert_eq!(
            unsafe {
                SetNamedSecurityInfoW(
                    path.as_ptr(),
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
        assert!(verify_file(temporary.as_file()).is_err());
    }
}
