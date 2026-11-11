//! Native private-file mechanism extracted from CLI recovery. No secret bytes
//! are written/read before creation-time protection/opened-handle validation.
use std::ffi::c_void;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::os::windows::{
    ffi::OsStrExt as _,
    fs::MetadataExt as _,
    io::{AsRawHandle as _, FromRawHandle as _},
};
use std::path::{Component, Path, Prefix};
use std::ptr::null_mut;

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
    CreateFileW, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_REPARSE_POINT,
    FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    OPEN_EXISTING,
};

pub(super) mod directory;
#[cfg(test)]
mod tests;

const ALLOW_ACE: u8 = 0;
const DENY_ACE: u8 = 1;

#[derive(Clone, Copy)]
enum PrivateObject {
    File,
    Directory,
}

struct LocalAllocation(*mut c_void);
impl Drop for LocalAllocation {
    fn drop(&mut self) {
        // SAFETY: this allocation is returned by APIs requiring LocalFree.
        unsafe {
            LocalFree(self.0);
        }
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::PermissionDenied, message)
}

fn wide(path: &Path) -> io::Result<Vec<u16>> {
    // The private Win32 calls need extended-length paths independently of the
    // caller's executable manifest and machine-wide long-path policy. Build
    // the same absolute namespace lexically; never follow a parent link here.
    if !path.is_absolute() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "private Windows path must be absolute",
        ));
    }
    let mut components = path.components();
    let Some(Component::Prefix(prefix)) = components.next() else {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "private Windows path needs a volume or share",
        ));
    };
    let mut value = Vec::new();
    match prefix.kind() {
        Prefix::Disk(drive) => {
            value.extend(r"\\?\".encode_utf16());
            value.push(u16::from(drive));
            value.push(u16::from(b':'));
        }
        Prefix::UNC(server, share) => {
            value.extend(r"\\?\UNC\".encode_utf16());
            value.extend(server.encode_wide());
            value.push(u16::from(b'\\'));
            value.extend(share.encode_wide());
        }
        Prefix::VerbatimDisk(_) | Prefix::VerbatimUNC(_, _) => {
            value.extend(prefix.as_os_str().encode_wide());
        }
        Prefix::DeviceNS(_) | Prefix::Verbatim(_) => {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "private Windows state requires a disk or UNC share",
            ));
        }
    }
    let mut separator = false;
    for component in components {
        match component {
            Component::RootDir => {
                value.push(u16::from(b'\\'));
                separator = false;
            }
            Component::Normal(name) => {
                if separator {
                    value.push(u16::from(b'\\'));
                }
                value.extend(name.encode_wide());
                separator = true;
            }
            Component::CurDir | Component::ParentDir | Component::Prefix(_) => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    "private Windows path must not contain relative components",
                ));
            }
        }
    }
    if value.contains(&0) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "private Windows path contains a NUL",
        ));
    }
    value.push(0);
    Ok(value)
}

fn descriptor(kind: PrivateObject) -> io::Result<LocalAllocation> {
    let sddl = match kind {
        PrivateObject::File => "D:P(A;;FA;;;OW)(A;;FA;;;SY)\0",
        // Existing node service installation permits Administrators on the
        // directory. Never inherit that grant into a protected secret file.
        PrivateObject::Directory => "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)\0",
    };
    let sddl: Vec<u16> = sddl.encode_utf16().collect();
    let mut pointer = null_mut();
    // SAFETY: terminated input and valid output; ownership follows LocalFree.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            1,
            &mut pointer,
            null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(LocalAllocation(pointer))
}

fn attributes(descriptor: &LocalAllocation) -> SECURITY_ATTRIBUTES {
    SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor.0,
        bInheritHandle: 0,
    }
}

pub(super) fn create_file_new(path: &Path) -> io::Result<File> {
    let path = std::path::absolute(path)?;
    directory::check_chain(
        path.parent()
            .ok_or_else(|| invalid("private file needs a parent"))?,
        directory::Mode::Existing,
    )?;
    let descriptor = descriptor(PrivateObject::File)?;
    let attributes = attributes(&descriptor);
    let path = wide(&path)?;
    // SAFETY: all input storage remains live; CREATE_NEW never replaces a
    // collision/reparse. Protection exists before the file becomes visible.
    let handle = unsafe {
        CreateFileW(
            path.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            &attributes,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: one valid owned handle from CreateFileW.
    let file = unsafe { File::from_raw_handle(handle) };
    verify_file(&file)?;
    Ok(file)
}

pub(super) fn open_file(path: &Path) -> io::Result<File> {
    let path = std::path::absolute(path)?;
    directory::check_chain(
        path.parent()
            .ok_or_else(|| invalid("private file needs a parent"))?,
        directory::Mode::Existing,
    )?;
    let path = wide(&path)?;
    // SAFETY: terminated path; reparse handles are opened, not followed, and
    // rejected below before a caller can read any secret bytes.
    let handle = unsafe {
        CreateFileW(
            path.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            null_mut(),
            OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: one valid owned handle from CreateFileW.
    let file = unsafe { File::from_raw_handle(handle) };
    verify_file(&file)?;
    Ok(file)
}

pub(super) fn verify_file(file: &File) -> io::Result<()> {
    verify(file, PrivateObject::File)
}

fn sid(value: &str) -> io::Result<LocalAllocation> {
    let value: Vec<u16> = value.encode_utf16().chain(Some(0)).collect();
    let mut pointer = null_mut();
    // SAFETY: terminated input and valid writable output pointer.
    if unsafe { ConvertStringSidToSidW(value.as_ptr(), &mut pointer) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(LocalAllocation(pointer))
}

fn verify(file: &File, kind: PrivateObject) -> io::Result<()> {
    let metadata = file.metadata()?;
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(invalid("private Windows state must not be a reparse point"));
    }
    let correct_type = match kind {
        PrivateObject::File => metadata.is_file(),
        PrivateObject::Directory => metadata.is_dir(),
    };
    if !correct_type {
        return Err(invalid("private Windows state has the wrong file type"));
    }
    let mut owner = null_mut();
    let mut dacl: *mut ACL = null_mut();
    let mut descriptor = null_mut();
    // SAFETY: live handle and valid outputs; returned pointers belong to the
    // descriptor until LocalFree, exactly as in the original recovery adapter.
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
        return Err(io::Error::from_raw_os_error(status as i32));
    }
    let _descriptor = LocalAllocation(descriptor);
    if owner.is_null() || dacl.is_null() {
        return Err(invalid(
            "private Windows state needs an owner and restrictive DACL",
        ));
    }
    let system = sid("S-1-5-18")?;
    let owner_rights = sid("S-1-3-4")?;
    let administrators = match kind {
        PrivateObject::File => None,
        PrivateObject::Directory => Some(sid("S-1-5-32-544")?),
    };
    // SAFETY: Windows supplied a valid ACL and bounds its ACE count.
    for index in 0..u32::from(unsafe { (*dacl).AceCount }) {
        let mut ace = null_mut();
        // SAFETY: index is bounded by the ACL's own count.
        if unsafe { GetAce(dacl, index, &mut ace) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: successful GetAce points at a live ACE_HEADER.
        let header = unsafe { &*(ace as *const ACE_HEADER) };
        match header.AceType {
            DENY_ACE => {}
            ALLOW_ACE => {
                if usize::from(header.AceSize) < std::mem::size_of::<ACCESS_ALLOWED_ACE>() {
                    return Err(invalid("private Windows state has a malformed ACE"));
                }
                // SAFETY: type/size checked; Windows owns the inline SID.
                let principal =
                    unsafe { std::ptr::addr_of!((*(ace as *const ACCESS_ALLOWED_ACE)).SidStart) }
                        as *mut c_void;
                // SAFETY: all compared SIDs remain live for this validation.
                let allowed = unsafe {
                    EqualSid(principal, owner) != 0
                        || EqualSid(principal, owner_rights.0) != 0
                        || EqualSid(principal, system.0) != 0
                        || administrators
                            .as_ref()
                            .is_some_and(|sid| EqualSid(principal, sid.0) != 0)
                };
                if !allowed {
                    return Err(invalid("private Windows state grants access beyond its permitted owner/SYSTEM policy"));
                }
            }
            _ => return Err(invalid("private Windows state has an unsupported ACE")),
        }
    }
    Ok(())
}
