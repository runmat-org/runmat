//! Release the original startup pipe before a detached service accepts readiness.
//! This is a process entrypoint operation, never a library/session log policy.
use crate::ProcessHostResult;
use std::io::Write;

pub fn release_startup_stderr() -> ProcessHostResult<()> {
    let mut stderr = std::io::stderr().lock();
    stderr.flush()?;
    redirect()?;
    Ok(())
}

#[cfg(unix)]
fn redirect() -> std::io::Result<()> {
    use std::os::fd::{AsRawFd, IntoRawFd};
    let null = std::fs::OpenOptions::new().write(true).open("/dev/null")?;
    // SAFETY: both descriptors are live; dup2 atomically replaces only stderr.
    if unsafe { libc::dup2(null.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if null.as_raw_fd() == libc::STDERR_FILENO {
        let _ = null.into_raw_fd();
    }
    Ok(())
}

#[cfg(windows)]
fn redirect() -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle, STD_ERROR_HANDLE};
    struct Descriptor(libc::c_int);
    impl Drop for Descriptor {
        fn drop(&mut self) {
            unsafe {
                libc::close(self.0);
            }
        }
    }
    // Keep the original pipe available until both CRT and Win32 handoff succeed.
    // SAFETY: descriptor2 belongs to this entrypoint, under the Rust stderr lock.
    unsafe {
        let old = GetStdHandle(STD_ERROR_HANDLE);
        let old_crt = libc::get_osfhandle(2);
        let saved = libc::dup(2);
        if saved < 0 {
            return Err(crt_error("_dup(original stderr)"));
        }
        let saved = Descriptor(saved);
        let null = libc::open(c"NUL".as_ptr(), libc::O_WRONLY | libc::O_BINARY);
        if null < 0 {
            return Err(crt_error("_open(NUL)"));
        }
        let null = Descriptor(null);
        let result = (|| {
            if libc::dup2(null.0, 2) < 0 {
                return Err(crt_error("_dup2(stderr)"));
            }
            let handle = libc::get_osfhandle(2) as *mut core::ffi::c_void;
            if SetStdHandle(STD_ERROR_HANDLE, handle) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            if !old.is_null()
                && old != INVALID_HANDLE_VALUE
                && old as isize != old_crt
                && CloseHandle(old) == 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        })();
        if let Err(error) = &result {
            let restored = libc::dup2(saved.0, 2) >= 0
                && SetStdHandle(
                    STD_ERROR_HANDLE,
                    libc::get_osfhandle(2) as *mut core::ffi::c_void,
                ) != 0;
            if !restored {
                // A failed restoration cannot silently discard the original cause.
                // This still-open original descriptor feeds the launcher's bounded reader.
                let text = format!("startup stderr handoff failed: {error}\n");
                let bytes = text.as_bytes();
                let mut offset = 0;
                while offset < bytes.len() {
                    let written = libc::write(
                        saved.0,
                        bytes[offset..].as_ptr().cast(),
                        (bytes.len() - offset) as libc::c_uint,
                    );
                    if written <= 0 {
                        break;
                    }
                    offset += written as usize;
                }
            }
        }
        result
    }
}

#[cfg(windows)]
fn crt_error(operation: &str) -> std::io::Error {
    extern "C" {
        fn _errno() -> *mut libc::c_int;
    }
    // SAFETY: the CRT returns this thread's non-null errno storage. Do not
    // interpret a CRT errno as an unrelated Win32 GetLastError value.
    let errno = unsafe { *_errno() };
    std::io::Error::other(format!("{operation} failed with CRT errno {errno}"))
}
