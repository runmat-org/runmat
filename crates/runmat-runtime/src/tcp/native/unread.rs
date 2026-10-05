//! A non-consuming native socket snapshot. This is not a promise that the
//! count remains stable after the query: peers may concurrently send/close.

use std::io;
use tokio::net::TcpStream;

#[cfg(unix)]
pub(super) fn socket_unread_bytes(stream: &TcpStream) -> io::Result<usize> {
    use std::os::fd::AsRawFd;

    let mut count: libc::c_int = 0;
    // SAFETY: the borrowed stream owns a valid descriptor and `count` is a
    // writable c_int for the duration of this synchronous ioctl.
    let result = unsafe { libc::ioctl(stream.as_raw_fd(), libc::FIONREAD, &mut count) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(count).map_err(|_| io::Error::other("negative native TCP unread count"))
}

#[cfg(windows)]
pub(super) fn socket_unread_bytes(stream: &TcpStream) -> io::Result<usize> {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{ioctlsocket, FIONREAD, SOCKET};

    let mut count: u32 = 0;
    // SAFETY: the borrowed stream owns a valid socket and `count` is a
    // writable u32 for the duration of this synchronous WinSock query.
    let result = unsafe { ioctlsocket(stream.as_raw_socket() as SOCKET, FIONREAD, &mut count) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(count as usize)
}
