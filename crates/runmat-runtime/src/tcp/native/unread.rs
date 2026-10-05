//! A non-consuming native socket snapshot. This is not a promise that the
//! count remains stable after the query: peers may concurrently send/close.

use std::io;
use tokio::net::TcpStream;

pub(super) struct SocketSnapshot {
    pub(super) unread_bytes: usize,
    pub(super) peer_closed: bool,
}

#[cfg(unix)]
pub(super) fn socket_snapshot(stream: &TcpStream) -> io::Result<SocketSnapshot> {
    use std::os::fd::AsRawFd;

    let mut count: libc::c_int = 0;
    // SAFETY: the borrowed stream owns a valid descriptor and `count` is a
    // writable c_int for the duration of this synchronous ioctl.
    let result = unsafe { libc::ioctl(stream.as_raw_fd(), libc::FIONREAD, &mut count) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    let unread_bytes =
        usize::try_from(count).map_err(|_| io::Error::other("negative native TCP unread count"))?;
    if unread_bytes != 0 {
        return Ok(SocketSnapshot {
            unread_bytes,
            peer_closed: false,
        });
    }
    let mut byte = 0u8;
    // SAFETY: the descriptor remains owned by `stream`; MSG_PEEK never
    // consumes the byte. Tokio sockets are nonblocking, so an idle live peer
    // returns WouldBlock rather than stalling status observation.
    let n = unsafe {
        libc::recv(
            stream.as_raw_fd(),
            (&mut byte as *mut u8).cast(),
            1,
            libc::MSG_PEEK,
        )
    };
    let peer_closed = match n {
        0 => true,
        1 => false,
        -1 => {
            let source = io::Error::last_os_error();
            if source.kind() == io::ErrorKind::WouldBlock {
                false
            } else {
                return Err(source);
            }
        }
        _ => return Err(io::Error::other("invalid TCP peek result")),
    };
    Ok(SocketSnapshot {
        unread_bytes,
        peer_closed,
    })
}

#[cfg(windows)]
pub(super) fn socket_snapshot(stream: &TcpStream) -> io::Result<SocketSnapshot> {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{
        ioctlsocket, recv, WSAGetLastError, FIONREAD, MSG_PEEK, SOCKET, WSAEWOULDBLOCK,
    };

    let mut count: u32 = 0;
    let socket = stream.as_raw_socket() as SOCKET;
    // SAFETY: the borrowed stream owns a valid socket and `count` is a
    // writable u32 for the duration of this synchronous WinSock query.
    let result = unsafe { ioctlsocket(socket, FIONREAD, &mut count) };
    if result != 0 {
        // WinSock uses WSAGetLastError, not the thread's GetLastError slot.
        return Err(io::Error::from_raw_os_error(unsafe { WSAGetLastError() }));
    }
    let unread_bytes = count as usize;
    if unread_bytes != 0 {
        return Ok(SocketSnapshot {
            unread_bytes,
            peer_closed: false,
        });
    }
    let mut byte = 0u8;
    // SAFETY: borrowed Tokio sockets are nonblocking and MSG_PEEK does not
    // consume data. `byte` remains writable for the synchronous WinSock call.
    let n = unsafe { recv(socket, (&mut byte as *mut u8).cast(), 1, MSG_PEEK) };
    let peer_closed = match n {
        0 => true,
        1 => false,
        -1 => {
            let code = unsafe { WSAGetLastError() };
            if code == WSAEWOULDBLOCK {
                false
            } else {
                return Err(io::Error::from_raw_os_error(code));
            }
        }
        _ => return Err(io::Error::other("invalid TCP peek result")),
    };
    Ok(SocketSnapshot {
        unread_bytes,
        peer_closed,
    })
}
