use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::windows::named_pipe::{
    ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions,
};
use windows_sys::Win32::Foundation::LocalFree;
use windows_sys::Win32::Security::Authorization::ConvertStringSecurityDescriptorToSecurityDescriptorW;
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::Storage::FileSystem::{SECURITY_IDENTIFICATION, SECURITY_SQOS_PRESENT};

use super::{
    BlockingControl, ControlConnection, ControlReader, ControlWriter, PrivateControlEndpoint,
};

pub type BlockingReader = File;
pub type BlockingWriter = File;
pub enum Reader {
    Listener(NamedPipeServer),
    Connector(NamedPipeClient),
}
pub enum Writer {
    Listener(NamedPipeServer),
    Connector(NamedPipeClient),
}

impl AsyncRead for Reader {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Listener(pipe) => Pin::new(pipe).poll_read(context, buffer),
            Self::Connector(pipe) => Pin::new(pipe).poll_read(context, buffer),
        }
    }
}
impl AsyncWrite for Writer {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Self::Listener(pipe) => Pin::new(pipe).poll_write(context, bytes),
            Self::Connector(pipe) => Pin::new(pipe).poll_write(context, bytes),
        }
    }
    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Listener(pipe) => Pin::new(pipe).poll_flush(context),
            Self::Connector(pipe) => Pin::new(pipe).poll_flush(context),
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Self::Listener(pipe) => Pin::new(pipe).poll_shutdown(context),
            Self::Connector(pipe) => Pin::new(pipe).poll_shutdown(context),
        }
    }
}

pub(super) struct Listener {
    endpoint: PrivateControlEndpoint,
    request: NamedPipeServer,
    response: NamedPipeServer,
}

pub(super) fn validate_endpoint(endpoint: &PrivateControlEndpoint) -> io::Result<()> {
    if endpoint.request == endpoint.response {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "control channels must be distinct",
        ));
    }
    for name in [&endpoint.request, &endpoint.response] {
        let Some(leaf) = name.strip_prefix(r"\\.\pipe\runmat-control-") else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid local control pipe namespace",
            ));
        };
        if leaf.is_empty()
            || leaf.len() > 64
            || !leaf
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() || byte == b'-')
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid local control pipe name",
            ));
        }
    }
    Ok(())
}

struct Descriptor(*mut c_void);

impl Drop for Descriptor {
    fn drop(&mut self) {
        // SAFETY: the conversion API allocated this descriptor with LocalAlloc.
        unsafe {
            LocalFree(self.0);
        }
    }
}

fn bind_pipe(name: &str, inbound: bool) -> io::Result<NamedPipeServer> {
    let sddl: Vec<u16> = "D:P(A;;GA;;;OW)(A;;GA;;;SY)\0".encode_utf16().collect();
    let mut descriptor = Descriptor(std::ptr::null_mut());
    // SAFETY: terminated SDDL and live out-pointer; the descriptor's RAII owner
    // outlives CreateNamedPipe, which copies its creation-time security state.
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            1,
            &mut descriptor.0,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let mut attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor.0,
        bInheritHandle: 0,
    };
    let mut options = ServerOptions::new();
    options
        .first_pipe_instance(true)
        .reject_remote_clients(true)
        .access_inbound(inbound)
        .access_outbound(!inbound);
    // SAFETY: valid SECURITY_ATTRIBUTES and descriptor for this synchronous
    // creation call. Neither the handle nor bootstrap authority is inherited.
    unsafe {
        options.create_with_security_attributes_raw(
            name,
            (&mut attributes as *mut SECURITY_ATTRIBUTES).cast(),
        )
    }
}

impl Listener {
    pub(super) fn bind() -> io::Result<Self> {
        use rand::RngCore;
        let mut nonce = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let identity = u128::from_le_bytes(nonce);
        let endpoint = PrivateControlEndpoint {
            request: format!(r"\\.\pipe\runmat-control-{identity:032x}-0"),
            response: format!(r"\\.\pipe\runmat-control-{identity:032x}-1"),
        };
        let request = bind_pipe(&endpoint.request, false)?;
        let response = bind_pipe(&endpoint.response, true)?;
        Ok(Self {
            endpoint,
            request,
            response,
        })
    }

    pub(super) fn endpoint(&self) -> &PrivateControlEndpoint {
        &self.endpoint
    }

    pub(super) async fn accept(self) -> io::Result<ControlConnection> {
        self.request.connect().await?;
        self.response.connect().await?;
        Ok(ControlConnection {
            reader: ControlReader(Reader::Listener(self.response)),
            writer: ControlWriter(Some(Writer::Listener(self.request))),
        })
    }
}

pub(super) fn connect_blocking(endpoint: &PrivateControlEndpoint) -> io::Result<BlockingControl> {
    let reader = OpenOptions::new()
        .read(true)
        .custom_flags(SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION)
        .open(&endpoint.request)?;
    let writer = OpenOptions::new()
        .write(true)
        .custom_flags(SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION)
        .open(&endpoint.response)?;
    Ok(BlockingControl { reader, writer })
}

pub(super) async fn connect(endpoint: &PrivateControlEndpoint) -> io::Result<ControlConnection> {
    let reader = ClientOptions::new()
        .read(true)
        .write(false)
        .security_qos_flags(SECURITY_IDENTIFICATION)
        .open(&endpoint.request)?;
    let writer = ClientOptions::new()
        .read(false)
        .write(true)
        .security_qos_flags(SECURITY_IDENTIFICATION)
        .open(&endpoint.response)?;
    Ok(ControlConnection {
        reader: ControlReader(Reader::Connector(reader)),
        writer: ControlWriter(Some(Writer::Connector(writer))),
    })
}
