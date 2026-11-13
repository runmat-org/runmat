use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use tokio::net::{UnixListener, UnixStream};

use super::{
    BlockingControl, ControlConnection, ControlReader, ControlWriter, PrivateControlEndpoint,
};

pub type BlockingReader = std::os::unix::net::UnixStream;
pub type BlockingWriter = std::os::unix::net::UnixStream;
pub type Reader = UnixStream;
pub type Writer = UnixStream;

pub(super) struct Listener {
    endpoint: PrivateControlEndpoint,
    request: UnixListener,
    response: UnixListener,
    _namespace: tempfile::TempDir,
}

pub(super) fn validate_endpoint(endpoint: &PrivateControlEndpoint) -> io::Result<()> {
    if endpoint.request == endpoint.response {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "control channels must be distinct",
        ));
    }
    for address in [&endpoint.request, &endpoint.response] {
        let path = Path::new(address);
        // sockaddr_un includes the terminating NUL in its pathname capacity.
        let capacity = std::mem::size_of::<libc::sockaddr_un>()
            - std::mem::offset_of!(libc::sockaddr_un, sun_path);
        if !path.is_absolute()
            || address.as_bytes().contains(&0)
            || path.as_os_str().as_bytes().len() >= capacity
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid local control socket path",
            ));
        }
    }
    Ok(())
}

impl Listener {
    pub(super) fn bind() -> io::Result<Self> {
        let namespace = tempfile::Builder::new().prefix("rm-ipc-").tempdir()?;
        // Set before creating either socket; no process-wide umask changes.
        std::fs::set_permissions(namespace.path(), std::fs::Permissions::from_mode(0o700))?;
        let address = |name| -> io::Result<String> {
            namespace
                .path()
                .join(name)
                .into_os_string()
                .into_string()
                .map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "control socket path is not Unicode",
                    )
                })
        };
        let endpoint = PrivateControlEndpoint {
            request: address("q")?,
            response: address("r")?,
        };
        validate_endpoint(&endpoint)?;
        let request = UnixListener::bind(&endpoint.request)?;
        let response = UnixListener::bind(&endpoint.response)?;
        Ok(Self {
            endpoint,
            request,
            response,
            _namespace: namespace,
        })
    }

    pub(super) fn endpoint(&self) -> &PrivateControlEndpoint {
        &self.endpoint
    }

    pub(super) async fn accept(self) -> io::Result<ControlConnection> {
        let (writer, _) = self.request.accept().await?;
        let (reader, _) = self.response.accept().await?;
        // Dropping the listeners removes the bootstrap paths, not the accepted
        // sockets. No temporary payload file or duplicate workspace is kept.
        Ok(ControlConnection {
            reader: ControlReader(reader),
            writer: ControlWriter(Some(writer)),
        })
    }
}

pub(super) fn connect_blocking(endpoint: &PrivateControlEndpoint) -> io::Result<BlockingControl> {
    let reader = BlockingReader::connect(&endpoint.request)?;
    let writer = BlockingWriter::connect(&endpoint.response)?;
    Ok(BlockingControl { reader, writer })
}

pub(super) async fn connect(endpoint: &PrivateControlEndpoint) -> io::Result<ControlConnection> {
    let reader = UnixStream::connect(&endpoint.request).await?;
    let writer = UnixStream::connect(&endpoint.response).await?;
    Ok(ControlConnection {
        reader: ControlReader(reader),
        writer: ControlWriter(Some(writer)),
    })
}
