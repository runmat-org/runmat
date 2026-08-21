use runmat_test::protocol::ProtocolHandshake;

use crate::host::HostCapabilities;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub host: HostCapabilities,
    pub handshake: ProtocolHandshake,
}
