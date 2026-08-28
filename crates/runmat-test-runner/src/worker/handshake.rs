use runmat_test::protocol::{negotiate, ProtocolHandshake, ProtocolLimits};

use crate::{RunnerError, RunnerResult};

pub fn validate_handshake(
    local: &ProtocolHandshake,
    remote: &ProtocolHandshake,
) -> RunnerResult<ProtocolLimits> {
    negotiate(local, remote).map_err(|error| RunnerError::Protocol(error.to_string()))
}
