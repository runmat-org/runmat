use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum TestDomainError {
    #[error("invalid {field}: {reason}")]
    InvalidField { field: &'static str, reason: String },
    #[error("duplicate {kind} identity {identity}")]
    DuplicateIdentity {
        kind: &'static str,
        identity: String,
    },
    #[error("test {test_id} references missing fixture group {fixture_group_id}")]
    MissingFixtureGroup {
        test_id: String,
        fixture_group_id: String,
    },
    #[error("event sequence expected {expected}, received {actual}")]
    EventSequence { expected: u64, actual: u64 },
    #[error("event stream changed run identity")]
    EventRunMismatch,
    #[error("event stream ended without a terminal run event")]
    IncompleteEventStream,
    #[error("protocol version {actual} is incompatible with supported version {supported}")]
    IncompatibleProtocol { actual: u16, supported: u16 },
    #[error("protocol payload {actual} bytes exceeds the {limit} byte limit")]
    ProtocolPayloadTooLarge { actual: usize, limit: usize },
    #[error("protocol collection {field} contains {actual} entries, exceeding limit {limit}")]
    ProtocolCollectionTooLarge {
        field: &'static str,
        actual: usize,
        limit: usize,
    },
}
