//! Bounded test protocol. Physical sources remain original immutable analysis objects.
use super::*;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use serde::{Deserialize, Serialize};
use std::io::Read;

const PREFIX: &[u8] = b"runmat-native-adaptive-process-witness/v1\0";
const LIMITS: runmat_canonical_codec::CanonicalLimits =
    runmat_canonical_codec::CanonicalLimits::new(4 * 1024 * 1024, 200_000, 65_536, 64);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(super) enum SourceAdmissionCase {
    Exact,
    RequiredMinusOne,
    Cancelled,
    MissingSourceChunk,
    CorruptSourceChunk,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum Action {
    Produce,
    Consume {
        admission: SourceAdmissionCase,
        evidence_file: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub action: Action,
    pub requests: [ProgramExecutionRequest; 2],
    pub map: SolverMeshTransferMap,
    pub lineage: SolverMeshAdaptationLineage,
    pub accuracy: AccuracyPlan,
    pub completion_file: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Completion {
    pub producer_pid: u32,
    pub responses: [ProgramExecutionResponse; 2],
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Evidence {
    pub consumer_pid: u32,
    pub producer_pid: u32,
    pub iteration: StructuralAdaptationNativeIteration,
}

pub(super) fn write<T: Serialize>(path: &Path, domain: &str, value: &T) {
    let bytes = runmat_canonical_codec::encode_contract(PREFIX, domain, value, LIMITS).unwrap();
    std::fs::write(path, bytes).unwrap();
}
pub(super) fn read<T: serde::de::DeserializeOwned>(path: &Path, domain: &str) -> T {
    let file = std::fs::File::open(path).unwrap();
    let mut bytes = Vec::new();
    file.take(LIMITS.maximum_encoded_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= LIMITS.maximum_encoded_bytes);
    runmat_canonical_codec::decode_contract(PREFIX, domain, &bytes, LIMITS).unwrap()
}
