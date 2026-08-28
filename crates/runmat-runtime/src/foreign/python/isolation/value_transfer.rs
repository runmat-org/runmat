use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    PythonValueTransfer, PythonWireError, PYTHON_HOST_INLINE_VALUE_BYTES,
    PYTHON_HOST_MAX_SNAPSHOT_BYTES,
};
use crate::execution::value_codec::{decode_inline_value, encode_inline_value};

pub(super) fn encode_portable(
    value: &Value,
    store: &SharedSnapshotStore,
) -> Result<PythonValueTransfer, PythonWireError> {
    let payload = encode_inline_value(value)
        .map_err(|error| wire_error("RunMat:Python:ValueEncode", error.to_string()))?;
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| wire_error("RunMat:Python:ValueEncode", error.to_string()))?;
    if encoded.len() <= PYTHON_HOST_INLINE_VALUE_BYTES {
        Ok(PythonValueTransfer::Inline(payload))
    } else if encoded.len() as u64 <= PYTHON_HOST_MAX_SNAPSHOT_BYTES {
        store
            .publish(&encoded)
            .map(PythonValueTransfer::Snapshot)
            .map_err(|error| wire_error("RunMat:Python:ValueSnapshot", error.to_string()))
    } else {
        Err(wire_error(
            "RunMat:Python:ValueSnapshot",
            format!(
                "encoded value exceeds the {PYTHON_HOST_MAX_SNAPSHOT_BYTES}-byte snapshot limit"
            ),
        ))
    }
}

pub(super) fn decode_portable(
    transfer: &PythonValueTransfer,
    store: &SharedSnapshotStore,
) -> Result<Value, PythonWireError> {
    let payload = match transfer {
        PythonValueTransfer::Inline(payload) => payload.clone(),
        PythonValueTransfer::Snapshot(descriptor) => {
            let bytes = store
                .consume(descriptor, PYTHON_HOST_MAX_SNAPSHOT_BYTES)
                .map_err(|error| wire_error("RunMat:Python:ValueSnapshot", error.to_string()))?;
            serde_json::from_slice(&bytes)
                .map_err(|error| wire_error("RunMat:Python:ValueDecode", error.to_string()))?
        }
    };
    payload
        .validate(runmat_execution::value::ValueLimits {
            max_inline_bytes: PYTHON_HOST_MAX_SNAPSHOT_BYTES,
            ..Default::default()
        })
        .map_err(|error| wire_error("RunMat:Python:ValueDecode", error.to_string()))?;
    decode_inline_value(&payload)
        .map_err(|error| wire_error("RunMat:Python:ValueDecode", error.to_string()))
}

pub(super) fn wire_error(identifier: &str, message: impl Into<String>) -> PythonWireError {
    PythonWireError {
        identifier: identifier.into(),
        message: message.into(),
        python: None,
    }
}
