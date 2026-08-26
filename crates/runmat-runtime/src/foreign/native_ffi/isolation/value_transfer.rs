use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    NativeValueTransfer, NativeWireError, NATIVE_FFI_HOST_INLINE_VALUE_BYTES,
    NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES,
};
use crate::execution::value_codec::{decode_inline_value, encode_inline_value};

pub(super) fn encode_portable(
    value: &Value,
    store: &SharedSnapshotStore,
) -> Result<NativeValueTransfer, NativeWireError> {
    let payload = encode_inline_value(value)
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueEncode", error.to_string()))?;
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueEncode", error.to_string()))?;
    if encoded.len() <= NATIVE_FFI_HOST_INLINE_VALUE_BYTES {
        Ok(NativeValueTransfer::Inline(payload))
    } else if encoded.len() as u64 <= NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES {
        store
            .publish(&encoded)
            .map(NativeValueTransfer::Snapshot)
            .map_err(|error| wire_error("RunMat:NativeFFI:ValueSnapshot", error.to_string()))
    } else {
        Err(wire_error(
            "RunMat:NativeFFI:ValueSnapshot",
            format!(
                "encoded value exceeds the {}-byte snapshot limit",
                NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES
            ),
        ))
    }
}

pub(super) fn decode_portable(
    transfer: &NativeValueTransfer,
    store: &SharedSnapshotStore,
) -> Result<Value, NativeWireError> {
    let payload = match transfer {
        NativeValueTransfer::Inline(payload) => payload.clone(),
        NativeValueTransfer::Snapshot(descriptor) => {
            let bytes = store
                .consume(descriptor, NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES)
                .map_err(|error| wire_error("RunMat:NativeFFI:ValueSnapshot", error.to_string()))?;
            serde_json::from_slice(&bytes)
                .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))?
        }
    };
    payload
        .validate(runmat_execution::value::ValueLimits {
            max_inline_bytes: NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES,
            ..Default::default()
        })
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))?;
    decode_inline_value(&payload)
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))
}

pub(super) fn wire_error(identifier: &str, message: impl Into<String>) -> NativeWireError {
    NativeWireError {
        identifier: identifier.into(),
        message: message.into(),
    }
}
