use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    NativeValueTransfer, NativeWireError, NATIVE_FFI_HOST_INLINE_VALUE_BYTES,
    NATIVE_FFI_HOST_MAX_SNAPSHOT_BYTES,
};
use crate::execution::value_codec::{decode_inline_value_rooted, encode_inline_value};
use crate::execution::RootedValueMember;

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
) -> Result<RootedValueMember, NativeWireError> {
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
    decode_inline_value_rooted(&payload)
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))
}

pub(super) fn wire_error(identifier: &str, message: impl Into<String>) -> NativeWireError {
    NativeWireError {
        identifier: identifier.into(),
        message: message.into(),
        cause: None,
    }
}

pub(super) fn root_plain_value(value: Value) -> Result<RootedValueMember, NativeWireError> {
    RootedValueMember::root(value)
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RootedValuesBuilder;

    #[test]
    fn decoded_batch_retains_earlier_managed_member_until_terminal_owner_retires() {
        runmat_gc::gc_test_context(|| {
            let source = runmat_gc::gc_allocate_rooted(Value::Num(41.0)).unwrap();
            let target = source.handle();
            let mut decoded = RootedValuesBuilder::with_capacity(2);
            decoded.push_rooted(
                root_plain_value(Value::HandleObject(runmat_value::HandleRef {
                    class_name: "NativeDecoded".into(),
                    target,
                    valid: true,
                }))
                .unwrap(),
            );
            source.unroot().unwrap();
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_ok());

            decoded.push_rooted(root_plain_value(Value::Num(1.0)).unwrap());
            let completed = decoded.finish().unwrap();
            assert_eq!(completed.values().len(), 2);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_ok());

            drop(completed);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&target, |_| ()).is_err());
        });
    }
}
