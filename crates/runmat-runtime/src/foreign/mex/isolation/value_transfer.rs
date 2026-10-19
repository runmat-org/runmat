use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    MexValueTransfer, MexWireError, MEX_HOST_INLINE_VALUE_BYTES, MEX_HOST_MAX_SNAPSHOT_BYTES,
};
use crate::execution::value_codec::{decode_inline_value_rooted, encode_inline_value};

pub(super) fn encode_value_transfer(
    value: &Value,
    store: &SharedSnapshotStore,
) -> Result<MexValueTransfer, MexWireError> {
    let payload = encode_inline_value(value).map_err(|error| {
        wire_error(
            "RunMat:MEX:ValueEncode",
            format!("could not encode value: {error}"),
        )
    })?;
    let encoded = serde_json::to_vec(&payload).map_err(|error| {
        wire_error(
            "RunMat:MEX:ValueEncode",
            format!("could not encode value snapshot: {error}"),
        )
    })?;
    if encoded.len() <= MEX_HOST_INLINE_VALUE_BYTES {
        Ok(MexValueTransfer::Inline(payload))
    } else if encoded.len() as u64 <= MEX_HOST_MAX_SNAPSHOT_BYTES {
        store
            .publish(&encoded)
            .map(MexValueTransfer::Snapshot)
            .map_err(|error| {
                wire_error(
                    "RunMat:MEX:ValueSnapshot",
                    format!("could not publish value snapshot: {error}"),
                )
            })
    } else {
        Err(wire_error(
            "RunMat:MEX:ValueSnapshot",
            format!(
                "encoded value exceeds the {}-byte snapshot limit",
                MEX_HOST_MAX_SNAPSHOT_BYTES
            ),
        ))
    }
}

pub(super) fn decode_value_transfer_rooted(
    transfer: &MexValueTransfer,
    store: &SharedSnapshotStore,
) -> Result<crate::execution::RootedValueMember, MexWireError> {
    let payload = match transfer {
        MexValueTransfer::Inline(payload) => payload.clone(),
        MexValueTransfer::Snapshot(descriptor) => {
            let bytes = store
                .consume(descriptor, MEX_HOST_MAX_SNAPSHOT_BYTES)
                .map_err(|error| {
                    wire_error(
                        "RunMat:MEX:ValueSnapshot",
                        format!("could not consume value snapshot: {error}"),
                    )
                })?;
            serde_json::from_slice(&bytes).map_err(|error| {
                wire_error(
                    "RunMat:MEX:ValueDecode",
                    format!("could not decode value snapshot: {error}"),
                )
            })?
        }
    };
    payload
        .validate(runmat_execution::value::ValueLimits {
            max_inline_bytes: MEX_HOST_MAX_SNAPSHOT_BYTES,
            ..Default::default()
        })
        .map_err(|error| {
            wire_error(
                "RunMat:MEX:ValueDecode",
                format!("value payload violates the transfer contract: {error}"),
            )
        })?;
    decode_inline_value_rooted(&payload).map_err(|error| {
        wire_error(
            "RunMat:MEX:ValueDecode",
            format!("could not decode value: {error}"),
        )
    })
}

pub(super) fn decode_value_transfers_rooted(
    transfers: &[MexValueTransfer],
    store: &SharedSnapshotStore,
) -> Result<crate::execution::RootedValues, MexWireError> {
    let mut outputs = crate::execution::RootedValuesBuilder::with_capacity(transfers.len());
    for transfer in transfers {
        outputs.push_rooted(decode_value_transfer_rooted(transfer, store)?);
    }
    outputs
        .finish()
        .map_err(|error| wire_error("RunMat:MEX:ValueRoot", error.to_string()))
}

pub(super) fn decode_value_transfer_rooted_value(
    transfer: &MexValueTransfer,
    store: &SharedSnapshotStore,
) -> Result<crate::execution::RootedValue, MexWireError> {
    decode_value_transfer_rooted(transfer, store)?
        .into_rooted_value()
        .map_err(|error| wire_error("RunMat:MEX:ValueRoot", error.to_string()))
}

fn wire_error(identifier: &str, message: impl Into<String>) -> MexWireError {
    MexWireError {
        identifier: identifier.into(),
        message: message.into(),
        dependency: None,
    }
}

#[cfg(test)]
mod tests {
    use runmat_value::{NumericStorage, Tensor};

    use super::*;

    #[test]
    fn large_typed_values_use_verified_snapshots() {
        let store = SharedSnapshotStore::create().unwrap();
        let value = Value::Tensor(
            Tensor::from_numeric_storage(
                NumericStorage::U64((0..32_768_u64).collect()),
                vec![256, 128],
            )
            .unwrap(),
        );
        let transfer = encode_value_transfer(&value, &store).unwrap();
        assert!(matches!(transfer, MexValueTransfer::Snapshot(_)));
        assert_eq!(
            decode_value_transfer_rooted_value(&transfer, &store)
                .unwrap()
                .value(),
            &value
        );
    }
}
