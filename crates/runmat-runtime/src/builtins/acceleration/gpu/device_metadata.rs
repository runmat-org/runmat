//! Shared, read-only projection of the active acceleration provider's device metadata.

use runmat_accelerate_api::{ApiDeviceInfo, ProviderPrecision};
use runmat_builtins::GPU_DEVICE_ACTIVE_INDEX;
use runmat_value::{IntValue, StructValue, Value};

/// An absent provider is a typed outcome, not another builtin's error identifier.
pub(super) fn active_device_metadata() -> Option<ActiveDeviceMetadata> {
    let provider = runmat_accelerate_api::provider()?;
    Some(ActiveDeviceMetadata {
        info: provider.device_info_struct(),
        precision: provider.precision(),
    })
}

pub(super) struct ActiveDeviceMetadata {
    info: ApiDeviceInfo,
    precision: ProviderPrecision,
}

impl ActiveDeviceMetadata {
    pub(super) fn into_struct(self) -> StructValue {
        let mut result = runmat_value::UnboundedStructValueBuilder::new();
        result.insert("device_id", Value::Int(IntValue::U32(self.info.device_id)));
        result.insert("index", Value::Int(IntValue::U32(GPU_DEVICE_ACTIVE_INDEX)));
        result.insert("name", Value::string(self.info.name));
        result.insert("vendor", Value::string(self.info.vendor));
        if let Some(backend) = self.info.backend {
            result.insert("backend", Value::string(backend));
        }
        if let Some(bytes) = self.info.memory_bytes {
            result.insert("memory_bytes", Value::Int(IntValue::U64(bytes)));
        }
        result.insert(
            "precision",
            Value::string(
                match self.precision {
                    ProviderPrecision::F64 => "double",
                    ProviderPrecision::F32 => "single",
                }
                .to_string(),
            ),
        );
        result.insert(
            "supports_double",
            Value::Bool(matches!(self.precision, ProviderPrecision::F64)),
        );
        result.finish()
    }
}
