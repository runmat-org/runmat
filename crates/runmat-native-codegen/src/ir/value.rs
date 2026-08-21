use super::{NativeLocalId, NativeValueId, NativeValueType};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBlockParameter {
    pub local: NativeLocalId,
    pub value: NativeValueId,
    pub value_type: NativeValueType,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOutput {
    pub value: NativeValueId,
    pub value_type: NativeValueType,
    /// MIR local receiving this output, when the operation publishes one.
    pub local: Option<NativeLocalId>,
}
