//! Private record representation of the existing typed execution policy.
use crate::execution::CallbackParallelMode;
use serde::{Deserialize, Deserializer, Serializer};
pub(in crate::builtins::math::optim::surrogateopt) fn serialize<S: Serializer>(
    value: &CallbackParallelMode,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_u8(match value {
        CallbackParallelMode::Off => 0,
        CallbackParallelMode::Auto => 1,
        CallbackParallelMode::On => 2,
    })
}
pub(in crate::builtins::math::optim::surrogateopt) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<CallbackParallelMode, D::Error> {
    match u8::deserialize(deserializer)? {
        0 => Ok(CallbackParallelMode::Off),
        1 => Ok(CallbackParallelMode::Auto),
        2 => Ok(CallbackParallelMode::On),
        _ => Err(serde::de::Error::custom("invalid saved parallel policy")),
    }
}
