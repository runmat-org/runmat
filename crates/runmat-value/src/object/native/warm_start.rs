//! One immutable native data header; aliases share it without mutation.
use super::*;
impl ObjectInstance {
    pub fn with_active_set_warm_start_payload(self, data: ActiveSetWarmStartData) -> Self {
        self.with_native(NativeObjectPayload::ActiveSetWarmStart(
            SharedNativeState::new(data),
        ))
    }
    pub fn active_set_warm_start_payload(&self) -> Option<&ActiveSetWarmStartData> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::ActiveSetWarmStart(value)) => Some(value.get()),
            _ => None,
        }
    }
}
