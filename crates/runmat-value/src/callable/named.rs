use std::fmt;
use std::ops::Deref;

use crate::Utf8Text;

use super::CallableInstanceId;

/// A named handle's spelling is semantic data; its instance is separate,
/// process-local identity used only to retain one physical callback proxy.
#[derive(Clone, Debug)]
pub struct NamedFunctionHandle {
    name: Utf8Text,
    instance: CallableInstanceId,
}

impl NamedFunctionHandle {
    pub fn new(name: String) -> Self {
        Self::from_text_owner(Utf8Text::new(name))
    }

    /// Retains the supplied original text owner without copying its backing.
    pub fn from_text_owner(name: Utf8Text) -> Self {
        Self {
            name,
            instance: CallableInstanceId::mint(),
        }
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub(crate) fn name_unclaimed_bytes(
        &self,
    ) -> Result<Option<usize>, crate::ExactHostNumericAllocationError> {
        self.name.unclaimed_bytes()
    }
    pub(crate) fn try_promote_name_claim(
        &self,
        permit: &dyn crate::ExactHostAllocationPermit,
    ) -> Result<(), crate::ExactHostNumericAllocationError> {
        self.name
            .clone()
            .try_admit_exact(permit)
            .map(|_| ())
            .map_err(|error| error.source())
    }

    pub fn into_text_owner(self) -> Utf8Text {
        self.name
    }

    pub fn into_name_unbounded(self) -> String {
        self.name.into_string_unbounded()
    }

    pub fn instance_id(&self) -> CallableInstanceId {
        self.instance
    }
}

impl Deref for NamedFunctionHandle {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.name.as_str()
    }
}

impl From<String> for NamedFunctionHandle {
    fn from(name: String) -> Self {
        Self::new(name)
    }
}

impl From<&str> for NamedFunctionHandle {
    fn from(name: &str) -> Self {
        Self::new(name.to_owned())
    }
}

impl PartialEq<str> for NamedFunctionHandle {
    fn eq(&self, other: &str) -> bool {
        self.name.as_str() == other
    }
}

impl PartialEq<&str> for NamedFunctionHandle {
    fn eq(&self, other: &&str) -> bool {
        self.name.as_str() == *other
    }
}

impl PartialEq for NamedFunctionHandle {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl fmt::Display for NamedFunctionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.fmt(f)
    }
}
