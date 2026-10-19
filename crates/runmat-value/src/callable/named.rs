use std::fmt;
use std::ops::Deref;

use super::CallableInstanceId;

/// A named handle's spelling is semantic data; its instance is separate,
/// process-local identity used only to retain one physical callback proxy.
#[derive(Clone, Debug)]
pub struct NamedFunctionHandle {
    name: String,
    instance: CallableInstanceId,
}

impl NamedFunctionHandle {
    pub fn new(name: String) -> Self {
        Self {
            name,
            instance: CallableInstanceId::mint(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn into_name(self) -> String {
        self.name
    }

    pub fn instance_id(&self) -> CallableInstanceId {
        self.instance
    }
}

impl Deref for NamedFunctionHandle {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.name
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
