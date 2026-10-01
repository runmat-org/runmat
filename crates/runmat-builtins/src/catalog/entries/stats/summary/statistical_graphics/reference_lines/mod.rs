//! Reference and fitted line identities over the shared graphics domain.
mod lsline;
mod refline;
mod facts;

use crate::BuiltinErrorDescriptor;

pub use lsline::*;
pub use refline::*;

/// Canonical public identity retained through graphics admission and error
/// presentation; text labels are never used to redispatch semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReferenceLineOwner {
    Refline,
    Lsline,
}

impl ReferenceLineOwner {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Refline => "refline",
            Self::Lsline => "lsline",
        }
    }

    pub const fn invalid_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Refline => &refline::REFLINE_ERROR_INVALID_ARGUMENT,
            Self::Lsline => &lsline::LSLINE_ERROR_INVALID_ARGUMENT,
        }
    }

    pub const fn internal_descriptor(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Refline => &refline::REFLINE_ERROR_INTERNAL,
            Self::Lsline => &lsline::LSLINE_ERROR_INTERNAL,
        }
    }
}
