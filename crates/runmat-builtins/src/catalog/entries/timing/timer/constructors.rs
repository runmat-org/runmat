//! Timer construction identity.

use super::{contract, documentation, inference};
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TIMER];

timer_entry!(
    TIMER,
    "timer",
    documentation::TIMER,
    &contract::TIMER_DESCRIPTOR,
    inference::constructor,
    ExecutionStackRequirement::Any,
    &contract::TIMER_EXTENSIONS,
    &contract::TIMER_INTEGER_CAPABILITIES,
    false
);
