//! Timer discovery identities.

use super::{contract, documentation, inference};
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TIMERFIND, &TIMERFINDALL];

timer_entry!(
    TIMERFIND,
    "timerfind",
    documentation::TIMERFIND,
    &contract::TIMERFIND_DESCRIPTOR,
    inference::find,
    ExecutionStackRequirement::Any,
    &contract::TIMERFIND_EXTENSIONS,
    &contract::TIMERFIND_INTEGER_CAPABILITIES,
    false
);
timer_entry!(
    TIMERFINDALL,
    "timerfindall",
    documentation::TIMERFINDALL,
    &contract::TIMERFINDALL_DESCRIPTOR,
    inference::find,
    ExecutionStackRequirement::Any,
    &contract::TIMERFIND_EXTENSIONS,
    &contract::TIMERFIND_INTEGER_CAPABILITIES,
    false
);
