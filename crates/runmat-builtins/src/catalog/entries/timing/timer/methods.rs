//! Session-bound timer lifecycle method identities.

use super::{contract, documentation, inference};
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&START, &STARTAT, &STOP, &WAIT, &DELETE];

timer_entry!(
    START,
    "__runmat_timer_start",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::start,
    ExecutionStackRequirement::Process,
    &contract::TIMER_METHOD_EXTENSIONS,
    &[],
    true
);
timer_entry!(
    STARTAT,
    "__runmat_timer_startat",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_STARTAT_DESCRIPTOR,
    inference::startat,
    ExecutionStackRequirement::Process,
    &contract::TIMER_STARTAT_EXTENSIONS,
    &[],
    true
);
timer_entry!(
    STOP,
    "__runmat_timer_stop",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::stop,
    ExecutionStackRequirement::Process,
    &contract::TIMER_METHOD_EXTENSIONS,
    &[],
    true
);
timer_entry!(
    WAIT,
    "__runmat_timer_wait",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::wait,
    ExecutionStackRequirement::Process,
    &contract::TIMER_METHOD_EXTENSIONS,
    &[],
    true
);
timer_entry!(
    DELETE,
    "__runmat_timer_delete",
    documentation::INTERNAL_METHOD,
    &contract::TIMER_METHOD_DESCRIPTOR,
    inference::delete,
    ExecutionStackRequirement::Process,
    &contract::TIMER_METHOD_EXTENSIONS,
    &[],
    true
);
