//! Compiler identity follows the canonical global storage target, while the
//! workspace binding table remains indexed by the source-visible spelling.

use crate::abi::{GlobalScopeKey, WorkspaceBindingKey};
use runmat_runtime::workspace::GlobalBindingTarget;
use uuid::Uuid;

pub(in crate::session) fn admission_error(
    context: &runmat_runtime::context::RuntimeContext,
    cause: runmat_runtime::context::host_allocation::HostAllocationError,
) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(format!("global binding admission failed: {cause}"))
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

pub(super) fn session_global_key(
    session: Uuid,
    target: &GlobalBindingTarget,
) -> WorkspaceBindingKey {
    match target {
        GlobalBindingTarget::Named(name) => WorkspaceBindingKey::Global {
            scope: GlobalScopeKey::Session(session),
            name: runmat_hir::BindingName(name.as_str().to_owned()),
        },
        GlobalBindingTarget::LegacySlot(slot) => WorkspaceBindingKey::GlobalSlot {
            scope: GlobalScopeKey::Session(session),
            slot: *slot,
        },
    }
}

pub(super) fn matches_session_global_target(
    key: &WorkspaceBindingKey,
    session: Uuid,
    target: &GlobalBindingTarget,
) -> bool {
    match (key, target) {
        (
            WorkspaceBindingKey::Global {
                scope: GlobalScopeKey::Session(bound_session),
                name,
            },
            GlobalBindingTarget::Named(target_name),
        ) => *bound_session == session && name.0 == target_name.as_str(),
        (
            WorkspaceBindingKey::GlobalSlot {
                scope: GlobalScopeKey::Session(bound_session),
                slot,
            },
            GlobalBindingTarget::LegacySlot(target_slot),
        ) => *bound_session == session && slot == target_slot,
        _ => false,
    }
}
