//! Pure resolution of one prospective workspace origin against the installed
//! compiler table and the already staged suffix.

use crate::abi::WorkspaceBindingKey;
use crate::session::workspace::global_key::matches_session_global_target;
use crate::session::SessionWorkspaceBinding;
use runmat_runtime::workspace::{GlobalBindingTarget, WorkspaceName, WorkspacePublicationOrigin};

use super::patch::{
    PendingBindingAction, PendingBindingKey, PendingBindingPatch, PendingBindingSlot,
};
use super::BindingPublication;

#[derive(Clone, Copy)]
pub(super) enum PlannedKey<'a> {
    Preserve,
    Source(&'a str),
    Interactive(&'a str),
    Global(&'a GlobalBindingTarget),
}

pub(super) struct PlannedMutation<'a> {
    pub(super) name: Option<&'a str>,
    pub(super) retained_name: Option<WorkspaceName>,
    pub(super) slot: PendingBindingSlot,
    pub(super) key: PlannedKey<'a>,
    pub(super) introduces_binding: bool,
    pub(super) retire: bool,
    pub(super) clear: bool,
    pub(super) noop: bool,
}

#[derive(Clone, Copy)]
enum EffectiveSlot {
    Known(usize),
    Dynamic,
}

struct EffectiveBinding<'a> {
    slot: EffectiveSlot,
    key: &'a WorkspaceBindingKey,
}

pub(super) fn plan<'a>(
    publication: &BindingPublication,
    patch: Option<&PendingBindingPatch>,
    installed: Option<(&WorkspaceName, &SessionWorkspaceBinding)>,
    origin: WorkspacePublicationOrigin<'a>,
) -> PlannedMutation<'a> {
    if matches!(origin, WorkspacePublicationOrigin::Cleared) {
        return PlannedMutation {
            name: None,
            retained_name: None,
            slot: PendingBindingSlot::Preserve,
            key: PlannedKey::Preserve,
            introduces_binding: false,
            retire: false,
            clear: true,
            noop: false,
        };
    }
    let name = origin
        .name()
        .expect("non-clear publication origin is named");
    let retained_name = patch
        .and_then(|patch| patch.retained_name(name))
        .or_else(|| installed.map(|(name, _)| name.clone()));
    let effective = effective_binding(patch, installed.map(|(_, binding)| binding), name);
    let introduces_binding =
        patch.is_some_and(PendingBindingPatch::was_cleared) || installed.is_none();

    let (slot, desired_key, retire) = match origin {
        WorkspacePublicationOrigin::ExecutorSlot { slot, global, .. } => (
            PendingBindingSlot::Explicit(slot),
            global.map_or(PlannedKey::Source(name), PlannedKey::Global),
            false,
        ),
        WorkspacePublicationOrigin::DynamicEffect {
            global: Some(target),
            ..
        }
        | WorkspacePublicationOrigin::BindGlobal { target, .. } => (
            effective.as_ref().map_or(PendingBindingSlot::Dynamic, |_| {
                PendingBindingSlot::Preserve
            }),
            PlannedKey::Global(target),
            false,
        ),
        WorkspacePublicationOrigin::DynamicEffect { global: None, .. } => (
            effective.as_ref().map_or(PendingBindingSlot::Dynamic, |_| {
                PendingBindingSlot::Preserve
            }),
            PlannedKey::Interactive(name),
            false,
        ),
        WorkspacePublicationOrigin::Retired { .. }
        | WorkspacePublicationOrigin::UnbindGlobal { .. } => {
            (PendingBindingSlot::Preserve, PlannedKey::Preserve, true)
        }
        WorkspacePublicationOrigin::Cleared => unreachable!(),
    };

    if retire {
        return PlannedMutation {
            name: Some(name),
            retained_name,
            slot,
            key: PlannedKey::Preserve,
            introduces_binding: false,
            retire: true,
            clear: false,
            noop: effective.is_none(),
        };
    }
    let key_matches = effective
        .as_ref()
        .is_some_and(|binding| desired_key.matches(publication, binding.key));
    let slot_matches = match (slot, effective.as_ref().map(|binding| binding.slot)) {
        (PendingBindingSlot::Preserve, Some(_)) => true,
        (PendingBindingSlot::Explicit(expected), Some(EffectiveSlot::Known(actual))) => {
            expected == actual
        }
        _ => false,
    };
    PlannedMutation {
        name: Some(name),
        retained_name,
        slot,
        key: if key_matches {
            PlannedKey::Preserve
        } else {
            desired_key
        },
        introduces_binding,
        retire: false,
        clear: false,
        noop: key_matches && slot_matches,
    }
}

impl PlannedKey<'_> {
    fn matches(&self, publication: &BindingPublication, key: &WorkspaceBindingKey) -> bool {
        match self {
            Self::Preserve => true,
            Self::Source(name) => publication.source_key_matches(key, name),
            Self::Interactive(name) => publication.interactive_key_matches(key, name),
            Self::Global(target) => matches_session_global_target(key, publication.session, target),
        }
    }
}

fn effective_binding<'a>(
    patch: Option<&'a PendingBindingPatch>,
    installed: Option<&'a SessionWorkspaceBinding>,
    name: &str,
) -> Option<EffectiveBinding<'a>> {
    let baseline = (!patch.is_some_and(PendingBindingPatch::was_cleared))
        .then_some(installed)
        .flatten()
        .map(|binding| EffectiveBinding {
            slot: EffectiveSlot::Known(binding.slot),
            key: &binding.key,
        });
    match patch.and_then(|patch| patch.current_action(name)) {
        None => baseline,
        Some(PendingBindingAction::Retire) => None,
        Some(PendingBindingAction::Upsert { slot, key, .. }) => {
            let slot = match slot {
                PendingBindingSlot::Preserve => baseline.as_ref()?.slot,
                PendingBindingSlot::Explicit(slot) => EffectiveSlot::Known(*slot),
                PendingBindingSlot::Dynamic => EffectiveSlot::Dynamic,
            };
            let key = match key {
                PendingBindingKey::Preserve => baseline.as_ref()?.key,
                PendingBindingKey::Replace(key) => key.as_key(),
            };
            Some(EffectiveBinding { slot, key })
        }
    }
}
