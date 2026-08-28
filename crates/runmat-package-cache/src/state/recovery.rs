use crate::materialize::MaterializationState;
use crate::state::CacheState;
use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "kebab-case")]
pub enum RecoveryAction {
    DropExpiredLease { lease_id: crate::LeaseId },
    DropInterruptedStaging { digest: ContentDigest },
    RemoveMissingObject { digest: ContentDigest },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPlan {
    pub actions: Vec<RecoveryAction>,
}

impl RecoveryPlan {
    pub fn inspect(
        state: &CacheState,
        now_ms: u64,
        physically_missing: impl IntoIterator<Item = ContentDigest>,
    ) -> Self {
        let mut actions = Vec::new();
        actions.extend(
            state
                .leases
                .values()
                .filter(|lease| !lease.is_active_at(now_ms))
                .map(|lease| RecoveryAction::DropExpiredLease {
                    lease_id: lease.id.clone(),
                }),
        );
        actions.extend(
            state
                .materializations
                .iter()
                .filter(|(_, materialization)| {
                    matches!(materialization.state, MaterializationState::Staging { .. })
                })
                .map(|(digest, _)| RecoveryAction::DropInterruptedStaging {
                    digest: digest.clone(),
                }),
        );
        actions.extend(
            physically_missing
                .into_iter()
                .map(|digest| RecoveryAction::RemoveMissingObject { digest }),
        );
        actions.sort();
        Self { actions }
    }

    pub fn apply(&self, state: &mut CacheState) {
        for action in &self.actions {
            match action {
                RecoveryAction::DropExpiredLease { lease_id } => {
                    state.leases.remove(lease_id);
                    state
                        .materializations
                        .retain(|_, materialization| &materialization.lease != lease_id);
                }
                RecoveryAction::DropInterruptedStaging { digest } => {
                    state.materializations.remove(digest);
                }
                RecoveryAction::RemoveMissingObject { digest } => {
                    remove_object_closure(state, digest);
                }
            }
        }
    }
}

fn remove_object_closure(state: &mut CacheState, initially_missing: &ContentDigest) {
    let mut remove = std::collections::BTreeSet::from([initially_missing.clone()]);
    loop {
        let dependent = state.objects.iter().find_map(|(digest, object)| {
            (!remove.contains(digest)
                && object
                    .references()
                    .iter()
                    .any(|reference| remove.contains(reference)))
            .then(|| digest.clone())
        });
        match dependent {
            Some(digest) => {
                remove.insert(digest);
            }
            None => break,
        }
    }
    for digest in &remove {
        state.objects.remove(digest);
        state.access.remove(digest);
        state.materializations.remove(digest);
    }
    state
        .leases
        .retain(|_, lease| lease.objects.is_disjoint(&remove));
    state.pins.retain(|_, pin| pin.objects.is_disjoint(&remove));
    let live_leases: std::collections::BTreeSet<_> = state.leases.keys().cloned().collect();
    state
        .materializations
        .retain(|_, materialization| live_leases.contains(&materialization.lease));
}
