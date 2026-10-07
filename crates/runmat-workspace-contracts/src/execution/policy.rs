//! Durable ownership policy shared by native, browser and server authorities.
use super::{ExecutionAuthorityError, SessionId, TurnId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// JSON timestamps also travel through JavaScript hosts. Reject values that
// cannot round-trip exactly through their number representation.
const MAX_TIMESTAMP_MS: i64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ExecutionWorkId {
    pub session_id: SessionId,
    pub turn_id: TurnId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ExecutionOwnerId(pub [u8; 16]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ExecutionFence(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ExecutionLease {
    pub work: ExecutionWorkId,
    pub owner: ExecutionOwnerId,
    pub fence: ExecutionFence,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub expires_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "ts", ts(tag = "type", rename_all = "snake_case"))]
pub enum ExecutionLeaseCommand {
    Acquire {
        owner: ExecutionOwnerId,
        duration_ms: u32,
    },
    Renew {
        lease: ExecutionLease,
        duration_ms: u32,
    },
    Release {
        lease: ExecutionLease,
    },
    Complete {
        lease: ExecutionLease,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "ts", ts(tag = "type", rename_all = "snake_case"))]
pub enum ExecutionLeaseOutcome {
    Acquired {
        lease: ExecutionLease,
    },
    Renewed {
        lease: ExecutionLease,
    },
    Released,
    Completed,
    Busy {
        #[cfg_attr(feature = "ts", ts(type = "number"))]
        expires_at_ms: i64,
    },
    Lost,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "ts", ts(tag = "type", rename_all = "snake_case"))]
pub enum ExecutionLeaseState {
    Held {
        lease: ExecutionLease,
        #[cfg_attr(feature = "ts", ts(type = "number"))]
        observed_at_ms: i64,
    },
    Released {
        fence: ExecutionFence,
        #[cfg_attr(feature = "ts", ts(type = "number"))]
        observed_at_ms: i64,
    },
    Finished {
        fence: ExecutionFence,
        #[cfg_attr(feature = "ts", ts(type = "number"))]
        observed_at_ms: i64,
    },
}

impl ExecutionLeaseState {
    fn fence(&self) -> ExecutionFence {
        match self {
            Self::Held { lease, .. } => lease.fence,
            Self::Released { fence, .. } | Self::Finished { fence, .. } => *fence,
        }
    }
    fn observed_at_ms(&self) -> i64 {
        match self {
            Self::Held { observed_at_ms, .. }
            | Self::Released { observed_at_ms, .. }
            | Self::Finished { observed_at_ms, .. } => *observed_at_ms,
        }
    }
}

/// Authorities call this while holding the same transaction/lock that commits
/// the checkpoint. Calling it before a later unconditional write is unsafe.
pub fn execution_lease_is_current(
    state: Option<&ExecutionLeaseState>,
    claimed: &ExecutionLease,
    authority_now_ms: i64,
) -> bool {
    (0..=MAX_TIMESTAMP_MS).contains(&authority_now_ms)
        && claimed.fence.0 > 0
        && matches!(state, Some(ExecutionLeaseState::Held { lease, observed_at_ms })
            if lease.work == claimed.work && lease.owner == claimed.owner
                && lease.fence == claimed.fence
                && (0..=MAX_TIMESTAMP_MS).contains(observed_at_ms)
                && lease.expires_at_ms <= MAX_TIMESTAMP_MS
                && lease.expires_at_ms > authority_now_ms.max(*observed_at_ms))
}

/// Pure policy shared by persistence authorities. Persist a changed state and
/// its associated effects in one transaction before returning the outcome.
pub fn transition_execution_lease(
    work: &ExecutionWorkId,
    current: Option<&ExecutionLeaseState>,
    command: &ExecutionLeaseCommand,
    authority_now_ms: i64,
) -> Result<(Option<ExecutionLeaseState>, ExecutionLeaseOutcome), ExecutionAuthorityError> {
    if work.session_id.0.is_empty()
        || work.turn_id.0.is_empty()
        || work.session_id.0.len() > 256
        || work.turn_id.0.len() > 256
        || work.session_id.0.contains('\0')
        || work.turn_id.0.contains('\0')
        || !(0..=MAX_TIMESTAMP_MS).contains(&authority_now_ms)
    {
        return Err(ExecutionAuthorityError::InvalidIdentity);
    }
    if let Some(state) = current {
        if state.fence().0 == 0
            || !(0..=MAX_TIMESTAMP_MS).contains(&state.observed_at_ms())
            || matches!(state, ExecutionLeaseState::Held { lease, .. } if lease.expires_at_ms > MAX_TIMESTAMP_MS)
            || matches!(state, ExecutionLeaseState::Held { lease, observed_at_ms } if lease.work != *work || lease.expires_at_ms <= *observed_at_ms)
        {
            return Err(ExecutionAuthorityError::InvalidState);
        }
    }
    let now = authority_now_ms.max(current.map_or(0, ExecutionLeaseState::observed_at_ms));
    let unchanged = |outcome| Ok((current.cloned(), outcome));
    if matches!(current, Some(ExecutionLeaseState::Finished { .. })) {
        return unchanged(ExecutionLeaseOutcome::Finished);
    }
    let expiry = |duration: u32| {
        if !(1_000..=300_000).contains(&duration) {
            return Err(ExecutionAuthorityError::InvalidDuration);
        }
        now.checked_add(i64::from(duration))
            .filter(|expiry| *expiry <= MAX_TIMESTAMP_MS)
            .ok_or(ExecutionAuthorityError::ExpiryOverflow)
    };
    if let ExecutionLeaseCommand::Acquire { owner, duration_ms } = command {
        let expires_at_ms = expiry(*duration_ms)?;
        if let Some(ExecutionLeaseState::Held { lease, .. }) = current {
            if lease.expires_at_ms > now {
                return Ok((
                    Some(ExecutionLeaseState::Held {
                        lease: lease.clone(),
                        observed_at_ms: now,
                    }),
                    if lease.owner == *owner {
                        ExecutionLeaseOutcome::Acquired {
                            lease: lease.clone(),
                        }
                    } else {
                        ExecutionLeaseOutcome::Busy {
                            expires_at_ms: lease.expires_at_ms,
                        }
                    },
                ));
            }
        }
        let fence = ExecutionFence(
            current
                .map_or(0, |state| state.fence().0)
                .checked_add(1)
                .ok_or(ExecutionAuthorityError::FenceOverflow)?,
        );
        let lease = ExecutionLease {
            work: work.clone(),
            owner: *owner,
            fence,
            expires_at_ms,
        };
        return Ok((
            Some(ExecutionLeaseState::Held {
                lease: lease.clone(),
                observed_at_ms: now,
            }),
            ExecutionLeaseOutcome::Acquired { lease },
        ));
    }
    let claimed = match command {
        ExecutionLeaseCommand::Renew { lease, .. }
        | ExecutionLeaseCommand::Release { lease }
        | ExecutionLeaseCommand::Complete { lease } => lease,
        ExecutionLeaseCommand::Acquire { .. } => unreachable!(),
    };
    let Some(ExecutionLeaseState::Held { lease, .. }) = current else {
        return unchanged(ExecutionLeaseOutcome::Lost);
    };
    if lease.expires_at_ms <= now {
        // Observed expiry is durable. A subsequent authority clock rollback
        // must not revive an already rejected owner.
        return Ok((
            Some(ExecutionLeaseState::Released {
                fence: lease.fence,
                observed_at_ms: now,
            }),
            ExecutionLeaseOutcome::Lost,
        ));
    }
    if claimed.work != *work || claimed.owner != lease.owner || claimed.fence != lease.fence {
        return unchanged(ExecutionLeaseOutcome::Lost);
    }
    match command {
        ExecutionLeaseCommand::Renew { duration_ms, .. } => {
            let mut lease = lease.clone();
            lease.expires_at_ms = expiry(*duration_ms)?.max(lease.expires_at_ms);
            Ok((
                Some(ExecutionLeaseState::Held {
                    lease: lease.clone(),
                    observed_at_ms: now,
                }),
                ExecutionLeaseOutcome::Renewed { lease },
            ))
        }
        ExecutionLeaseCommand::Release { .. } => Ok((
            Some(ExecutionLeaseState::Released {
                fence: lease.fence,
                observed_at_ms: now,
            }),
            ExecutionLeaseOutcome::Released,
        )),
        ExecutionLeaseCommand::Complete { .. } => Ok((
            Some(ExecutionLeaseState::Finished {
                fence: lease.fence,
                observed_at_ms: now,
            }),
            ExecutionLeaseOutcome::Completed,
        )),
        ExecutionLeaseCommand::Acquire { .. } => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_values_round_trip_and_reject_unrepresentable_time() {
        let json =
            serde_json::json!({ "type": "acquire", "owner": ([1; 16]), "duration_ms": 1000 });
        let command: ExecutionLeaseCommand = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(command).unwrap(), json);
        assert!(
            serde_json::from_value::<ExecutionLeaseCommand>(serde_json::json!({
                "type": "acquire", "owner": ([1; 16]), "duration_ms": 1000, "admin": true
            }))
            .is_err()
        );
        assert!(matches!(
            transition_execution_lease(
                &work(),
                None,
                &ExecutionLeaseCommand::Acquire {
                    owner: ExecutionOwnerId([1; 16]),
                    duration_ms: 1000
                },
                MAX_TIMESTAMP_MS
            ),
            Err(ExecutionAuthorityError::ExpiryOverflow)
        ));
        assert!(matches!(
            transition_execution_lease(
                &work(),
                None,
                &ExecutionLeaseCommand::Acquire {
                    owner: ExecutionOwnerId([1; 16]),
                    duration_ms: 1000
                },
                MAX_TIMESTAMP_MS + 1
            ),
            Err(ExecutionAuthorityError::InvalidIdentity)
        ));
    }
    #[cfg(feature = "ts")]
    #[test]
    fn typescript_preserves_wire_tags_and_json_numbers() {
        use ts_rs::TS;
        let command = ExecutionLeaseCommand::decl();
        for tag in ["acquire", "renew", "release", "complete"] {
            assert!(
                command.contains(&format!("\"type\": \"{tag}\"")),
                "{command}"
            );
        }
        assert!(ExecutionLeaseState::decl().contains("\"type\": \"finished\""));
        for declaration in [
            ExecutionLease::decl(),
            ExecutionLeaseState::decl(),
            ExecutionLeaseOutcome::decl(),
            ExecutionFence::decl(),
        ] {
            assert!(
                !declaration.contains("bigint"),
                "JSON number changed representation: {declaration}"
            );
        }
    }
    fn work() -> ExecutionWorkId {
        ExecutionWorkId {
            session_id: SessionId("session".into()),
            turn_id: TurnId("turn".into()),
        }
    }
    fn acquire(
        state: Option<&ExecutionLeaseState>,
        owner: u8,
        now: i64,
    ) -> (Option<ExecutionLeaseState>, ExecutionLease) {
        let (state, outcome) = transition_execution_lease(
            &work(),
            state,
            &ExecutionLeaseCommand::Acquire {
                owner: ExecutionOwnerId([owner; 16]),
                duration_ms: 1_000,
            },
            now,
        )
        .unwrap();
        let ExecutionLeaseOutcome::Acquired { lease } = outcome else {
            panic!("expected ownership");
        };
        (state, lease)
    }
    #[test]
    fn expiry_fences_the_previous_owner_even_when_it_reacquires() {
        let (state, first) = acquire(None, 1, 100);
        assert!(matches!(
            transition_execution_lease(
                &work(),
                state.as_ref(),
                &ExecutionLeaseCommand::Acquire {
                    owner: ExecutionOwnerId([2; 16]),
                    duration_ms: 1_000
                },
                1099
            )
            .unwrap()
            .1,
            ExecutionLeaseOutcome::Busy { .. }
        ));
        let (state, next) = acquire(state.as_ref(), 1, 1100);
        assert_eq!(next.fence.0, first.fence.0 + 1);
        for command in [
            ExecutionLeaseCommand::Renew {
                lease: first.clone(),
                duration_ms: 1_000,
            },
            ExecutionLeaseCommand::Release {
                lease: first.clone(),
            },
            ExecutionLeaseCommand::Complete { lease: first },
        ] {
            assert_eq!(
                transition_execution_lease(&work(), state.as_ref(), &command, 1101)
                    .unwrap()
                    .1,
                ExecutionLeaseOutcome::Lost
            );
        }
    }
    #[test]
    fn completion_and_release_cannot_be_revived_by_a_late_heartbeat() {
        let (state, lease) = acquire(None, 1, 100);
        let (finished, result) = transition_execution_lease(
            &work(),
            state.as_ref(),
            &ExecutionLeaseCommand::Complete {
                lease: lease.clone(),
            },
            101,
        )
        .unwrap();
        assert_eq!(result, ExecutionLeaseOutcome::Completed);
        assert_eq!(
            transition_execution_lease(
                &work(),
                finished.as_ref(),
                &ExecutionLeaseCommand::Renew {
                    lease: lease.clone(),
                    duration_ms: 1_000
                },
                102
            )
            .unwrap()
            .1,
            ExecutionLeaseOutcome::Finished
        );
        let (released, _) = transition_execution_lease(
            &work(),
            state.as_ref(),
            &ExecutionLeaseCommand::Release {
                lease: lease.clone(),
            },
            101,
        )
        .unwrap();
        assert_eq!(
            transition_execution_lease(
                &work(),
                released.as_ref(),
                &ExecutionLeaseCommand::Renew {
                    lease,
                    duration_ms: 1_000
                },
                102
            )
            .unwrap()
            .1,
            ExecutionLeaseOutcome::Lost
        );
        assert_eq!(acquire(released.as_ref(), 2, 102).1.fence.0, 2);
    }
    #[test]
    fn renewals_never_shorten_ownership_and_clock_rollback_does_not_revive_it() {
        let (state, lease) = acquire(None, 1, 100);
        let (state, _) = transition_execution_lease(
            &work(),
            state.as_ref(),
            &ExecutionLeaseCommand::Renew {
                lease: lease.clone(),
                duration_ms: 10_000,
            },
            200,
        )
        .unwrap();
        let (state, _) = transition_execution_lease(
            &work(),
            state.as_ref(),
            &ExecutionLeaseCommand::Renew {
                lease: lease.clone(),
                duration_ms: 1_000,
            },
            150,
        )
        .unwrap();
        assert!(matches!(
            state,
            Some(ExecutionLeaseState::Held {
                lease: ExecutionLease {
                    expires_at_ms: 10200,
                    ..
                },
                observed_at_ms: 200
            })
        ));
        let (expired, outcome) = transition_execution_lease(
            &work(),
            state.as_ref(),
            &ExecutionLeaseCommand::Renew {
                lease: lease.clone(),
                duration_ms: 1_000,
            },
            10200,
        )
        .unwrap();
        assert_eq!(outcome, ExecutionLeaseOutcome::Lost);
        assert_eq!(
            transition_execution_lease(
                &work(),
                expired.as_ref(),
                &ExecutionLeaseCommand::Renew {
                    lease,
                    duration_ms: 1_000
                },
                200
            )
            .unwrap()
            .1,
            ExecutionLeaseOutcome::Lost
        );
    }
    #[test]
    fn rejects_cross_work_tokens_and_invalid_durations() {
        let (state, mut lease) = acquire(None, 1, 100);
        lease.work.turn_id = TurnId("other".into());
        assert_eq!(
            transition_execution_lease(
                &work(),
                state.as_ref(),
                &ExecutionLeaseCommand::Complete { lease },
                101
            )
            .unwrap()
            .1,
            ExecutionLeaseOutcome::Lost
        );
        for duration_ms in [0, 999, 300_001, u32::MAX] {
            assert!(transition_execution_lease(
                &work(),
                None,
                &ExecutionLeaseCommand::Acquire {
                    owner: ExecutionOwnerId([1; 16]),
                    duration_ms
                },
                100
            )
            .is_err());
        }
    }
    #[test]
    fn duplicate_acquisition_is_idempotent_and_expired_ownership_cannot_complete() {
        let (state, first) = acquire(None, 1, 100);
        assert_eq!(acquire(state.as_ref(), 1, 200).1, first);
        assert_eq!(
            transition_execution_lease(
                &work(),
                state.as_ref(),
                &ExecutionLeaseCommand::Complete { lease: first },
                1100
            )
            .unwrap()
            .1,
            ExecutionLeaseOutcome::Lost
        );
    }
}
