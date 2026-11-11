//! The authenticated driver partitions one terminal control envelope before
//! admitting its child. The linear grant cannot fund a second allocation.
use runmat_execution::resource::{ExactShared, REMOTE_DRIVER_COMPLETION_CONTROL_BYTES};

use super::pool_execution::RemotePoolExecutionResult;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) struct RemoteCompletionGrant {
    bytes: u64,
}

struct ClaimedCompletion {
    result: Option<RemotePoolExecutionResult>,
    // ExactShared deallocates its header before dropping this payload. Keep
    // the original grant after the result so cancellation drops its contents
    // before releasing the allocation's claim.
    _grant: RemoteCompletionGrant,
}

pub(super) struct OwnedRemoteCompletion(ExactShared<ClaimedCompletion>);

fn completion_capacity_bytes() -> NativeExecutionResult<u64> {
    let backing = u64::try_from(ExactShared::<ClaimedCompletion>::requested_bytes())
        .map_err(|_| invalid("remote completion allocation footprint overflows"))?;
    if backing > REMOTE_DRIVER_COMPLETION_CONTROL_BYTES {
        return Err(invalid(
            "remote completion backing exceeds its portable control component",
        ));
    }
    Ok(REMOTE_DRIVER_COMPLETION_CONTROL_BYTES)
}

impl RemoteCompletionGrant {
    pub(super) fn admit(memory_bytes: u64) -> NativeExecutionResult<(Self, u64)> {
        let bytes = completion_capacity_bytes()?;
        let remaining = memory_bytes.checked_sub(bytes).ok_or_else(|| {
            invalid("authenticated driver memory cannot fund its terminal completion envelope")
        })?;
        Ok((Self { bytes }, remaining))
    }

    pub(super) fn retain(
        self,
        result: RemotePoolExecutionResult,
    ) -> NativeExecutionResult<OwnedRemoteCompletion> {
        debug_assert!(ExactShared::<ClaimedCompletion>::requested_bytes() as u64 <= self.bytes);
        ExactShared::try_new(ClaimedCompletion {
            result: Some(result),
            _grant: self,
        })
        .map(OwnedRemoteCompletion)
        .map_err(|_| invalid("remote completion allocation failed"))
    }
}

impl OwnedRemoteCompletion {
    /// Publication moves the existing values out once. The original claim
    /// remains with this allocation throughout publication or cancellation.
    pub(super) fn take_result(&mut self) -> RemotePoolExecutionResult {
        self.0
            .get_mut_unique()
            .result
            .take()
            .expect("remote completion is published once")
    }
}

fn invalid(message: &str) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::resource::remote_driver_memory_bytes;
    use runmat_execution_artifact::ProgramExecutionResponse;

    #[test]
    fn terminal_grant_partitions_exact_storage_and_moves_original_response() {
        let bytes = REMOTE_DRIVER_COMPLETION_CONTROL_BYTES;
        assert!(ExactShared::<ClaimedCompletion>::requested_bytes() as u64 <= bytes);
        assert!(RemoteCompletionGrant::admit(bytes - 1).is_err());
        assert!(remote_driver_memory_bytes(u64::MAX).is_err());
        let parent_memory = remote_driver_memory_bytes(4096).unwrap();
        assert_eq!(parent_memory, bytes + 4096);
        let (grant, child_memory) = RemoteCompletionGrant::admit(parent_memory).unwrap();
        assert_eq!(child_memory, 4096);
        let message = String::from("original remote failure");
        let original = message.as_ptr();
        let mut retained = grant
            .retain(RemotePoolExecutionResult::without_objects(
                ProgramExecutionResponse::Failure { message },
            ))
            .unwrap();
        let result = retained.take_result();
        let ProgramExecutionResponse::Failure { message } = result.response else {
            panic!("terminal response changed");
        };
        assert_eq!(message.as_ptr(), original);
        assert_eq!(retained.0.get()._grant.bytes, bytes);
        assert!(retained.0.get().result.is_none());
        drop(retained);

        // An unpublished/cancelled terminal envelope owns and drops the same
        // original response; it needs no publication callback to release it.
        let (grant, _) = RemoteCompletionGrant::admit(bytes).unwrap();
        drop(
            grant
                .retain(RemotePoolExecutionResult::without_objects(
                    ProgramExecutionResponse::Failure { message },
                ))
                .unwrap(),
        );

        let (grant, _) = RemoteCompletionGrant::admit(bytes).unwrap();
        let response = String::from("allocation refusal retains the original payload");
        let original = response.as_ptr();
        // SAFETY: the injected allocator always returns null, transferring no
        // block. ExactShared must return the original payload and its grant.
        let refused = unsafe {
            ExactShared::try_new_with(
                ClaimedCompletion {
                    result: Some(RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::Failure { message: response },
                    )),
                    _grant: grant,
                },
                |_| std::ptr::null_mut(),
            )
        };
        let Err(refused) = refused else {
            panic!("injected allocation refusal was ignored");
        };
        assert_eq!(refused._grant.bytes, bytes);
        let ProgramExecutionResponse::Failure { message } =
            &refused.result.as_ref().unwrap().response
        else {
            panic!("allocation refusal changed its response");
        };
        assert_eq!(message.as_ptr(), original);
        drop(refused);
    }
}
