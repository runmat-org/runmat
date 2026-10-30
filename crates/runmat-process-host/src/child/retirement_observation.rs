pub(super) enum RetirementObservation {
    Retired,
    Pending,
    // Only an Apple group-absence probe in the post-signaling retirement wait.
    // This is uncertainty, never evidence that the group is gone.
    #[cfg(any(
        target_vendor = "apple",
        all(test, any(target_os = "linux", target_os = "android"))
    ))]
    GroupProbeDenied(std::io::Error),
}

impl RetirementObservation {
    pub(super) fn from_retired(retired: bool) -> Self {
        if retired {
            Self::Retired
        } else {
            Self::Pending
        }
    }
}

// Shared by async and exceptional-Drop waits. No signal authority is present
// here: a retry can only observe, and success still requires positive absence.
pub(super) fn retirement_observed(
    observation: RetirementObservation,
    expired: bool,
    first_probe_error: &mut Option<std::io::Error>,
) -> std::io::Result<bool> {
    match observation {
        RetirementObservation::Retired => return Ok(true),
        RetirementObservation::Pending => {}
        #[cfg(any(
            target_vendor = "apple",
            all(test, any(target_os = "linux", target_os = "android"))
        ))]
        RetirementObservation::GroupProbeDenied(error) => {
            first_probe_error.get_or_insert(error);
        }
    }
    if expired {
        Err(first_probe_error.take().unwrap_or_else(retirement_timeout))
    } else {
        Ok(false)
    }
}

fn retirement_timeout() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "owned process tree did not retire within the deadline",
    )
}

#[cfg(any(
    target_vendor = "apple",
    all(test, any(target_os = "linux", target_os = "android"))
))]
pub(super) fn group_probe_error(error: std::io::Error) -> std::io::Result<RetirementObservation> {
    if error.raw_os_error() == Some(libc::EPERM) {
        Ok(RetirementObservation::GroupProbeDenied(error))
    } else {
        Err(error)
    }
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
mod tests {
    #[test]
    fn group_probe_observation_matrix_preserves_absence_and_first_error() {
        use super::{group_probe_error, retirement_observed, RetirementObservation};
        use std::io::Error;

        for code in [libc::EIO, libc::EACCES] {
            let error = group_probe_error(Error::from_raw_os_error(code))
                .err()
                .expect("only a group-probe EPERM may be retried");
            assert_eq!(error.raw_os_error(), Some(code));
        }
        // The only retryable branch remains pending, including at a deadline;
        // an expired wait cannot manufacture an absence certificate. A later
        // confirmed absence is the sole success path after an ambiguous probe.
        for (leader_reaped, group_absent, expired) in [
            (false, false, false),
            (false, true, false),
            (true, false, false),
            (true, true, false),
            (false, false, true),
            (true, false, true),
        ] {
            let mut first = None;
            let denied = group_probe_error(Error::from_raw_os_error(libc::EPERM))
                .expect("typed probe denial");
            assert!(!retirement_observed(denied, false, &mut first).unwrap());
            assert_eq!(first.as_ref().unwrap().raw_os_error(), Some(libc::EPERM));
            // Use a distinguishable retained error to exercise first-error
            // ownership independently of the syscall classifier above.
            first = Some(Error::new(
                std::io::ErrorKind::PermissionDenied,
                "original probe",
            ));
            let denied = group_probe_error(Error::from_raw_os_error(libc::EPERM))
                .expect("repeated probe denial");
            assert!(!retirement_observed(denied, false, &mut first).unwrap());
            assert_eq!(first.as_ref().unwrap().to_string(), "original probe");
            let absence = leader_reaped && group_absent;
            let observation = RetirementObservation::from_retired(absence);
            let result = retirement_observed(observation, expired, &mut first);
            if expired {
                assert_eq!(result.unwrap_err().to_string(), "original probe");
            } else {
                assert_eq!(result.unwrap(), absence);
            }
        }
        let denied = group_probe_error(Error::from_raw_os_error(libc::EPERM)).unwrap();
        let error = retirement_observed(denied, true, &mut None).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EPERM));
    }
}
