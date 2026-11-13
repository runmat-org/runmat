//! One retry policy for lease renewals and settlements while RunMat Server is briefly
//! unreachable, such as during a restart. Only transport failures and 5xx answers are
//! retried; denials, sign-in, license and allowance answers end the request at once.
//! Pauses back off exponentially with jitter inside a window derived from the lease's
//! renewal interval (`runmat_simulation_usage::outage`). When the window is spent the
//! caller returns the original error unchanged.
use chrono::{DateTime, Utc};
use rand::Rng;
use runmat_simulation_usage::outage;
use std::time::{Duration, Instant};

use crate::{simulation::SimulationLeaseError, ServerLeaseTransportError};

impl ServerLeaseTransportError {
    /// True when the server could not be reached or answered 5xx: connection refused or
    /// reset, request timeout, an interrupted body, or HTTP 500–599.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Http(error) => {
                !(error.is_builder()
                    || error.is_redirect()
                    || error.is_status()
                    || error.is_decode())
            }
            Self::ServerStatus(status) => (500..=599).contains(status),
            Self::Endpoint(_) | Self::Credential(_) | Self::BodyTooLarge | Self::Decode(_) => false,
        }
    }
}

impl SimulationLeaseError {
    /// True only for transport failures and 5xx answers; every typed server answer is final.
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Transport(error) if error.is_transient())
    }
}

/// The outage window and backoff for one lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseRetry {
    window: Duration,
    first_pause: Duration,
    longest_pause: Duration,
}

impl LeaseRetry {
    /// Policy for a lease renewed every `renewal_interval` and valid until `expires_at`.
    pub fn for_lease(renewal_interval: Duration, expires_at: DateTime<Utc>) -> Self {
        let until_expiry = (expires_at - Utc::now()).to_std().unwrap_or(Duration::ZERO);
        let (first_pause, longest_pause) = outage::outage_backoff(renewal_interval);
        Self {
            window: outage::outage_window(renewal_interval, until_expiry),
            first_pause,
            longest_pause,
        }
    }

    pub fn window(&self) -> Duration {
        self.window
    }

    /// Start timing one request's outage. `operation` names the request in logs.
    pub fn begin(&self, operation: &'static str) -> OutageClock {
        OutageClock {
            policy: *self,
            operation,
            started: Instant::now(),
            attempts: 1,
            pause: self.first_pause,
        }
    }
}

/// One request's retries. Call [`OutageClock::retry`] after each transient failure.
#[derive(Debug)]
pub struct OutageClock {
    policy: LeaseRetry,
    operation: &'static str,
    started: Instant,
    attempts: u32,
    pause: Duration,
}

impl OutageClock {
    /// Time left for the complete request, including transport and token refresh.
    pub fn remaining(&self) -> Duration {
        self.policy.window.saturating_sub(self.started.elapsed())
    }

    /// The next pause, or `None` when sleeping it would pass the window. Equal jitter:
    /// half the current backoff plus a random part of the other half.
    pub fn next_pause(&mut self) -> Option<Duration> {
        let half = self.pause / 2;
        let jitter = rand::thread_rng().gen_range(Duration::ZERO..=half);
        let pause = half + jitter;
        if self.started.elapsed().saturating_add(pause) >= self.policy.window {
            return None;
        }
        self.pause = self.pause.saturating_mul(2).min(self.policy.longest_pause);
        Some(pause)
    }

    /// Sleep before the next attempt and return true, or return false when the window
    /// is spent. Each retry is logged at info level with its attempt and elapsed time.
    pub async fn retry(&mut self, error: &dyn std::fmt::Display) -> bool {
        let elapsed = self.started.elapsed();
        let Some(pause) = self.next_pause() else {
            tracing::warn!(
                operation = self.operation,
                attempts = self.attempts,
                elapsed_ms = elapsed.as_millis() as u64,
                window_ms = self.policy.window.as_millis() as u64,
                %error,
                "RunMat Server stayed unreachable for the lease outage window"
            );
            return false;
        };
        tracing::info!(
            operation = self.operation,
            attempt = self.attempts,
            elapsed_ms = elapsed.as_millis() as u64,
            retry_in_ms = pause.as_millis() as u64,
            window_ms = self.policy.window.as_millis() as u64,
            %error,
            "RunMat Server unreachable; retrying lease request"
        );
        tokio::time::sleep(pause).await;
        self.attempts += 1;
        true
    }

    /// Log the recovery when an earlier attempt failed.
    pub fn recovered(&self) {
        if self.attempts > 1 {
            tracing::info!(
                operation = self.operation,
                attempt = self.attempts,
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                "RunMat Server reachable again; lease request succeeded"
            );
        }
    }

    pub fn attempts(&self) -> u32 {
        self.attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_transport_failures_and_5xx_are_transient() {
        assert!(ServerLeaseTransportError::ServerStatus(502).is_transient());
        assert!(ServerLeaseTransportError::ServerStatus(503).is_transient());
        assert!(ServerLeaseTransportError::ServerStatus(500).is_transient());
        assert!(!ServerLeaseTransportError::BodyTooLarge.is_transient());
        let decode = serde_json::from_slice::<u8>(b"x").unwrap_err();
        assert!(!ServerLeaseTransportError::Decode(decode).is_transient());
        let rejected = SimulationLeaseError::Rejected {
            status: 409,
            code: crate::simulation::SimulationErrorCode::AllowanceUsed,
            message: None,
            details: Default::default(),
        };
        assert!(!rejected.is_transient());
        assert!(!SimulationLeaseError::InvalidResponse("x").is_transient());
        assert!(
            SimulationLeaseError::Transport(ServerLeaseTransportError::ServerStatus(504))
                .is_transient()
        );
    }

    #[tokio::test]
    async fn refused_connections_are_transient() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let error = reqwest::Client::new()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap_err();
        assert!(ServerLeaseTransportError::Http(error).is_transient());
    }

    #[test]
    fn pauses_double_with_jitter_and_stop_at_the_window() {
        let policy = LeaseRetry::for_lease(
            Duration::from_secs(30),
            Utc::now() + chrono::Duration::hours(24),
        );
        assert_eq!(policy.window(), Duration::from_secs(300));
        let mut clock = policy.begin("test");
        assert!(clock.remaining() <= policy.window());
        let mut total = Duration::ZERO;
        let mut step = Duration::from_secs(1);
        while let Some(pause) = clock.next_pause() {
            // Equal jitter keeps each pause within [step/2, step] of its doubling step.
            assert!(pause >= step / 2 && pause <= step, "{pause:?} for {step:?}");
            total += pause;
            step = (step * 2).min(Duration::from_secs(30));
            // Stand in for the sleep by moving the clock's start back.
            clock.started -= pause;
        }
        assert!(total < Duration::from_secs(300));
        assert!(total >= Duration::from_secs(270), "{total:?}");
        // An expired lease has no window.
        let expired = LeaseRetry::for_lease(Duration::from_secs(30), Utc::now());
        assert_eq!(expired.begin("expired").remaining(), Duration::ZERO);
        assert_eq!(expired.begin("expired").next_pause(), None);
    }
}
