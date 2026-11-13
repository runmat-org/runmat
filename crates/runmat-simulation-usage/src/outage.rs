//! How long a lease client keeps a run alive while its server is unreachable.
//!
//! The server keeps an active lease for many renewal intervals after its last
//! usage report, so a client that misses a few renewals during a short server
//! restart finds the same lease when the server returns. Usage reports carry
//! absolute totals, so a resent report cannot count work twice.
use std::time::Duration;

/// Renewal intervals a client may miss before an unreachable server ends the run.
/// With the server's 30 s interval this is 5 minutes, which covers a rebuild and
/// restart of a development server.
pub const OUTAGE_RENEWALS: u32 = 10;

/// Longest one lease exchange may take when the server answers: a token refresh
/// and the request, each with its own 10 s transport timeout, plus their retry
/// after a rejected token. Proxies that forward lease calls to a parent wait at
/// least this long for one reply.
pub const LEASE_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(35);

/// The outage window for a lease renewed every `renewal_interval` that stays
/// valid for `until_expiry` more: `OUTAGE_RENEWALS` intervals, never past the
/// lease's own expiry.
pub fn outage_window(renewal_interval: Duration, until_expiry: Duration) -> Duration {
    renewal_interval
        .saturating_mul(OUTAGE_RENEWALS)
        .min(until_expiry)
}

/// Longest a forwarded renewal or settlement may wait for its parent's reply:
/// the parent's whole outage window, then one more exchange.
pub fn forwarded_reply_timeout(renewal_interval: Duration) -> Duration {
    renewal_interval
        .saturating_mul(OUTAGE_RENEWALS)
        .saturating_add(LEASE_EXCHANGE_TIMEOUT)
}

/// First and longest pause between attempts while the server is unreachable.
/// Pauses start at 1/30 of the renewal interval (1 s for 30 s) and double up to
/// one renewal interval.
pub fn outage_backoff(renewal_interval: Duration) -> (Duration, Duration) {
    let first = (renewal_interval / 30).max(Duration::from_millis(10));
    (first, renewal_interval.max(first))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_derives_from_the_renewal_interval_and_never_passes_expiry() {
        let thirty = Duration::from_secs(30);
        assert_eq!(
            outage_window(thirty, Duration::from_secs(86_400)),
            Duration::from_secs(300)
        );
        assert_eq!(
            outage_window(thirty, Duration::from_secs(90)),
            Duration::from_secs(90)
        );
        assert_eq!(
            outage_window(Duration::from_secs(1), Duration::MAX),
            Duration::from_secs(10)
        );
        assert_eq!(forwarded_reply_timeout(thirty), Duration::from_secs(335));
        assert_eq!(
            outage_backoff(thirty),
            (Duration::from_secs(1), Duration::from_secs(30))
        );
    }
}
