//! Content-free native admission identities plus bounded compiler failures.
//! No workspace values, source text, or unredacted function/dependency names.
use super::{CompilationKey, Digest};
use runmat_jit::invalidation::{DependencyKey, DependencySnapshot};
use runmat_jit::tiering::{TierAvailability, TierDecision, TierSiteId, TieringSession};

const TARGET: &str = "runmat_core::native_tiering";

fn entry_id(key: &CompilationKey) -> Digest {
    Digest::sha256(key.entry.0.as_bytes())
}

pub(super) fn decision(
    site: &TierSiteId,
    decision: &TierDecision,
    availability: &TierAvailability,
    profiled: bool,
    tiering: &TieringSession,
) {
    if !log::log_enabled!(target: TARGET, log::Level::Debug) {
        return;
    }
    let feedback = tiering
        .snapshot()
        .sites
        .into_iter()
        .find(|entry| entry.site == *site);
    log::debug!(target: TARGET,
        "native decision entry={} function={} decision={decision:?} profiled={profiled} invocations={} backedges={} availability={availability:?}",
        Digest::sha256(site.entry.as_bytes()), site.function.0,
        feedback.as_ref().map_or(0, |f| f.invocations),
        feedback.as_ref().map_or(0, |f| f.backedges));
}

fn dependency_id(key: &DependencyKey) -> (&'static str, Digest) {
    match key {
        DependencyKey::Program(name) => ("program", Digest::sha256(name.as_bytes())),
        DependencyKey::Catalog(name) => ("catalog", Digest::sha256(name.as_bytes())),
        DependencyKey::SessionFunction(name) => {
            ("session_function", Digest::sha256(name.as_bytes()))
        }
        DependencyKey::SessionCatalog => ("session_catalog", Digest::sha256(b"session_catalog")),
        DependencyKey::Provider(id) => ("provider", Digest::sha256(id.to_le_bytes())),
    }
}

pub(super) fn dependency_changed(key: &DependencyKey) {
    if !log::log_enabled!(target: TARGET, log::Level::Debug) {
        return;
    }
    let (kind, id) = dependency_id(key);
    log::debug!(target: TARGET, "native dependency changed kind={kind} id={id}");
}

pub(super) fn scheduled(key: &CompilationKey, dependencies: &DependencySnapshot) {
    log::debug!(target: TARGET, "native compile scheduled entry={} tier={:?} dependencies={}",
        entry_id(key), key.tier, dependencies.iter().len());
}

pub(super) fn completed<T>(key: &CompilationKey, result: Option<Result<T, String>>) -> Option<T> {
    match result {
        Some(Ok(product)) => Some(product),
        Some(Err(error)) => {
            log::warn!(target: TARGET, "native background compilation failed entry={} tier={:?} error={}",
                entry_id(key), key.tier, bounded_error(&error));
            None
        }
        None => {
            log::warn!(target: TARGET, "native background compilation disconnected entry={} tier={:?}",
                entry_id(key), key.tier);
            None
        }
    }
}

fn bounded_error(error: &str) -> String {
    // Preserve the compiler's actual error, not just a generic fallback label.
    // Bound a single diagnostic and prevent multiline log injection.
    error
        .chars()
        .take(2048)
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

pub(super) fn stale(
    key: &CompilationKey,
    expected: &DependencySnapshot,
    current: &DependencySnapshot,
) {
    if !log::log_enabled!(target: TARGET, log::Level::Debug) {
        return;
    }
    let changed = expected
        .iter()
        .filter(|(dependency, generation)| current.generation(dependency) != Some(*generation))
        .collect::<Vec<_>>();
    log::debug!(target: TARGET, "native publication rejected entry={} tier={:?} reason=stale_dependencies changed={}", entry_id(key), key.tier, changed.len());
    for (dependency, generation) in changed.into_iter().take(16) {
        let (kind, id) = dependency_id(dependency);
        log::debug!(target: TARGET, "native stale dependency kind={kind} id={id} expected={} current={:?}",
            generation.get(), current.generation(dependency).map(|g| g.get()));
    }
}

pub(super) fn rejected(key: &CompilationKey, reason: &str, bytes: Option<u64>) {
    log::debug!(target: TARGET, "native publication rejected entry={} tier={:?} reason={reason} product_bytes={bytes:?}", entry_id(key), key.tier);
}

pub(super) fn publishing(key: &CompilationKey, bytes: u64) {
    log::debug!(target: TARGET, "native publication attempt entry={} tier={:?} product_bytes={bytes}", entry_id(key), key.tier);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_diagnostic_is_bounded_and_single_line() {
        assert_eq!(
            bounded_error("failed\nreason\r\tmore"),
            "failed reason  more"
        );
        assert_eq!(bounded_error(&"é".repeat(3000)).chars().count(), 2048);
    }
    #[test]
    fn identities_expose_no_authored_names() {
        let (kind, digest) = dependency_id(&DependencyKey::SessionFunction(
            "private_customer_function".into(),
        ));
        assert_eq!(kind, "session_function");
        assert!(!digest.to_string().contains("private_customer"));
    }
}
