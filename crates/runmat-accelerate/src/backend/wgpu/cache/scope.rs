//! Cache publication policy belongs to the backend, not the device gate.
use runmat_accelerate_api::WgpuEmissionMode;
use std::{error::Error, fmt, sync::OnceLock};

#[derive(Debug)]
pub(crate) struct UnvalidatedScopedCacheMiss;
impl fmt::Display for UnvalidatedScopedCacheMiss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("scoped cache miss requires checked construction and publication after scope resolution")
    }
}
impl Error for UnvalidatedScopedCacheMiss {}

pub(crate) fn allow_creation(mode: WgpuEmissionMode) -> Result<(), UnvalidatedScopedCacheMiss> {
    match mode {
        WgpuEmissionMode::Shared => Ok(()),
        WgpuEmissionMode::Scoped => Err(UnvalidatedScopedCacheMiss),
    }
}

pub(crate) fn get_or_init<T>(
    cache: &OnceLock<T>,
    mode: WgpuEmissionMode,
    create: impl FnOnce() -> T,
) -> Result<&T, UnvalidatedScopedCacheMiss> {
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    allow_creation(mode)?;
    Ok(cache.get_or_init(create))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn scoped_miss_never_creates_or_publishes_but_existing_hit_is_reused() {
        let cache = OnceLock::new();
        let builds = AtomicUsize::new(0);
        assert!(get_or_init(&cache, WgpuEmissionMode::Scoped, || {
            builds.fetch_add(1, Ordering::SeqCst);
            41
        })
        .is_err());
        assert!(cache.get().is_none());
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        let value = get_or_init(&cache, WgpuEmissionMode::Shared, || {
            builds.fetch_add(1, Ordering::SeqCst);
            42
        })
        .unwrap();
        let hit = get_or_init(&cache, WgpuEmissionMode::Scoped, || {
            panic!("hit must not build")
        })
        .unwrap();
        assert!(std::ptr::eq(value, hit));
        assert_eq!(*hit, 42);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }
}
