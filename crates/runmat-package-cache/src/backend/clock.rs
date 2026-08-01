pub trait CacheClock: Send + Sync {
    fn now_ms(&self) -> u64;
}

impl<T> CacheClock for std::sync::Arc<T>
where
    T: CacheClock + ?Sized,
{
    fn now_ms(&self) -> u64 {
        (**self).now_ms()
    }
}
