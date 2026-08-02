use super::PortFuture;

pub trait Clock {
    fn now_ms(&self) -> u64;
    fn sleep_until<'a>(&'a self, deadline_ms: u64) -> PortFuture<'a, ()>;
}
