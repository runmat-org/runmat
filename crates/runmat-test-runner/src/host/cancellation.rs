use std::future::{pending, Future};
use std::pin::Pin;

pub type PortFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

pub trait CancellationPort {
    fn is_cancelled(&self) -> bool;
    fn reason(&self) -> Option<String>;
    fn cancelled<'a>(&'a self) -> PortFuture<'a, String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NeverCancelled;

impl CancellationPort for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }

    fn reason(&self) -> Option<String> {
        None
    }

    fn cancelled<'a>(&'a self) -> PortFuture<'a, String> {
        Box::pin(pending())
    }
}
