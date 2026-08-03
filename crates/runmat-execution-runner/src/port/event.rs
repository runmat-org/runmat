use super::PortFuture;
use crate::driver::DriverEvent;
use crate::RunnerResult;

pub trait EventPort {
    fn publish<'a>(&'a mut self, event: &'a DriverEvent) -> PortFuture<'a, RunnerResult<()>>;
}
