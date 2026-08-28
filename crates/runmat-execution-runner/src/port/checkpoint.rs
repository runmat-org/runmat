use super::PortFuture;
use crate::driver::DriverSnapshot;
use crate::RunnerResult;

pub trait CheckpointPort {
    fn save<'a>(&'a mut self, snapshot: &'a DriverSnapshot) -> PortFuture<'a, RunnerResult<()>>;
}
