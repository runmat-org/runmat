use std::collections::VecDeque;

use crate::RunnerResult;

use super::{Driver, DriverAction, DriverCommand};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActorStep {
    pub command: DriverCommand,
    pub actions: Vec<DriverAction>,
}

pub struct DriverActor {
    driver: Driver,
    mailbox: VecDeque<DriverCommand>,
}

impl DriverActor {
    pub fn new(driver: Driver) -> Self {
        Self {
            driver,
            mailbox: VecDeque::new(),
        }
    }

    pub fn enqueue(&mut self, command: DriverCommand) {
        self.mailbox.push_back(command);
    }

    pub fn run_next(&mut self) -> RunnerResult<Option<ActorStep>> {
        let Some(command) = self.mailbox.pop_front() else {
            return Ok(None);
        };
        let actions = self.driver.handle(command.clone())?;
        Ok(Some(ActorStep { command, actions }))
    }

    pub fn run_until_action_or_idle(&mut self) -> RunnerResult<Vec<ActorStep>> {
        let mut steps = Vec::new();
        while let Some(step) = self.run_next()? {
            let has_actions = !step.actions.is_empty();
            steps.push(step);
            if has_actions {
                break;
            }
        }
        Ok(steps)
    }

    pub fn driver(&self) -> &Driver {
        &self.driver
    }

    pub fn driver_mut(&mut self) -> &mut Driver {
        &mut self.driver
    }
}
