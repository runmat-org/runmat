use crate::protocol::ProtocolLimits;

use super::TestCommand;

#[derive(Clone, Debug)]
pub struct CommandRecorder {
    limits: ProtocolLimits,
    commands: Vec<TestCommand>,
}

impl CommandRecorder {
    pub fn new(limits: ProtocolLimits) -> Self {
        Self {
            limits,
            commands: Vec::new(),
        }
    }

    pub fn record(&mut self, command: TestCommand) -> bool {
        if self.commands.len() >= self.limits.max_commands_per_invocation as usize {
            return false;
        }
        self.commands.push(command);
        true
    }

    pub fn into_commands(self) -> Vec<TestCommand> {
        self.commands
    }
}
