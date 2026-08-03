use std::ffi::{OsStr, OsString};

use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HiddenMode {
    TestWorker,
    ExecutionWorker,
    ExecutionDriver,
    LocalSupervisor,
}

impl HiddenMode {
    pub const fn marker(self) -> &'static str {
        match self {
            Self::TestWorker => "--__runmat-test-worker",
            Self::ExecutionWorker => "--__runmat-execution-worker",
            Self::ExecutionDriver => "--__runmat-execution-driver",
            Self::LocalSupervisor => "--__runmat-local-supervisor",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HiddenModeRegistry;

impl HiddenModeRegistry {
    pub const fn standard() -> Self {
        Self
    }

    pub fn detect(
        &self,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> ProcessHostResult<Option<HiddenMode>> {
        let mut arguments = arguments.into_iter();
        let _executable = arguments.next();
        let remaining = arguments.collect::<Vec<_>>();
        let matches = remaining
            .iter()
            .filter_map(|argument| self.mode_for_marker(argument))
            .collect::<Vec<_>>();
        if matches.is_empty() {
            return Ok(None);
        }
        if remaining.len() != 1 || matches.len() != 1 {
            return Err(ProcessHostError::Configuration(
                "a private RunMat host mode must be the sole process argument".into(),
            ));
        }
        Ok(matches.into_iter().next())
    }

    fn mode_for_marker(&self, argument: &OsStr) -> Option<HiddenMode> {
        [
            HiddenMode::TestWorker,
            HiddenMode::ExecutionWorker,
            HiddenMode::ExecutionDriver,
            HiddenMode::LocalSupervisor,
        ]
        .into_iter()
        .find(|mode| argument == OsStr::new(mode.marker()))
    }
}
