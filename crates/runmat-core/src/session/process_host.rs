//! Core's process capability uses the shared host driver without detaching the
//! child future from its invoking request.

use std::rc::Rc;

use runmat_process_host::{CompleteProcessOutput, HostCommand, ProcessHostError};
use runmat_runtime::context::{RuntimeProcessService, RuntimeServiceFuture};

use super::host_driver::{host_driver, EnteredHostFuture};

#[derive(Default)]
struct NativeProcessService;

impl RuntimeProcessService for NativeProcessService {
    fn output_complete(
        &self,
        command: HostCommand,
    ) -> RuntimeServiceFuture<Result<CompleteProcessOutput, ProcessHostError>> {
        Box::pin(async move {
            let driver = host_driver()
                .map_err(|source| ProcessHostError::Io(std::io::Error::other(source)))?;
            EnteredHostFuture::new(driver.handle().clone(), command.output_complete()).await
        })
    }

    fn output_complete_with_input(
        &self,
        command: HostCommand,
        input: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<CompleteProcessOutput, ProcessHostError>> {
        Box::pin(async move {
            let driver = host_driver()
                .map_err(|source| ProcessHostError::Io(std::io::Error::other(source)))?;
            EnteredHostFuture::new(
                driver.handle().clone(),
                command.output_complete_with_input(input),
            )
            .await
        })
    }
}

pub(super) fn default_process_service() -> Rc<dyn RuntimeProcessService> {
    Rc::new(NativeProcessService)
}

#[cfg(test)]
#[path = "process_host/tests.rs"]
mod tests;
