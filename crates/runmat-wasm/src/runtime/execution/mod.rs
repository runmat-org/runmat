mod admission;
pub(crate) mod selected;
mod host;
mod model;
pub(crate) mod program;
mod request;
mod resources;
mod service;
mod state;
mod task_turn;

pub(crate) use admission::{execute_local_program, execute_local_worker_program};
pub(crate) use host::execution_host_from_options;
pub(crate) use request::BrowserWorkerRequest;
pub(crate) use service::BrowserExecutionService;

pub(crate) fn browser_analysis_host_inventory()->Result<runmat_execution::host::ExecutionHostInventory,String> {
    resources::browser_execution_host_inventory(&[]).map_err(|e|e.to_string())
}
