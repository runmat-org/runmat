mod attempt;
mod discovery;
mod executor;
mod procedure;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use procedure::function_name as test_procedure_function_name;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use procedure::method_name as test_procedure_method_name;
mod run;
pub(crate) mod runtime_adapter;
mod source_catalog;
mod snapshot_environment;
mod value;

pub use attempt::CoreTestAttempt;
pub use executor::CoreTestExecutor;
pub use run::CoreTestRun;
