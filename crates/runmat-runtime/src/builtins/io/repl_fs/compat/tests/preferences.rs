use super::*;
use crate::builtins::io::repl_fs::preferences::{getpref_builtin, ispref_builtin, setpref_builtin};
use crate::context::{IsolatedMemoryPreferenceService, RuntimeContext, RuntimeServicePorts};

fn preference_context() -> RuntimeContext {
    RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_preferences(std::rc::Rc::new(IsolatedMemoryPreferenceService::new())),
    )
}

#[path = "preferences/contracts.rs"]
mod contracts;
#[path = "preferences/residency.rs"]
mod residency;
#[path = "preferences/values.rs"]
mod values;
