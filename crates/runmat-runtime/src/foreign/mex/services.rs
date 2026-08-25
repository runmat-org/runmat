use runmat_mex::{MexDiagnostic, MexHostServices};
use runmat_value::Value;

use crate::context::RuntimeContext;

/// Runtime-owned callback authority for one active MEX invocation.
///
/// The exact invocation context is retained so native execution sees its
/// scoped workspace service instead of falling back to a process-global
/// interpreter workspace.
pub struct RuntimeMexHostServices {
    runtime: RuntimeContext,
}

impl RuntimeMexHostServices {
    pub fn new(runtime: RuntimeContext) -> Self {
        Self { runtime }
    }
}

impl MexHostServices for RuntimeMexHostServices {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        pollster::block_on(self.runtime.scope(crate::call_builtin_async_with_outputs(
            "eval",
            &[Value::String(command.to_string())],
            0,
        )))
        .map(|_| ())
        .map_err(runtime_diagnostic)
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        let value = pollster::block_on(self.runtime.scope(crate::call_feval_async_with_outputs(
            Value::from(function),
            &arguments,
            requested_outputs,
        )))
        .map_err(runtime_diagnostic)?;
        match (requested_outputs, value) {
            (0, _) => Ok(Vec::new()),
            (_, Value::OutputList(values)) => Ok(values),
            (1, value) => Ok(vec![value]),
            (count, _) => Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:CallbackOutputs".into()),
                message: format!("callback did not produce the requested {count} outputs"),
            }),
        }
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        validate_workspace(workspace)?;
        if workspace == "global" {
            let _scope = self.runtime.enter();
            return Ok(crate::workspace::session::global_value(name));
        }
        if let Some(service) = self.runtime.service_ports().workspace() {
            Ok(service.lookup(name))
        } else {
            let _scope = self.runtime.enter();
            Ok(crate::workspace::lookup(name))
        }
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        validate_workspace(workspace)?;
        if workspace == "global" {
            let _scope = self.runtime.enter();
            crate::workspace::session::store_global_named(name, value);
            return Ok(());
        }
        if let Some(service) = self.runtime.service_ports().workspace() {
            service.assign(name, value).map_err(runtime_diagnostic)
        } else {
            let _scope = self.runtime.enter();
            crate::workspace::assign(name, value).map_err(|message| MexDiagnostic {
                identifier: Some("RunMat:MEX:Workspace".into()),
                message,
            })
        }
    }
}

fn validate_workspace(workspace: &str) -> Result<(), MexDiagnostic> {
    if matches!(workspace, "base" | "caller" | "global") {
        Ok(())
    } else {
        Err(MexDiagnostic {
            identifier: Some("RunMat:MEX:Workspace".into()),
            message: format!("unknown MEX workspace '{workspace}'"),
        })
    }
}

fn runtime_diagnostic(error: crate::RuntimeError) -> MexDiagnostic {
    MexDiagnostic {
        identifier: error.identifier().map(str::to_string),
        message: error.message().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeServicePorts, RuntimeWorkspaceService};
    use crate::execution::RuntimeExecutionService;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    #[derive(Default)]
    struct ScopedWorkspace {
        values: RefCell<BTreeMap<String, Value>>,
    }

    impl RuntimeWorkspaceService for ScopedWorkspace {
        fn lookup(&self, name: &str) -> Option<Value> {
            self.values.borrow().get(name).cloned()
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            self.values
                .borrow()
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        }

        fn global_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn assign(&self, name: &str, value: Value) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().insert(name.to_string(), value);
            Ok(())
        }

        fn clear(&self) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().clear();
            Ok(())
        }

        fn remove(&self, name: &str) -> Result<(), crate::RuntimeError> {
            self.values.borrow_mut().remove(name);
            Ok(())
        }
    }

    #[test]
    fn workspace_callbacks_use_the_exact_invocation_service() {
        let workspace = Rc::new(ScopedWorkspace::default());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_workspace(workspace.clone()));
        let services = RuntimeMexHostServices::new(runtime);

        services
            .put_variable("caller", "answer", Value::Num(42.0))
            .expect("write scoped caller workspace");
        assert_eq!(
            services
                .get_variable("base", "answer")
                .expect("read scoped base workspace"),
            Some(Value::Num(42.0))
        );
        assert_eq!(workspace.lookup("answer"), Some(Value::Num(42.0)));
    }

    #[test]
    fn workspace_callbacks_reject_unknown_workspace_names() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let services = RuntimeMexHostServices::new(runtime);
        let error = services
            .get_variable("workspace-typo", "answer")
            .expect_err("unknown workspace must fail");

        assert_eq!(error.identifier.as_deref(), Some("RunMat:MEX:Workspace"));
    }
}
