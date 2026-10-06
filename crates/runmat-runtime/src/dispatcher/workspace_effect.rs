use std::rc::Rc;

use crate::context::{RuntimeContext, RuntimeWorkspaceService};
use crate::RuntimeError;
use runmat_value::Value;

/// Retains the exact workspace port selected for one contextual call and
/// reconciles reentrant effects before execution resumes in its caller.
pub(super) struct PostCallWorkspaceEffectFence {
    workspace: Option<Rc<dyn RuntimeWorkspaceService>>,
}

impl PostCallWorkspaceEffectFence {
    pub(super) fn capture(context: Option<&RuntimeContext>) -> Self {
        Self {
            workspace: context.and_then(|context| context.service_ports().workspace().cloned()),
        }
    }

    pub(super) fn finish(self, result: Result<Value, RuntimeError>) -> Result<Value, RuntimeError> {
        self.finish_optional(result.map(Some))
            .map(|value| value.expect("scalar builtin result remains present"))
    }

    pub(super) fn finish_optional(
        self,
        result: Result<Option<Value>, RuntimeError>,
    ) -> Result<Option<Value>, RuntimeError> {
        if self.workspace.is_none() {
            return result;
        }
        match result {
            // No implementation was selected, so no builtin call completed
            // and this probe must remain observationally inert.
            Ok(None) => Ok(None),
            // A language or implementation failure remains the primary cause.
            // The checkpoint still runs so a catch handler observes effects
            // completed before that failure.
            Err(error) => {
                let _ = self.checkpoint();
                Err(error)
            }
            Ok(value) => {
                // The builtin has already released its argument/result roots,
                // while its caller cannot root the returned value until this
                // boundary completes. Reconciliation can retire workspace
                // values and collect, so retain the result across that work.
                let roots = value
                    .as_ref()
                    .map(|value| {
                        crate::execution::RootedValueHandles::for_values(std::iter::once(value))
                            .map_err(|source| {
                                crate::build_runtime_error(
                                    "could not retain a builtin result during workspace reconciliation",
                                )
                                .with_identifier("RunMat:Workspace:Rooting")
                                .with_source(source)
                                .build()
                            })
                    })
                    .transpose();
                let checkpoint = self.checkpoint();
                let _roots = match roots {
                    Ok(roots) => roots,
                    Err(error) => return Err(error),
                };
                checkpoint.map(|()| value)
            }
        }
    }

    fn checkpoint(&self) -> Result<(), RuntimeError> {
        self.workspace
            .as_ref()
            .map_or(Ok(()), |workspace| workspace.ensure_effect_authority())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{HandleRef, Value};

    fn runtime_with_workspace(workspace: Rc<dyn RuntimeWorkspaceService>) -> RuntimeContext {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let ports = runtime.service_ports().clone().with_workspace(workspace);
        runtime.with_service_ports(ports)
    }

    #[test]
    fn successful_call_reports_exact_retained_workspace_revocation() {
        let owner = crate::workspace::BaseWorkspaceOwner::new();
        let service: Rc<dyn RuntimeWorkspaceService> =
            Rc::new(crate::workspace::BaseWorkspaceService::new(owner.clone()));
        let runtime = runtime_with_workspace(service);
        let fence = PostCallWorkspaceEffectFence::capture(Some(&runtime));
        owner.close().unwrap();

        let error = fence.finish(Ok(Value::Nothing)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    }

    #[test]
    fn failed_call_keeps_its_original_cause_after_workspace_revocation() {
        let owner = crate::workspace::BaseWorkspaceOwner::new();
        let service: Rc<dyn RuntimeWorkspaceService> =
            Rc::new(crate::workspace::BaseWorkspaceService::new(owner.clone()));
        let runtime = runtime_with_workspace(service);
        let fence = PostCallWorkspaceEffectFence::capture(Some(&runtime));
        owner.close().unwrap();
        let original =
            crate::runtime_error::semantic_error("RunMat:PostCallFixture", "original call failure");

        let error = fence.finish(Err(original)).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:PostCallFixture"));
    }

    #[test]
    fn authority_free_call_has_no_workspace_side_effect() {
        let result = PostCallWorkspaceEffectFence::capture(None)
            .finish(Ok(Value::Num(7.0)))
            .unwrap();
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn optional_dispatch_miss_does_not_checkpoint_workspace() {
        let owner = crate::workspace::BaseWorkspaceOwner::new();
        let service: Rc<dyn RuntimeWorkspaceService> =
            Rc::new(crate::workspace::BaseWorkspaceService::new(owner.clone()));
        let runtime = runtime_with_workspace(service);
        owner.close().unwrap();

        let result = PostCallWorkspaceEffectFence::capture(Some(&runtime))
            .finish_optional(Ok(None))
            .unwrap();
        assert_eq!(result, None);
    }

    struct CollectingWorkspace {
        initial_root: std::cell::RefCell<Option<runmat_gc::ExplicitRoot>>,
        retained_during_checkpoint: std::cell::Cell<bool>,
        handle: runmat_gc::GcHandle,
    }

    impl RuntimeWorkspaceService for CollectingWorkspace {
        fn lookup(&self, _name: &str) -> Option<Value> {
            None
        }

        fn snapshot(&self) -> Vec<(String, Value)> {
            Vec::new()
        }

        fn global_names(&self) -> Vec<String> {
            Vec::new()
        }

        fn ensure_effect_authority(&self) -> Result<(), RuntimeError> {
            self.initial_root
                .borrow_mut()
                .take()
                .expect("one checkpoint")
                .unroot()
                .expect("release fixture root");
            runmat_gc::gc_collect_major().expect("collect during checkpoint");
            self.retained_during_checkpoint
                .set(runmat_gc::gc_clone_value(&self.handle).is_ok());
            Ok(())
        }

        fn assign(&self, _name: &str, _value: Value) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn clear(&self) -> Result<(), RuntimeError> {
            Ok(())
        }

        fn remove(&self, _name: &str) -> Result<(), RuntimeError> {
            Ok(())
        }
    }

    #[test]
    fn returned_managed_value_remains_rooted_during_workspace_reconciliation() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("retained".into()))
                .expect("allocate fixture value");
            let handle = initial.handle();
            let service = Rc::new(CollectingWorkspace {
                initial_root: std::cell::RefCell::new(Some(initial)),
                retained_during_checkpoint: std::cell::Cell::new(false),
                handle,
            });
            let runtime = runtime_with_workspace(service.clone());
            let result = Value::HandleObject(HandleRef {
                class_name: "PostCallFenceFixture".into(),
                target: handle,
                valid: true,
            });

            let returned = PostCallWorkspaceEffectFence::capture(Some(&runtime))
                .finish(Ok(result))
                .expect("checkpoint succeeds");
            assert!(service.retained_during_checkpoint.get());
            assert!(matches!(returned, Value::HandleObject(_)));
        });
    }
}
