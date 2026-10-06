use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use runmat_runtime::context::{
    RuntimeContext, RuntimeServiceFuture, RuntimeSourceEvaluationAuthority,
    RuntimeSourceEvaluationService, SessionSourceEvaluation,
};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::ValueSequence;

#[path = "source_evaluation/generation.rs"]
mod generation;
use generation::{
    CallableArtifactIndex, CallableArtifactSelectionError, FunctionRegistryGeneration,
    GenerationRetirement, PreparedGenerationImport,
};
#[path = "source_evaluation/callable_artifacts.rs"]
mod callable_artifacts;

pub(super) use callable_artifacts::{install_bound_invoker, service as callable_artifact_service};

pub(super) struct SessionSourceEvaluator {
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    runtime_owner: runmat_runtime::context::RuntimeContextLocalId,
    generation: RefCell<Rc<FunctionRegistryGeneration>>,
}

pub(super) struct FunctionRegistryPublication {
    evaluator: Rc<SessionSourceEvaluator>,
    active: Rc<FunctionRegistryGeneration>,
    restore: Option<Rc<FunctionRegistryGeneration>>,
}

impl FunctionRegistryPublication {
    pub(super) fn finish(mut self, committed: runmat_vm::FunctionRegistry) {
        self.active.commit(committed);
        self.restore = None;
    }
}

impl Drop for FunctionRegistryPublication {
    fn drop(&mut self) {
        if let Some(restore) = self.restore.take() {
            self.active.retain_escaped_into(&restore);
            self.active.invalidate();
            self.evaluator.restore_generation(&self.active, restore);
        } else {
            self.active.invalidate();
        }
    }
}

#[derive(Debug)]
struct CapturedSourceEvaluationAuthority {
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    generation: Rc<FunctionRegistryGeneration>,
}

impl RuntimeSourceEvaluationAuthority for CapturedSourceEvaluationAuthority {
    fn evaluate(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<ValueSequence, RuntimeError>> {
        let registry = match self.generation.registry() {
            Ok(registry) => registry,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let base_service = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            self.base_workspace.clone(),
        ));
        Box::pin(async move {
            let source_ref = request.runtime_source_ref().ok_or_else(|| {
                build_runtime_error("source evaluation lacks authenticated runtime provenance")
                    .with_identifier("RunMat:SourceEvaluation:UnauthenticatedSource")
                    .build()
            })?;
            if source_ref.resolve_in(&runtime).is_none() {
                return Err(build_runtime_error(
                    "source evaluation provenance does not belong to this runtime session",
                )
                .with_identifier("RunMat:SourceEvaluation:SourceMismatch")
                .build());
            }
            let policy = request.policy();
            let source = request.source().to_string();
            let source_label = request.source_label().to_string();
            let requested_outputs = request.requested_outputs();
            let runtime = runtime.with_language_policy(policy);
            let services = runtime
                .service_ports()
                .clone()
                .with_base_workspace(base_service);
            let runtime = runtime.with_service_ports(services);
            runtime
                .clone()
                .scope(runmat_vm::evaluate_session_source_in_context(
                    runtime,
                    source,
                    source_label,
                    source_ref,
                    requested_outputs,
                    &registry,
                ))
                .await
        })
    }
}

struct BoundSessionSourceEvaluator {
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    generation: Rc<FunctionRegistryGeneration>,
}

impl SessionSourceEvaluator {
    pub(super) fn new(
        base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
        runtime_owner: runmat_runtime::context::RuntimeContextLocalId,
    ) -> Self {
        Self {
            base_workspace,
            runtime_owner,
            generation: RefCell::new(Rc::new(FunctionRegistryGeneration::committed(
                runmat_vm::FunctionRegistry::default(),
            ))),
        }
    }

    pub(super) fn current_generation(&self) -> Rc<FunctionRegistryGeneration> {
        self.generation.borrow().clone()
    }

    pub(super) fn ensure_runtime_owner(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        if runtime.local_identity() != self.runtime_owner {
            return Err(build_runtime_error(
                "callable artifact operation belongs to a different runtime session",
            )
            .with_identifier("RunMat:CallableArtifact:ForeignSession")
            .build());
        }
        Ok(())
    }

    pub(super) fn prepare_import(
        &self,
        registry: runmat_vm::FunctionRegistry,
        artifacts: CallableArtifactIndex,
        source_owners: Vec<Rc<runmat_vm::RuntimeBoundExecutableUnit>>,
    ) -> PreparedGenerationImport {
        PreparedGenerationImport::new(registry, artifacts, source_owners)
    }

    pub(super) fn commit_import(
        &self,
        expected: &Rc<FunctionRegistryGeneration>,
        expected_epoch: &Rc<()>,
        prepared: PreparedGenerationImport,
    ) -> Result<GenerationRetirement, RuntimeError> {
        let current = self.generation.borrow();
        if !Rc::ptr_eq(&current, expected) {
            return Err(build_runtime_error(
                "callable artifact destination changed during admission",
            )
            .with_identifier("RunMat:CallableArtifact:DestinationChanged")
            .build());
        }
        expected.commit_import(expected_epoch, prepared)
    }

    fn replace_generation(
        &self,
        generation: Rc<FunctionRegistryGeneration>,
    ) -> Rc<FunctionRegistryGeneration> {
        let retired = {
            let mut current = self.generation.borrow_mut();
            std::mem::replace(&mut *current, generation)
        };
        retired
    }

    fn restore_generation(
        &self,
        active: &Rc<FunctionRegistryGeneration>,
        restore: Rc<FunctionRegistryGeneration>,
    ) {
        let retired = {
            let mut current = self.generation.borrow_mut();
            if !Rc::ptr_eq(&current, active) {
                return;
            }
            std::mem::replace(&mut *current, restore)
        };
        drop(retired);
    }

    pub(super) fn publish_for_execution(
        self: &Rc<Self>,
        active: runmat_vm::FunctionRegistry,
        unit_bytes: Option<Vec<u8>>,
    ) -> Result<FunctionRegistryPublication, RuntimeError> {
        let current = self.generation.borrow().clone();
        let inherited = current.registry()?;
        let mut artifacts = current.artifacts()?.as_ref().clone();
        let source_owners = current.source_owners()?;
        if let Some(bytes) = unit_bytes {
            artifacts.insert_canonical_unit(bytes).map_err(|source| {
                build_runtime_error(source.to_string())
                    .with_identifier("RunMat:CallableArtifact:InvalidExecutable")
                    .with_source(source)
                    .build()
            })?;
        }
        let active = Rc::new(FunctionRegistryGeneration::active(
            active,
            artifacts,
            source_owners,
            &inherited,
        ));
        let restore = self.replace_generation(active.clone());
        Ok(FunctionRegistryPublication {
            evaluator: self.clone(),
            active,
            restore: Some(restore),
        })
    }

    fn capture_with_generation(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
        generation: Rc<FunctionRegistryGeneration>,
    ) -> Result<SessionSourceEvaluation, RuntimeError> {
        capture_source(
            runtime,
            source,
            source_label,
            requested_outputs,
            self.base_workspace.clone(),
            generation,
        )
    }
}

impl RuntimeSourceEvaluationService for SessionSourceEvaluator {
    fn bind_background_authority(
        &self,
    ) -> Result<Rc<dyn RuntimeSourceEvaluationService>, RuntimeError> {
        Ok(Rc::new(BoundSessionSourceEvaluator {
            base_workspace: self.base_workspace.clone(),
            generation: self.generation.borrow().clone(),
        }))
    }

    fn capture_session_source(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Result<SessionSourceEvaluation, RuntimeError> {
        self.capture_with_generation(
            runtime,
            source,
            source_label,
            requested_outputs,
            self.generation.borrow().clone(),
        )
    }

    fn evaluate_session_source(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<ValueSequence, RuntimeError>> {
        let Some(authority) = request.evaluation_authority() else {
            return Box::pin(async {
                Err(
                    build_runtime_error("source evaluation lacks captured executable authority")
                        .with_identifier("RunMat:SourceEvaluation:MissingAuthority")
                        .build(),
                )
            });
        };
        authority.evaluate(runtime, request)
    }
}

impl RuntimeSourceEvaluationService for BoundSessionSourceEvaluator {
    fn bind_background_authority(
        &self,
    ) -> Result<Rc<dyn RuntimeSourceEvaluationService>, RuntimeError> {
        Ok(Rc::new(Self {
            base_workspace: self.base_workspace.clone(),
            generation: self.generation.clone(),
        }))
    }

    fn capture_session_source(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Result<SessionSourceEvaluation, RuntimeError> {
        capture_source(
            runtime,
            source,
            source_label,
            requested_outputs,
            self.base_workspace.clone(),
            self.generation.clone(),
        )
    }

    fn evaluate_session_source(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<ValueSequence, RuntimeError>> {
        let Some(authority) = request.evaluation_authority() else {
            return Box::pin(async {
                Err(
                    build_runtime_error("source evaluation lacks captured executable authority")
                        .with_identifier("RunMat:SourceEvaluation:MissingAuthority")
                        .build(),
                )
            });
        };
        authority.evaluate(runtime, request)
    }
}

fn capture_source(
    runtime: &RuntimeContext,
    source: Arc<str>,
    source_label: Arc<str>,
    requested_outputs: usize,
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    generation: Rc<FunctionRegistryGeneration>,
) -> Result<SessionSourceEvaluation, RuntimeError> {
    generation.registry()?;
    let lease = runmat_runtime::source_context::lease_source_context_in_context(
        runtime,
        Arc::clone(&source_label),
        None,
        Arc::clone(&source),
    )
    .map_err(|error| {
        build_runtime_error(error.to_string())
            .with_identifier("RunMat:SourceEvaluation:SourceIdentityExhausted")
            .with_source(error)
            .build()
    })?;
    Ok(SessionSourceEvaluation::capture_with_authority(
        runtime,
        source,
        Rc::new(lease),
        Rc::new(CapturedSourceEvaluationAuthority {
            base_workspace,
            generation,
        }),
        source_label,
        requested_outputs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_session_owner_rejects_prepared_source_before_execution() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let evaluator = Rc::new(SessionSourceEvaluator::new(
            owner.clone(),
            runtime.local_identity(),
        ));
        let base = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
        let runtime = runtime.with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_workspace(base.clone())
                .with_base_workspace(base)
                .with_source_evaluation(evaluator),
        );
        let request = SessionSourceEvaluation::prepare(
            &runtime,
            Arc::from("x = 1;"),
            Arc::from("<closed-session-source>"),
            0,
        )
        .expect("capture source while owner is live");
        owner.close().expect("close session workspace");

        let error = futures::executor::block_on(request.evaluate(runtime))
            .expect_err("closed session must reject callback source");
        assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    }

    #[test]
    fn failed_foreground_generation_invalidates_prepared_background_source() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let evaluator = Rc::new(SessionSourceEvaluator::new(
            owner.clone(),
            runtime.local_identity(),
        ));
        let publication = evaluator
            .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
            .expect("publish active generation");
        let bound = evaluator
            .bind_background_authority()
            .expect("bind active executable generation");
        let base = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(owner));
        let runtime = runtime.with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_workspace(base.clone())
                .with_base_workspace(base)
                .with_source_evaluation(bound),
        );
        let request = SessionSourceEvaluation::prepare(
            &runtime,
            Arc::from("x = 1;"),
            Arc::from("<invalidated-source>"),
            0,
        )
        .expect("capture while foreground generation is active");

        drop(publication);
        let error = futures::executor::block_on(request.evaluate(runtime))
            .expect_err("failed foreground generation must not execute later");
        assert_eq!(
            error.identifier(),
            Some("RunMat:SourceEvaluation:GenerationInvalidated")
        );
    }
}
