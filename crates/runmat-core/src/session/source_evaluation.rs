use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use runmat_runtime::context::{
    RuntimeContext, RuntimeServiceFuture, RuntimeSourceEvaluationAuthority,
    RuntimeSourceEvaluationService, SessionSourceEvaluation,
};
use runmat_runtime::execution::RootedValueSequence;
use runmat_runtime::{build_runtime_error, RuntimeError};

#[cfg(test)]
#[path = "source_evaluation/terminal_tests.rs"]
mod terminal_tests;

#[cfg(not(target_arch = "wasm32"))]
type ForeignPreparation = crate::foreign_artifacts::SelectedForeignPreparation;
#[cfg(not(target_arch = "wasm32"))]
type ForeignIssuanceSource = crate::foreign_artifacts::ForeignIssuanceSource;
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
enum ForeignPreparation {}
#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
enum ForeignIssuanceSource {}

#[path = "source_evaluation/generation.rs"]
mod generation;
pub(in crate::session) use generation::SelectedDeclarationAdmission;
use generation::{
    CallableArtifactIndex, CallableArtifactSelectionError, FunctionRegistryGeneration,
    GenerationRetirement, PreparedGenerationImport,
};
#[path = "source_evaluation/publication.rs"]
mod publication;
pub(super) use publication::FunctionRegistryPublication;
#[path = "source_evaluation/callable_artifacts.rs"]
mod callable_artifacts;
#[path = "source_evaluation/dynamic_identity.rs"]
mod dynamic_identity;
#[cfg(not(target_arch = "wasm32"))]
#[path = "source_evaluation/selected_foreign.rs"]
mod selected_foreign;
#[cfg(not(target_arch = "wasm32"))]
pub(in crate::session) use selected_foreign::ordinary_native_route;
#[path = "source_evaluation/selected_graph.rs"]
mod selected_graph;
use dynamic_identity::ProjectSourceIdentityIndex;
#[path = "source_evaluation/dynamic_admission.rs"]
mod dynamic_admission;
use dynamic_admission::CoreDynamicEvalAdmission;
pub(crate) use dynamic_admission::SelectedStaticTarget;
pub(in crate::session) use dynamic_admission::{
    selected_static_imports, selected_static_imports_with_preselected, static_capture,
};
pub(crate) use selected_graph::bind_function_resolvers;
pub(super) use selected_graph::{
    SelectedGraphCodeSource, SelectedGraphPrimary, SelectedGraphPublication,
    SelectedGraphSelection, SelectedGraphSnapshot, SelectedGraphSource,
};

pub(super) use callable_artifacts::{
    admit_dynamic_callable, install_bound_invoker, service as callable_artifact_service,
    DynamicCallableDependency,
};

pub(super) struct SessionSourceEvaluator {
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    runtime_owner: runmat_runtime::context::RuntimeContextLocalId,
    generation: RefCell<Rc<FunctionRegistryGeneration>>,
    publication_head: RefCell<Option<Rc<publication::PublicationFrame>>>,
    project_sources: RefCell<Option<Rc<ProjectSourceIdentityIndex>>>,
    foreign_preparation: RefCell<Option<ForeignPreparation>>,
    foreign_issuance_source: RefCell<Option<ForeignIssuanceSource>>,
    #[cfg(not(target_arch = "wasm32"))]
    selected_foreign_hosts: selected_foreign::SelectedForeignHostCache,
    self_weak: Weak<Self>,
}

/// The compiler-bearing half of a selected native image's optional source
/// service. Construction does not grant HostRuntime or publish a port.
pub struct SelectedNativeSourceEvaluator {
    evaluator: Rc<SessionSourceEvaluator>,
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
}

impl SelectedNativeSourceEvaluator {
    pub fn new(runtime: &RuntimeContext) -> Self {
        let base_workspace = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let evaluator =
            SessionSourceEvaluator::new(base_workspace.clone(), runtime.local_identity());
        Self {
            evaluator,
            base_workspace,
        }
    }

    pub fn service(&self) -> Rc<dyn RuntimeSourceEvaluationService> {
        self.evaluator
            .source_evaluation_for_selected_body(Some(self.evaluator.dynamic_admission_port()))
    }

    pub fn base_workspace_service(&self) -> Rc<runmat_runtime::workspace::BaseWorkspaceService> {
        Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            self.base_workspace.clone(),
        ))
    }

    pub fn close(&self) -> Result<bool, runmat_runtime::workspace::BaseWorkspaceError> {
        self.base_workspace.close()
    }
}

struct CapturedSourceEvaluationAuthority {
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    generation: Rc<FunctionRegistryGeneration>,
    source_lease: runmat_runtime::source_context::RuntimeSourceLease,
    evaluator: Weak<SessionSourceEvaluator>,
    admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
}

impl std::fmt::Debug for CapturedSourceEvaluationAuthority {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CapturedSourceEvaluationAuthority")
            .finish_non_exhaustive()
    }
}

impl RuntimeSourceEvaluationAuthority for CapturedSourceEvaluationAuthority {
    fn evaluate(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<RootedValueSequence, RuntimeError>> {
        let registry = match self.generation.registry() {
            Ok(registry) => registry,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let base_workspace = self.base_workspace.clone();
        let source_lease = self.source_lease.clone();
        let evaluator = self.evaluator.clone();
        let admission = self.admission.clone();
        Box::pin(async move {
            let source_ref = request.runtime_source_ref().ok_or_else(|| {
                build_runtime_error("source evaluation lacks authenticated runtime provenance")
                    .with_identifier("RunMat:SourceEvaluation:UnauthenticatedSource")
                    .build()
            })?;
            if source_ref != source_lease.source_ref() || source_ref.resolve_in(&runtime).is_none()
            {
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
            // Dynamic source is a newly compiled, separately admitted unit.
            // It must not inherit the caller's selected V3 graph capability.
            let runtime = runtime
                .without_selected_graph_program()
                .with_language_policy(policy);
            let runtime = if matches!(
                request.purpose(),
                runmat_runtime::context::SessionSourcePurpose::WorkspaceCallback
            ) {
                let inherited_projection = runtime
                    .service_ports()
                    .base_workspace()
                    .filter(|base| base.owner().same_owner(&base_workspace))
                    .and_then(|base| base.invocation_projection());
                let mut base_service =
                    runmat_runtime::workspace::BaseWorkspaceService::new(base_workspace);
                if let Some(projection) = inherited_projection {
                    base_service = base_service.with_invocation_projection(projection)?;
                }
                let base_service = Rc::new(base_service);
                let services = runtime
                    .service_ports()
                    .clone()
                    .with_base_workspace(base_service);
                runtime.with_service_ports(services)
            } else if matches!(
                request.purpose(),
                runmat_runtime::context::SessionSourcePurpose::NumericText(
                    runmat_runtime::context::NumericTextEvaluation::All
                )
            ) {
                let mut services = runtime.service_ports().clone().without_base_workspace();
                if let Some(base) = request.explicit_base_workspace() {
                    services = services.with_base_workspace(base);
                }
                runtime.with_service_ports(services)
            } else {
                let services = runtime.service_ports().clone().without_base_workspace();
                runtime.with_service_ports(services)
            };
            let _evaluator = evaluator.upgrade().ok_or_else(|| {
                build_runtime_error("source evaluation session has ended")
                    .with_identifier("RunMat:SourceEvaluation:RetiredSession")
                    .build()
            })?;
            let admission = admission.ok_or_else(|| {
                build_runtime_error("selected source evaluation has no issuing admission")
                    .with_identifier("RunMat:SourceEvaluation:MissingSelectedAdmission")
                    .build()
            })?;
            let selected = runmat_vm::DynamicSessionEvalAdmission {
                source_lease,
                port: admission,
            };
            match request.purpose() {
                runmat_runtime::context::SessionSourcePurpose::WorkspaceCallback => {
                    runtime
                        .clone()
                        .scope(
                            runmat_vm::evaluate_session_source_with_dynamic_admission_in_context(
                                runtime,
                                source,
                                source_label,
                                source_ref,
                                requested_outputs,
                                &registry,
                                Some(selected),
                            ),
                        )
                        .await
                }
                runmat_runtime::context::SessionSourcePurpose::NumericText(method) => {
                    runtime
                        .clone()
                        .scope(runmat_vm::evaluate_numeric_text_source_in_context(
                            runtime,
                            source,
                            source_label,
                            source_ref,
                            method,
                            &registry,
                            selected,
                        ))
                        .await
                }
            }
        })
    }
}

struct BoundSessionSourceEvaluator {
    base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
    generation: Rc<FunctionRegistryGeneration>,
    evaluator: Weak<SessionSourceEvaluator>,
    admission_scope: AdmissionScope,
}

#[derive(Clone)]
enum AdmissionScope {
    Session {
        project_sources: Option<Rc<ProjectSourceIdentityIndex>>,
        foreign_preparation: Option<ForeignPreparation>,
        foreign_issuance_source: Option<ForeignIssuanceSource>,
    },
    /// `None` permits pure imported code but refuses dynamic source demand.
    Selected(Option<runmat_vm::DynamicEvalAdmissionPort>),
}

impl SessionSourceEvaluator {
    pub(super) fn new(
        base_workspace: runmat_runtime::workspace::BaseWorkspaceOwner,
        runtime_owner: runmat_runtime::context::RuntimeContextLocalId,
    ) -> Rc<Self> {
        Rc::new_cyclic(|self_weak| Self {
            base_workspace,
            runtime_owner,
            generation: RefCell::new(Rc::new(FunctionRegistryGeneration::committed(
                runmat_vm::FunctionRegistry::default(),
            ))),
            publication_head: RefCell::new(None),
            project_sources: RefCell::new(None),
            foreign_preparation: RefCell::new(None),
            foreign_issuance_source: RefCell::new(None),
            #[cfg(not(target_arch = "wasm32"))]
            selected_foreign_hosts: selected_foreign::SelectedForeignHostCache::default(),
            self_weak: self_weak.clone(),
        })
    }

    pub(super) fn set_project_sources(
        &self,
        handoff: Option<&runmat_package::FrozenProjectHandoff>,
    ) {
        *self.project_sources.borrow_mut() =
            handoff.map(|handoff| Rc::new(ProjectSourceIdentityIndex::from_handoff(handoff)));
    }

    fn project_sources(&self) -> Option<Rc<ProjectSourceIdentityIndex>> {
        self.project_sources.borrow().clone()
    }

    fn foreign_preparation(&self) -> Option<ForeignPreparation> {
        self.foreign_preparation.borrow().clone()
    }

    fn foreign_issuance_source(&self) -> Option<ForeignIssuanceSource> {
        self.foreign_issuance_source.borrow().clone()
    }

    fn current_generation(&self) -> Rc<FunctionRegistryGeneration> {
        self.generation.borrow().clone()
    }

    /// Snapshot exact class-only products from this generation before a new
    /// constructor product is attached. These Rcs contain checked code/source
    /// and static-target products, never a mutable class-name lookup.
    pub(super) fn selected_declaration_products(
        &self,
    ) -> Result<Vec<Rc<runmat_vm::SelectedDeclarationProduct>>, RuntimeError> {
        Ok(self
            .current_generation()
            .selected_declarations()?
            .iter()
            .map(|admission| Rc::clone(admission.product()))
            .collect())
    }

    pub(super) fn dynamic_admission_port(&self) -> runmat_vm::DynamicEvalAdmissionPort {
        self.dynamic_admission_port_for(
            self.project_sources(),
            self.foreign_preparation(),
            self.foreign_issuance_source(),
        )
    }

    pub(super) fn dynamic_admission_port_for_selected_body(
        self: &Rc<Self>,
        body: &runmat_vm::FunctionBytecode,
    ) -> Result<Option<runmat_vm::DynamicEvalAdmissionPort>, RuntimeError> {
        match (
            body.selected_executable_product.as_ref(),
            body.selected_source_owner.as_ref(),
        ) {
            (Some(product), Some(owner)) if product.source_owner().same_issuance(owner) => {
                // A Graph/AOT-admitted product may be pure code/source with no
                // local dynamic-source authority. It can still be invoked;
                // only source evaluation at demand time needs an admission port.
                Ok(product.dynamic_admission().cloned())
            }
            (None, None) => Ok(Some(self.dynamic_admission_port())),
            _ => Err(
                build_runtime_error("bound callback has no exact selected source")
                    .with_identifier("RunMat:DynamicEval:SelectedForeignSource")
                    .build(),
            ),
        }
    }

    pub(super) fn source_evaluation_for_selected_body(
        self: &Rc<Self>,
        admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
    ) -> Rc<dyn RuntimeSourceEvaluationService> {
        Rc::new(BoundSessionSourceEvaluator {
            base_workspace: self.base_workspace.clone(),
            generation: self.current_generation(),
            evaluator: Rc::downgrade(self),
            admission_scope: AdmissionScope::Selected(admission),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn dynamic_admission_port_for_invocation(
        self: &Rc<Self>,
        foreign_preparation: Option<ForeignPreparation>,
        foreign_issuance_source: Option<ForeignIssuanceSource>,
    ) -> runmat_vm::DynamicEvalAdmissionPort {
        self.dynamic_admission_port_for(
            self.project_sources(),
            foreign_preparation,
            foreign_issuance_source,
        )
    }

    fn dynamic_admission_port_for(
        &self,
        project_sources: Option<Rc<ProjectSourceIdentityIndex>>,
        foreign_preparation: Option<ForeignPreparation>,
        foreign_issuance_source: Option<ForeignIssuanceSource>,
    ) -> runmat_vm::DynamicEvalAdmissionPort {
        runmat_vm::DynamicEvalAdmissionPort::new(Rc::new(CoreDynamicEvalAdmission::new(
            self.self_weak.clone(),
            project_sources,
            foreign_preparation,
            foreign_issuance_source,
        )))
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

    pub(in crate::session) fn retains_binding(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<bool, RuntimeError> {
        Ok(self.selected_binding_view(binding)?.is_some())
    }

    pub(in crate::session) fn selected_binding_view(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<Option<runmat_vm::FunctionRegistry>, RuntimeError> {
        let generation = self.current_generation();
        let registry = generation.registry()?;
        Ok(registry.owned_view(runmat_hir::FunctionId(binding.function), &binding.authority))
    }

    fn prepare_import(
        &self,
        registry: runmat_vm::FunctionRegistry,
        artifacts: CallableArtifactIndex,
        source_owners: Vec<Rc<runmat_vm::RetainedExecutableSourceOwner>>,
    ) -> PreparedGenerationImport {
        PreparedGenerationImport::new(registry, artifacts, source_owners)
    }

    fn commit_import(
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

    pub(super) fn publish_for_execution(
        self: &Rc<Self>,
        active: runmat_vm::FunctionRegistry,
        unit_bytes: Option<Arc<[u8]>>,
    ) -> Result<FunctionRegistryPublication, RuntimeError> {
        self.publish_for_execution_with_owner(active, unit_bytes, None)
    }

    /// Stage one dynamically compiled selected product only while its eval
    /// frame executes. The ordinary publication guard restores the prior named
    /// registry on Drop; only actual escaped definition handles retain this
    /// code/source owner and its canonical V3 artifact.
    pub(super) fn publish_selected_dynamic_for_execution(
        self: &Rc<Self>,
        active: runmat_vm::FunctionRegistry,
        canonical_v3: Arc<[u8]>,
        source_owner: runmat_runtime::source_context::SelectedProgramSourceOwner,
    ) -> Result<FunctionRegistryPublication, RuntimeError> {
        // The active registry and its code/source owner are authenticated in
        // one operation. An independently constructed owner cannot certify a
        // later or different provisional registry with the same names.
        let owner = Rc::new(
            runmat_vm::RetainedExecutableSourceOwner::from_selected_product(
                Arc::clone(&canonical_v3),
                source_owner,
                &active,
            )
            .map_err(|message| {
                build_runtime_error(message)
                    .with_identifier("RunMat:CallableArtifact:InvalidSelectedProduct")
                    .build()
            })?,
        );
        self.publish_for_execution_with_owner(active, Some(canonical_v3), Some(owner))
    }

    fn publish_for_execution_with_owner(
        self: &Rc<Self>,
        active: runmat_vm::FunctionRegistry,
        unit_bytes: Option<Arc<[u8]>>,
        new_owner: Option<Rc<runmat_vm::RetainedExecutableSourceOwner>>,
    ) -> Result<FunctionRegistryPublication, RuntimeError> {
        let current = self.generation.borrow().clone();
        let inherited = current.registry()?;
        let mut artifacts = current.artifacts()?.as_ref().clone();
        let source_owners = current.source_owners()?;
        let selected_declarations = current.selected_declarations()?;
        if let Some(bytes) = unit_bytes {
            artifacts
                .insert_canonical_arc_unit(bytes)
                .map_err(|source| {
                    build_runtime_error(source.to_string())
                        .with_identifier("RunMat:CallableArtifact:InvalidExecutable")
                        .with_source(source)
                        .build()
                })?;
        }
        let source_owners = if let Some(owner) = new_owner {
            let mut selected = source_owners.as_ref().clone();
            selected.push(owner);
            Rc::new(selected)
        } else {
            source_owners
        };
        let active = Rc::new(FunctionRegistryGeneration::active(
            active,
            artifacts,
            source_owners,
            selected_declarations,
            &inherited,
        ));
        let restore = self.replace_generation(active.clone());
        Ok(FunctionRegistryPublication::new(
            self.clone(),
            active,
            restore,
        ))
    }

    fn capture_with_generation(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
        generation: Rc<FunctionRegistryGeneration>,
    ) -> Result<SessionSourceEvaluation, RuntimeError> {
        let issuer = BoundSessionSourceEvaluator {
            base_workspace: self.base_workspace.clone(),
            generation,
            evaluator: self.self_weak.clone(),
            admission_scope: AdmissionScope::Session {
                project_sources: self.project_sources(),
                foreign_preparation: self.foreign_preparation(),
                foreign_issuance_source: self.foreign_issuance_source(),
            },
        };
        capture_source(runtime, source, source_label, requested_outputs, &issuer)
    }
}

impl RuntimeSourceEvaluationService for SessionSourceEvaluator {
    fn bind_background_authority(
        &self,
    ) -> Result<Rc<dyn RuntimeSourceEvaluationService>, RuntimeError> {
        Ok(Rc::new(BoundSessionSourceEvaluator {
            base_workspace: self.base_workspace.clone(),
            generation: self.generation.borrow().clone(),
            evaluator: self.self_weak.clone(),
            admission_scope: AdmissionScope::Session {
                project_sources: self.project_sources(),
                foreign_preparation: self.foreign_preparation(),
                foreign_issuance_source: self.foreign_issuance_source(),
            },
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
    ) -> RuntimeServiceFuture<Result<RootedValueSequence, RuntimeError>> {
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

    fn evaluate_dynamic_workspace<'a>(
        &'a self,
        runtime: &'a RuntimeContext,
        invocation: runmat_runtime::context::DynamicWorkspaceInvocation<'a>,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<
            'a,
            runmat_runtime::execution::RootedValueSequence,
        >,
        RuntimeError,
    > {
        self.ensure_runtime_owner(runtime)?;
        let registry = self.current_generation().registry()?;
        let admission = self.dynamic_admission_port();
        runmat_runtime::execution::admit_dynamic_workspace_sequence(runtime, async move {
            runmat_vm::evaluate_dynamic_workspace_in_context(
                runtime,
                invocation,
                registry.as_ref(),
                Some(admission),
            )
            .await
        })
    }
}

impl RuntimeSourceEvaluationService for BoundSessionSourceEvaluator {
    fn bind_background_authority(
        &self,
    ) -> Result<Rc<dyn RuntimeSourceEvaluationService>, RuntimeError> {
        Ok(Rc::new(Self {
            base_workspace: self.base_workspace.clone(),
            generation: self.generation.clone(),
            evaluator: self.evaluator.clone(),
            admission_scope: self.admission_scope.clone(),
        }))
    }

    fn capture_session_source(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Result<SessionSourceEvaluation, RuntimeError> {
        capture_source(runtime, source, source_label, requested_outputs, self)
    }

    fn evaluate_session_source(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<RootedValueSequence, RuntimeError>> {
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

    fn evaluate_dynamic_workspace<'a>(
        &'a self,
        runtime: &'a RuntimeContext,
        invocation: runmat_runtime::context::DynamicWorkspaceInvocation<'a>,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<
            'a,
            runmat_runtime::execution::RootedValueSequence,
        >,
        RuntimeError,
    > {
        let evaluator = self.evaluator.upgrade().ok_or_else(|| {
            build_runtime_error("dynamic source session has ended")
                .with_identifier("RunMat:DynamicEval:RetiredSession")
                .build()
        })?;
        evaluator.ensure_runtime_owner(runtime)?;
        let registry = self.generation.registry()?;
        let admission = match &self.admission_scope {
            AdmissionScope::Selected(admission) => admission.clone().ok_or_else(|| {
                build_runtime_error("selected executable has no dynamic-source authority")
                    .with_identifier("RunMat:DynamicEval:SourceUnavailable")
                    .build()
            })?,
            AdmissionScope::Session {
                project_sources,
                foreign_preparation,
                foreign_issuance_source,
            } => evaluator.dynamic_admission_port_for(
                project_sources.clone(),
                foreign_preparation.clone(),
                foreign_issuance_source.clone(),
            ),
        };
        runmat_runtime::execution::admit_dynamic_workspace_sequence(runtime, async move {
            runmat_vm::evaluate_dynamic_workspace_in_context(
                runtime,
                invocation,
                registry.as_ref(),
                Some(admission),
            )
            .await
        })
    }
}

fn capture_source(
    runtime: &RuntimeContext,
    source: Arc<str>,
    source_label: Arc<str>,
    requested_outputs: usize,
    issuer: &BoundSessionSourceEvaluator,
) -> Result<SessionSourceEvaluation, RuntimeError> {
    issuer.generation.registry()?;
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
    let authority_lease = lease.clone();
    let admission = match &issuer.admission_scope {
        AdmissionScope::Selected(selected) => selected.clone(),
        AdmissionScope::Session {
            project_sources,
            foreign_preparation,
            foreign_issuance_source,
        } => Some(
            issuer
                .evaluator
                .upgrade()
                .ok_or_else(|| {
                    build_runtime_error("source evaluation session has ended")
                        .with_identifier("RunMat:SourceEvaluation:RetiredSession")
                        .build()
                })?
                .dynamic_admission_port_for(
                    project_sources.clone(),
                    foreign_preparation.clone(),
                    foreign_issuance_source.clone(),
                ),
        ),
    };
    Ok(SessionSourceEvaluation::capture_with_authority(
        runtime,
        source,
        Rc::new(lease),
        Rc::new(CapturedSourceEvaluationAuthority {
            base_workspace: issuer.base_workspace.clone(),
            generation: issuer.generation.clone(),
            source_lease: authority_lease,
            evaluator: issuer.evaluator.clone(),
            admission,
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
        let evaluator = SessionSourceEvaluator::new(owner.clone(), runtime.local_identity());
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
        let evaluator = SessionSourceEvaluator::new(owner.clone(), runtime.local_identity());
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

    #[test]
    fn numeric_text_uses_captured_generation_without_base_workspace_publication() {
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let evaluator = SessionSourceEvaluator::new(owner.clone(), runtime.local_identity());
        let _publication = evaluator
            .publish_for_execution(runmat_vm::FunctionRegistry::default(), None)
            .expect("publish exact generation");
        let bound = evaluator
            .bind_background_authority()
            .expect("bind generation");
        let base = Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
        let runtime = runtime.with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_workspace(base.clone())
                .with_base_workspace(base)
                .with_source_evaluation(bound),
        );
        let request = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "1+2",
            runmat_runtime::context::NumericTextEvaluation::Restricted,
        )
        .expect("capture numeric expression");
        let result = futures::executor::block_on(request.evaluate(runtime.clone()))
            .expect("evaluate through VM compiler");
        assert_eq!(result.as_slice().len(), 1);
        match &result.as_slice()[0] {
            runmat_value::Value::Num(value) => assert_eq!(*value, 3.0),
            runmat_value::Value::Tensor(value) => {
                assert_eq!(value.as_f64_slice(), Some([3.0].as_slice()))
            }
            other => panic!("numeric expression produced {other:?}"),
        }
        let matrix = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "1 2; 3 4",
            runmat_runtime::context::NumericTextEvaluation::Restricted,
        )
        .expect("capture numeric matrix");
        let matrix = futures::executor::block_on(matrix.evaluate(runtime.clone()))
            .expect("evaluate matrix through canonical parser");
        assert!(matches!(
            matrix.as_slice(),
            [runmat_value::Value::Tensor(value)]
                if value.shape() == [2, 2]
                    && value.as_f64_slice() == Some([1.0, 3.0, 2.0, 4.0].as_slice())
        ));
        let exponent_matrix = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "1D3 .5d-2; 2D+1 4e0",
            runmat_runtime::context::NumericTextEvaluation::Restricted,
        )
        .expect("capture decimal-exponent matrix");
        let exponent_matrix =
            futures::executor::block_on(exponent_matrix.evaluate(runtime.clone()))
                .expect("evaluate decimal exponents through the canonical parser");
        assert!(matches!(
            exponent_matrix.as_slice(),
            [runmat_value::Value::Tensor(value)]
                if value.shape() == [2, 2]
                    && value.as_f64_slice() == Some([1000.0, 20.0, 0.005, 4.0].as_slice())
        ));
        let function = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "exp(2)",
            runmat_runtime::context::NumericTextEvaluation::Restricted,
        )
        .expect("capture numerical function");
        let function = futures::executor::block_on(function.evaluate(runtime.clone()))
            .expect("evaluate through existing callable path");
        assert_eq!(function.as_slice().len(), 1);
        let observed = match &function.as_slice()[0] {
            runmat_value::Value::Num(value) => *value,
            runmat_value::Value::Tensor(value) => value.as_f64_slice().expect("double result")[0],
            other => panic!("numerical function produced {other:?}"),
        };
        assert!((observed - 2.0f64.exp()).abs() < 1e-12);
        let all_numeric = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "1+2",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture default-all numeric expression");
        let all_numeric = futures::executor::block_on(all_numeric.evaluate(runtime.clone()))
            .expect("default-all ordinary numeric expression");
        match all_numeric.as_slice() {
            [runmat_value::Value::Num(value)] => assert_eq!(*value, 3.0),
            [runmat_value::Value::Tensor(value)] => {
                assert_eq!(value.as_f64_slice(), Some([3.0].as_slice()))
            }
            other => panic!("default-all numeric expression produced {other:?}"),
        }
        let invalid = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "[1",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture malformed text");
        assert!(
            futures::executor::block_on(invalid.evaluate(runtime.clone()))
                .expect("syntax failure is failed conversion")
                .as_slice()
                .is_empty()
        );
        let malformed_matrix = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "1 2; 3",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture valid syntax with invalid matrix geometry");
        assert!(
            futures::executor::block_on(malformed_matrix.evaluate(runtime.clone()))
                .expect("source-owned matrix shape failure is failed conversion")
                .as_slice()
                .is_empty()
        );
        let selected_callee = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "reshape([1 2], 2, 2)",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture selected numerical call");
        futures::executor::block_on(selected_callee.evaluate(runtime.clone()))
            .expect_err("selected callee's shape error is not a matrix-literal conversion");
        let undefined = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "unknown_numeric_name",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture undefined expression name");
        assert!(
            futures::executor::block_on(undefined.evaluate(runtime.clone()))
                .expect("typed undefined source name is failed conversion")
                .as_slice()
                .is_empty()
        );
        let missing_call = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "missing_numeric_function(2)",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture unresolved numeric call");
        assert!(
            futures::executor::block_on(missing_call.evaluate(runtime.clone()))
                .expect("no selected target at the captured source is failed conversion")
                .as_slice()
                .is_empty()
        );
        let rejected = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "missing_numeric_function(2)",
            runmat_runtime::context::NumericTextEvaluation::Restricted,
        )
        .expect("capture rejected expression");
        assert!(
            futures::executor::block_on(rejected.evaluate(runtime.clone()))
                .expect("restricted admission returns failed conversion")
                .as_slice()
                .is_empty()
        );
        // A numeric-text evaluation has no implicit base-workspace projection.
        assert_eq!(owner.snapshot().expect("snapshot base workspace").len(), 0);

        let mut seed = runmat_runtime::workspace::WorkspaceDelta::new();
        seed.assign("base_answer", runmat_value::Value::Num(11.0))
            .expect("valid base binding");
        owner.apply(seed).expect("seed explicit base workspace");
        let explicit_base = SessionSourceEvaluation::prepare_numeric_text(
            &runtime,
            "evalin('base', 'base_answer')",
            runmat_runtime::context::NumericTextEvaluation::All,
        )
        .expect("capture explicit base access with host-granted port");
        let explicit_base = futures::executor::block_on(explicit_base.evaluate(runtime))
            .expect("All-mode explicit base read uses the captured service");
        match explicit_base.as_slice() {
            [runmat_value::Value::Num(value)] => assert_eq!(*value, 11.0),
            [runmat_value::Value::Tensor(value)] => {
                assert_eq!(value.as_f64_slice(), Some([11.0].as_slice()))
            }
            other => panic!("explicit base expression produced {other:?}"),
        }
        let snapshot = owner.snapshot().expect("base remains live");
        assert_eq!(snapshot.len(), 1);
        assert_eq!(
            snapshot.get("base_answer"),
            Some(&runmat_value::Value::Num(11.0))
        );
    }
}
