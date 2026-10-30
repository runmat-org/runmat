//! Core admission of one VM-owned, single-pass dynamic compiler product.

use std::collections::HashSet;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::source_context::RuntimeSourceLease;
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_types::{ProgramSourceId, SourceId};
use runmat_vm::{
    proves_no_graph_demand, AdmittedDynamicEval, AdmittedPendingClosure, Bytecode,
    DynamicEvalAdmissionPort, DynamicEvalCompileProduct, DynamicEvalProductAdmission,
    DynamicEvalPublication, FunctionRegistry, PendingClosureAdmissionRequest,
    SelectedExecutableAttachmentError, SelectedExecutableProduct,
};

use super::dynamic_identity::{selected_dynamic_source_identity, ProjectSourceIdentityIndex};
#[cfg(target_arch = "wasm32")]
use super::SelectedGraphSnapshot;
use super::{
    ForeignIssuanceSource, ForeignPreparation, FunctionRegistryPublication, SelectedGraphPrimary,
    SelectedGraphPublication, SessionSourceEvaluator,
};

#[path = "dynamic_admission/static_import.rs"]
mod static_import;
pub(crate) use static_import::SelectedStaticTarget;
use static_import::{
    selected_pending_closure_imports_with_preselected_and_linked,
    selected_static_imports_with_preselected_and_linked, static_capture_with_linked,
};
pub(in crate::session) use static_import::{
    selected_static_imports, selected_static_imports_with_preselected, static_capture,
};

#[derive(Clone)]
pub(super) struct CoreDynamicEvalAdmission {
    evaluator: Weak<SessionSourceEvaluator>,
    project_sources: Option<Rc<ProjectSourceIdentityIndex>>,
    foreign_preparation: Option<ForeignPreparation>,
    foreign_issuance_source: Option<ForeignIssuanceSource>,
}

/// Keep the exact no-invoke source selections counted through interpretation
/// and provisional rollback. The publication field drops first, while these
/// leases still prevent the unnamed source views from being pruned.
struct DynamicSelectedPublication {
    _publication: FunctionRegistryPublication,
    _selected_sources: Vec<runmat_value::BoundFunctionBinding>,
}

impl DynamicEvalPublication for DynamicSelectedPublication {}

impl CoreDynamicEvalAdmission {
    pub(super) fn new(
        evaluator: Weak<SessionSourceEvaluator>,
        project_sources: Option<Rc<ProjectSourceIdentityIndex>>,
        foreign_preparation: Option<ForeignPreparation>,
        foreign_issuance_source: Option<ForeignIssuanceSource>,
    ) -> Self {
        Self {
            evaluator,
            project_sources,
            foreign_preparation,
            foreign_issuance_source,
        }
    }
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message.into())
        .with_identifier("RunMat:DynamicEval:SelectedProduct")
        .build()
}

fn invalid_attachment(error: SelectedExecutableAttachmentError) -> RuntimeError {
    build_runtime_error(error.to_string())
        .with_identifier("RunMat:DynamicEval:SelectedProduct")
        .with_source(error)
        .build()
}

impl DynamicEvalProductAdmission for CoreDynamicEvalAdmission {
    fn selected_semantic_source_binding(
        &self,
        runtime: &RuntimeContext,
        function: FunctionId,
        expected_owner: &Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Result<runmat_value::BoundFunctionBinding, RuntimeError> {
        let evaluator = self
            .evaluator
            .upgrade()
            .ok_or_else(|| invalid("dynamic source session has ended"))?;
        evaluator.ensure_runtime_owner(runtime)?;
        let generation = evaluator.current_generation();
        let registry = generation.registry()?;
        let body = registry
            .get_current(function)
            .filter(|body| Arc::ptr_eq(&body.definition_owner, expected_owner))
            .ok_or_else(|| invalid("observed eval source changed its exact definition"))?;
        let binding = registry
            .exact_bound_binding(body.function)
            .filter(|binding| {
                binding
                    .authority
                    .local_definition()
                    .is_some_and(|lease| lease.matches(expected_owner))
            })
            .ok_or_else(|| invalid("observed eval source lacks its exact dependency graph"))?;
        Ok(binding)
    }

    fn selected_source_output_arity(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Result<(usize, bool), RuntimeError> {
        let evaluator = self
            .evaluator
            .upgrade()
            .ok_or_else(|| invalid("dynamic source session has ended"))?;
        let view = evaluator
            .selected_binding_view(binding)?
            .ok_or_else(|| invalid("selected eval source lost its exact body"))?;
        let lease = binding
            .authority
            .local_definition()
            .ok_or_else(|| invalid("selected eval source has no exact definition lease"))?;
        let body = view
            .get_owned(FunctionId(binding.function), lease)
            .ok_or_else(|| invalid("selected eval source body has retired"))?;
        let source_owner = body
            .selected_source_owner
            .as_ref()
            .ok_or_else(|| invalid("selected eval source has no issued source"))?;
        if !source_owner.is_live()
            || !binding
                .authority
                .admits_local(&source_owner.session_lease())
            || body
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::runtime)
                .is_none_or(|source| !source_owner.owns_source(source))
            || body
                .selected_executable_product
                .as_ref()
                .is_none_or(|product| !product.source_owner().same_issuance(source_owner))
        {
            return Err(invalid(
                "selected eval source signature lacks exact source authority",
            ));
        }
        Ok((body.output_slots.len(), body.varargout_slot.is_some()))
    }

    fn requires_selected_product(
        &self,
        canonical: &Bytecode,
        _inherited: &FunctionRegistry,
    ) -> Result<bool, RuntimeError> {
        #[cfg(target_arch = "wasm32")]
        {
            debug_assert!(self.foreign_preparation.is_none());
            debug_assert!(self.foreign_issuance_source.is_none());
        }
        Ok(!proves_no_graph_demand(canonical))
    }

    fn admit(
        &self,
        runtime: &RuntimeContext,
        lease: &RuntimeSourceLease,
        source_text: &str,
        caller_source: Option<SourceId>,
        mut product: DynamicEvalCompileProduct,
        inherited: &FunctionRegistry,
    ) -> Result<AdmittedDynamicEval, RuntimeError> {
        let evaluator = self
            .evaluator
            .upgrade()
            .ok_or_else(|| invalid("dynamic source session has ended"))?;
        evaluator.ensure_runtime_owner(runtime)?;
        if lease.source_info().text.as_ref() != source_text {
            return Err(invalid(
                "compiled dynamic text differs from its issued source",
            ));
        }
        let identity = selected_dynamic_source_identity(
            runtime,
            lease,
            caller_source,
            inherited,
            self.project_sources.as_deref(),
        )
        .map_err(invalid)?;
        let revision = super::super::dynamic::dynamic_executable_revision(
            &identity.source,
            &identity.map,
            product.compat_mode,
        )?;
        product.canonical_bytecode.source_id =
            Some(runmat_vm::BytecodeSourceRef::Program(ProgramSourceId(0)));
        super::super::dynamic::attach_dynamic_function_origins(
            &mut product.canonical_bytecode,
            &revision.program_revision,
            &identity.map,
        )
        .map_err(|error| invalid(error.to_string()))?;
        crate::ExecutableUnit::retain_vm_layout(
            &mut product.canonical_bytecode,
            &product.hir,
            &product.mir,
        )
        .map_err(|error| invalid(error.to_string()))?;
        let analysis = runmat_mir::analysis::analyze_assembly(&product.mir);
        let unit = Rc::new(
            crate::executable::ExecutableUnitDraft::prepare(
                identity.source,
                revision,
                identity.map,
                product.mir.clone(),
                analysis,
                product.canonical_bytecode.clone(),
            )
            .map_err(invalid)?
            .seal(),
        );
        let original_ids = unit
            .functions()
            .functions
            .keys()
            .copied()
            .collect::<HashSet<_>>();
        let mut preselected = std::collections::HashMap::new();
        for source in &product.selected_sources {
            let view = evaluator
                .selected_binding_view(&source.binding)?
                .ok_or_else(|| invalid("precompiled source lost its exact selected body"))?;
            if preselected
                .insert(source.alias, (source.binding.clone(), view))
                .is_some()
            {
                return Err(invalid("precompiled source repeats a compiler alias"));
            }
        }
        let mut linked_native = std::collections::HashMap::new();
        for source in &product.selected_external_sources {
            if !source.target.is_live()
                || linked_native
                    .insert(source.alias, source.target.clone())
                    .is_some()
                || preselected.contains_key(&source.alias)
            {
                return Err(invalid("precompiled linked source lost its exact target"));
            }
        }
        let static_imports = selected_static_imports_with_preselected_and_linked(
            &unit,
            inherited,
            &original_ids,
            &HashSet::new(),
            &preselected,
            &linked_native,
        )
        .map_err(invalid)?;
        let serialize = |interop| {
            unit.portable_primary_source_envelope(
                &original_ids,
                Some(unit.primary_source_id()?),
                interop,
            )
        };
        let inferred = serialize(runmat_types::InteropManifest::empty()).map_err(invalid)?;
        #[cfg(not(target_arch = "wasm32"))]
        let fresh_parent = if inferred.manifest.interop != runmat_types::InteropManifest::empty() {
            self.foreign_issuance_source
                .as_ref()
                .map(|source| source.prepare_for(&inferred.manifest.interop))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?
        } else {
            None
        };
        #[cfg(not(target_arch = "wasm32"))]
        let explicit_interop = if let Some(prepared) = fresh_parent.as_ref() {
            prepared
                .enrich_inferred_child(&inferred.manifest.interop)
                .map_err(|error| invalid(error.to_string()))?
        } else {
            self.foreign_preparation
                .as_ref()
                .map(ForeignPreparation::interop)
                .cloned()
                .unwrap_or_else(runmat_types::InteropManifest::empty)
        };
        #[cfg(target_arch = "wasm32")]
        let explicit_interop = runmat_types::InteropManifest::empty();
        let mut envelope = if explicit_interop == runmat_types::InteropManifest::empty() {
            inferred
        } else {
            serialize(explicit_interop.clone()).map_err(invalid)?
        };
        #[cfg(not(target_arch = "wasm32"))]
        if fresh_parent.is_some() && envelope.manifest.interop != explicit_interop {
            return Err(invalid(
                "dynamic foreign scope changed during selected serialization",
            ));
        }
        if !static_imports.imports.is_empty() {
            let section =
                runmat_execution::executable::static_import_section(static_imports.imports)
                    .map_err(|error| invalid(error.to_string()))?;
            envelope.manifest.optional_sections.push(section);
            envelope
                .manifest
                .optional_sections
                .sort_by(|left, right| left.name.cmp(&right.name));
        }
        if !static_imports.linked_native.is_empty() {
            let section = runmat_execution::executable::linked_native_import_section(
                static_imports
                    .linked_native
                    .iter()
                    .map(|target| target.import)
                    .collect(),
            )
            .map_err(|error| invalid(error.to_string()))?;
            envelope.manifest.optional_sections.push(section);
            envelope
                .manifest
                .optional_sections
                .sort_by(|left, right| left.name.cmp(&right.name));
        }
        let canonical_v3 = Arc::<[u8]>::from(
            envelope
                .canonical_bytes()
                .map_err(|error| invalid(error.to_string()))?,
        );
        let portable_map = unit.source_map().portable_v2().map_err(invalid)?;
        let selected_runtime =
            runmat_runtime::source_context::bind_portable_source_map_v2_with_existing_lease(
                runtime.clone().without_selected_graph_program(),
                &portable_map,
                lease.clone(),
            )
            .map_err(|error| invalid(error.to_string()))?;
        let admitted_bytecode = unit.runtime_bytecode(&selected_runtime).map_err(invalid)?;
        let mut execution_bytecode =
            product.project_admitted_bytecode(admitted_bytecode, inherited)?;
        let mut active = execution_bytecode.function_registry.clone();
        // Compiler-only aliases do not publish target names. Keep the exact
        // admitted target code in an unnamed shared view so an escaped site
        // lease can reopen old and new same-numbered bodies by owner.
        for (binding, view) in preselected.values() {
            active
                .retain_selected_site_view(view, binding)
                .map_err(invalid)?;
        }
        let current_ids = original_ids
            .into_iter()
            .map(|id| product.function_remap.get(&id).copied().unwrap_or(id))
            .collect::<HashSet<FunctionId>>();
        let execution_only = current_ids.clone();
        // This provisional product has no named publication yet. Passing its
        // active execution view as a durable registry would falsely claim
        // every eval-local body was already named-published.
        let unpublished = FunctionRegistry::default();
        let source_owner = unit
            .selected_source_owner(&selected_runtime)
            .map_err(invalid)?;
        #[cfg(not(target_arch = "wasm32"))]
        let (foreign_basis, receipt) = if let Some(prepared) = fresh_parent.as_ref() {
            let (basis, receipt) = prepared
                .issue_for_compiled_unit(Arc::clone(&canonical_v3), &source_owner)
                .map_err(|error| invalid(error.to_string()))?;
            (Some(basis), Some(receipt))
        } else {
            let scoped = self
                .foreign_preparation
                .as_ref()
                .map(|prepared| prepared.scoped_to(&envelope.manifest.interop))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?;
            let basis = if envelope.manifest.interop == runmat_types::InteropManifest::empty() {
                None
            } else {
                scoped
                    .as_ref()
                    .map(ForeignPreparation::selected_reuse_basis)
                    .transpose()
                    .map_err(|error| invalid(error.to_string()))?
            };
            let receipt = crate::foreign_artifacts::selected_unit_interop_receipt(
                Arc::clone(&canonical_v3),
                &source_owner,
                scoped.as_ref(),
            )
            .map_err(invalid)?;
            (basis, receipt)
        };
        #[cfg(target_arch = "wasm32")]
        let foreign_basis = None;
        #[cfg(target_arch = "wasm32")]
        let receipt = None;
        // The evaluator's provisional generation may roll back after an eval
        // result escapes. Bind its exact body-owned code/source/static closure
        // before publishing, so later invocation never rediscovers a target
        // through the then-current registry or caller's ambient foreign port.
        let selected_product =
            SelectedExecutableProduct::admit_with_basis_receipt_and_dynamic_admission(
                Arc::clone(&canonical_v3),
                source_owner.clone(),
                foreign_basis,
                receipt.clone(),
                Some(DynamicEvalAdmissionPort::new(Rc::new(self.clone()))),
            )
            .map_err(invalid)?;
        let capture = static_capture_with_linked(
            Rc::clone(&selected_product),
            &static_imports.targets,
            &static_imports.linked_native,
        )
        .map_err(invalid)?;
        active
            .attach_current_selected_executable_product(
                current_ids.iter().copied(),
                selected_product,
            )
            .map_err(invalid_attachment)?;
        active
            .attach_current_selected_static_capture(current_ids.iter().copied(), &capture)
            .map_err(invalid)?;
        let mut script_sites = std::collections::HashMap::new();
        for target in &static_imports.targets {
            if target.import.caller == runmat_execution::executable::ExecutableStaticCaller::Script
                && script_sites
                    .insert(target.import.site, target.binding.clone())
                    .is_some()
            {
                return Err(invalid("selected script repeats a VM static site"));
            }
        }
        for site in capture.linked_native() {
            if site.import().caller == runmat_execution::executable::ExecutableStaticCaller::Script
            {
                let target = site
                    .binding()
                    .ok_or_else(|| invalid("linked-native script site source retired"))?;
                if script_sites.insert(site.import().site, target).is_some() {
                    return Err(invalid(
                        "selected script repeats a VM or native static site",
                    ));
                }
            }
        }
        execution_bytecode.function_registry = active.clone();
        for id in &current_ids {
            if let (Some(selected), Some(bound)) = (
                active.get_current(*id),
                execution_bytecode.bound_functions.get_mut(id),
            ) {
                if Arc::ptr_eq(&selected.definition_owner, &bound.definition_owner)
                    && selected.binding_origin == bound.binding_origin
                {
                    bound.selected_executable_product =
                        selected.selected_executable_product.clone();
                    bound.selected_static_capture = selected.selected_static_capture.clone();
                }
            }
        }
        let primary = Rc::new(
            SelectedGraphPrimary::new(
                super::SelectedGraphSource {
                    unit: Rc::clone(&unit),
                    primary_source: Some(ProgramSourceId(0)),
                    code: super::SelectedGraphCodeSource::PublishedBytes(Arc::clone(&canonical_v3)),
                },
                &selected_runtime,
                active.clone(),
                SelectedGraphPublication::new(&unpublished, &execution_only),
                super::SelectedGraphSelection {
                    current_functions: current_ids,
                    interop: envelope.manifest.interop.clone(),
                    static_targets: static_imports.targets,
                },
            )
            .and_then(|primary| primary.with_selected_interop_receipt(receipt.clone()))
            .map_err(invalid)?,
        );
        let publication =
            evaluator.publish_selected_dynamic_for_execution(active, canonical_v3, source_owner)?;
        let selected_sources = preselected
            .values()
            .map(|(binding, _)| binding.clone())
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        let selected_runtime =
            evaluator.bind_selected_foreign(selected_runtime, primary, None, receipt, None)?;
        #[cfg(target_arch = "wasm32")]
        let selected_runtime = {
            let ports = selected_runtime.service_ports().clone().without_foreign();
            selected_runtime
                .with_service_ports(ports)
                .with_selected_graph_program(SelectedGraphSnapshot::program(primary, &evaluator))
        };
        let selected_runtime = if script_sites.is_empty() {
            selected_runtime
        } else {
            let script_sites = Rc::new(script_sites);
            let retained_capture = Rc::clone(&capture);
            selected_runtime.with_selected_static_site_resolver(Rc::new(move |caller, site| {
                let _retained_code = &retained_capture;
                Ok(caller
                    .is_none()
                    .then(|| script_sites.get(&site).cloned())
                    .flatten())
            }))
        };
        Ok(AdmittedDynamicEval {
            runtime: selected_runtime,
            execution_bytecode,
            result: product.result,
            publication: Box::new(DynamicSelectedPublication {
                _publication: publication,
                _selected_sources: selected_sources,
            }),
        })
    }

    fn admit_pending_closure(
        &self,
        request: PendingClosureAdmissionRequest<'_>,
    ) -> Result<AdmittedPendingClosure, RuntimeError> {
        let PendingClosureAdmissionRequest {
            runtime,
            source: lease,
            caller_source: _,
            mut product,
            compile_inherited,
            executing,
            local_function,
            execution_function,
            expected_issuing_owner,
            expected_source,
            siblings,
        } = request;
        let evaluator = self
            .evaluator
            .upgrade()
            .ok_or_else(|| invalid("dynamic source session has ended"))?;
        evaluator.ensure_runtime_owner(runtime)?;
        // The pending closure is issued from the already-bound eval unit,
        // whose portable source zero is this exact runtime lease. The outer
        // request's caller source belongs to a different source table.
        if runtime.program_source_ref(ProgramSourceId(0)) != Some(lease.source_ref()) {
            return Err(invalid(
                "pending closure lost its exact eval source projection",
            ));
        }
        let identity = selected_dynamic_source_identity(
            runtime,
            lease,
            Some(lease.source_id()),
            executing,
            self.project_sources.as_deref(),
        )
        .map_err(invalid)?;
        let local_body = product
            .canonical_bytecode
            .function_registry
            .get_current(local_function)
            .ok_or_else(|| invalid("pending closure has no exact compiled body"))?;
        if !matches!(
            local_body.source_role,
            runmat_types::FunctionSourceRole::Anonymous | runmat_types::FunctionSourceRole::Nested
        ) {
            return Err(invalid("pending definition is not a closure body"));
        }
        let fingerprint = runmat_vm::semantic_definition_fingerprint(local_body)
            .map_err(|error| invalid(error.to_string()))?;
        // Local opcode IDs and the executing registry's IDs are different
        // domains after the eval product is remapped. The original creation
        // opcode owns the execution ID; a post-await recompile may only prove
        // agreement with it, never select a fresh execution identity.
        let recompiled_execution_id = product
            .function_remap
            .get(&local_function)
            .copied()
            .unwrap_or(local_function);
        if recompiled_execution_id != execution_function {
            return Err(invalid(
                "pending closure recompile changed its issuing execution identity",
            ));
        }
        let issuing_body = executing
            .get_current(execution_function)
            .ok_or_else(|| invalid("pending closure left its exact executing view"))?;
        if !Arc::ptr_eq(&issuing_body.definition_owner, expected_issuing_owner)
            || issuing_body.source_id != expected_source
        {
            return Err(invalid(
                "pending closure differs from its exact running definition",
            ));
        }
        let mut preselected = std::collections::HashMap::new();
        let mut selected_basis = Vec::new();
        for source in &product.selected_sources {
            let view = evaluator
                .selected_binding_view(&source.binding)?
                .ok_or_else(|| invalid("late source lost its exact selected body"))?;
            let selected_lease = source
                .binding
                .authority
                .local_definition()
                .ok_or_else(|| invalid("late source has no exact definition lease"))?;
            let target = view
                .get_owned(FunctionId(source.binding.function), selected_lease)
                .ok_or_else(|| invalid("late source retired before closure admission"))?;
            let target_product = target
                .selected_executable_product
                .as_ref()
                .ok_or_else(|| invalid("late source has no selected executable product"))?;
            let target_owner = target
                .selected_source_owner
                .as_ref()
                .ok_or_else(|| invalid("late source has no issued source"))?;
            if !target_product.source_owner().same_issuance(target_owner) || !target_owner.is_live()
            {
                return Err(invalid("late source changed its exact selected issuance"));
            }
            selected_basis.push((source.alias, target_product.full_digest()));
            if preselected
                .insert(source.alias, (source.binding.clone(), view))
                .is_some()
            {
                return Err(invalid("late source repeats one compiler alias"));
            }
        }
        let mut linked_native = std::collections::HashMap::new();
        for source in &product.selected_external_sources {
            if !source.target.is_live()
                || linked_native
                    .insert(source.alias, source.target.clone())
                    .is_some()
                || preselected.contains_key(&source.alias)
            {
                return Err(invalid("late linked source lost its exact target"));
            }
            let mut fingerprint = runmat_execution::DigestStream::new();
            fingerprint.update(b"runmat-pending-linked-native-basis-v1\0");
            let selected_product = source
                .target
                .selected_product()
                .ok_or_else(|| invalid("late linked source has no selected deployment product"))?;
            fingerprint.update(selected_product.bytes());
            fingerprint.update(source.target.native_ir().bytes());
            fingerprint.update(source.target.native_function().bytes());
            selected_basis.push((source.alias, fingerprint.finish()));
        }
        let mut partitioned_siblings = HashSet::new();
        for sibling in siblings {
            if sibling.local_function == local_function
                || sibling.execution_function == execution_function
                || Arc::ptr_eq(&sibling.expected_owner, expected_issuing_owner)
                || !partitioned_siblings.insert(sibling.local_function)
            {
                return Err(invalid(
                    "pending closure repeats or aliases its Local dependency",
                ));
            }
            let selected = executing
                .get_current(sibling.execution_function)
                .ok_or_else(|| invalid("pending Local dependency left its executing view"))?;
            if !Arc::ptr_eq(&selected.definition_owner, &sibling.expected_owner)
                || selected.source_id != sibling.expected_source
            {
                return Err(invalid(
                    "pending Local dependency changed its exact running definition",
                ));
            }
            let source_owner = selected
                .selected_source_owner
                .as_ref()
                .ok_or_else(|| invalid("pending Local dependency lacks its selected source"))?;
            let source = selected
                .source_id
                .and_then(runmat_vm::BytecodeSourceRef::runtime)
                .ok_or_else(|| invalid("pending Local dependency lacks a runtime source"))?;
            let target_product = selected
                .selected_executable_product
                .as_ref()
                .ok_or_else(|| invalid("pending Local dependency lacks its selected product"))?;
            if !source_owner.is_live()
                || !source_owner.owns_source(source)
                || !target_product.source_owner().same_issuance(source_owner)
                || selected
                    .binding_origin
                    .is_none_or(|origin| target_product.body_digest(origin).is_none())
            {
                return Err(invalid(
                    "pending Local dependency changed its selected issuance",
                ));
            }
            let binding = executing
                .exact_bound_binding(sibling.execution_function)
                .ok_or_else(|| invalid("pending Local dependency lacks its counted authority"))?;
            let definition = runmat_value::FunctionDefinitionOwner::issue(
                &selected.definition_owner,
                sibling.execution_function.0,
            );
            if binding.authority.local_definition() != Some(&definition) {
                return Err(invalid(
                    "pending Local dependency selected another definition",
                ));
            }
            let view = executing
                .owned_view(sibling.execution_function, &binding.authority)
                .ok_or_else(|| invalid("pending Local dependency lacks its exact owned view"))?;
            let owned = view
                .get_owned(sibling.execution_function, &definition)
                .ok_or_else(|| invalid("pending Local dependency lost its exact owned body"))?;
            if !Arc::ptr_eq(&owned.definition_owner, &selected.definition_owner)
                || owned.source_id != selected.source_id
                || owned.binding_origin != selected.binding_origin
                || owned
                    .selected_source_owner
                    .as_ref()
                    .is_none_or(|owner| !owner.same_issuance(source_owner))
                || owned
                    .selected_executable_product
                    .as_ref()
                    .is_none_or(|product| !Rc::ptr_eq(product, target_product))
            {
                return Err(invalid(
                    "pending Local dependency owned view changed its body",
                ));
            }
            selected_basis.push((sibling.local_function, target_product.full_digest()));
            if preselected
                .insert(sibling.local_function, (binding, view))
                .is_some()
                || linked_native.contains_key(&sibling.local_function)
            {
                return Err(invalid(
                    "pending Local dependency conflicts with a source alias",
                ));
            }
        }
        selected_basis.sort_by_key(|(function, _)| function.0);
        let revision = super::super::dynamic::dynamic_pending_closure_revision(
            super::super::dynamic::dynamic_executable_revision(
                &identity.source,
                &identity.map,
                product.compat_mode,
            )?,
            local_function,
            &fingerprint,
            &selected_basis,
        )?;
        product.canonical_bytecode.source_id =
            Some(runmat_vm::BytecodeSourceRef::Program(ProgramSourceId(0)));
        super::super::dynamic::attach_dynamic_function_origins(
            &mut product.canonical_bytecode,
            &revision.program_revision,
            &identity.map,
        )
        .map_err(|error| invalid(error.to_string()))?;
        crate::ExecutableUnit::retain_vm_layout(
            &mut product.canonical_bytecode,
            &product.hir,
            &product.mir,
        )
        .map_err(|error| invalid(error.to_string()))?;
        let analysis = runmat_mir::analysis::analyze_assembly(&product.mir);
        let unit = crate::executable::ExecutableUnitDraft::prepare(
            identity.source,
            revision,
            identity.map,
            product.mir.clone(),
            analysis,
            product.canonical_bytecode.clone(),
        )
        .map_err(invalid)?
        .seal();
        let static_imports = selected_pending_closure_imports_with_preselected_and_linked(
            &unit,
            compile_inherited,
            local_function,
            &partitioned_siblings,
            &preselected,
            &linked_native,
        )
        .map_err(invalid)?;
        for sibling in siblings {
            if !static_imports.targets.iter().any(|target| {
                target.import.kind
                    == runmat_execution::executable::ExecutableStaticImportKind::PartitionedDefinition
                    && target.function == sibling.execution_function
            }) {
                return Err(invalid("pending Local dependency lost its sealed call site"));
            }
        }
        let serialize = |interop| {
            unit.portable_envelope_for_pending_closure(local_function, ProgramSourceId(0), interop)
        };
        let selected_native = product
            .selected_native_interop_for(local_function)
            .map_err(invalid)?;
        let inferred = serialize(selected_native).map_err(invalid)?;
        #[cfg(not(target_arch = "wasm32"))]
        let fresh_parent = if inferred.manifest.interop != runmat_types::InteropManifest::empty() {
            self.foreign_issuance_source
                .as_ref()
                .map(|source| source.prepare_for(&inferred.manifest.interop))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?
        } else {
            None
        };
        #[cfg(not(target_arch = "wasm32"))]
        let prepared = fresh_parent.as_ref().or(self.foreign_preparation.as_ref());
        #[cfg(not(target_arch = "wasm32"))]
        let explicit_interop = if let Some(prepared) = prepared {
            prepared
                .enrich_inferred_child(&inferred.manifest.interop)
                .map_err(|error| invalid(error.to_string()))?
        } else {
            inferred.manifest.interop.clone()
        };
        #[cfg(target_arch = "wasm32")]
        let explicit_interop = runmat_types::InteropManifest::empty();
        let mut envelope = if explicit_interop == runmat_types::InteropManifest::empty() {
            inferred
        } else {
            serialize(explicit_interop.clone()).map_err(invalid)?
        };
        #[cfg(not(target_arch = "wasm32"))]
        if fresh_parent.is_some() && envelope.manifest.interop != explicit_interop {
            return Err(invalid(
                "pending foreign scope changed during selected serialization",
            ));
        }
        if !static_imports.imports.is_empty() {
            let section =
                runmat_execution::executable::static_import_section(static_imports.imports)
                    .map_err(|error| invalid(error.to_string()))?;
            envelope.manifest.optional_sections.push(section);
            envelope
                .manifest
                .optional_sections
                .sort_by(|left, right| left.name.cmp(&right.name));
        }
        if !static_imports.linked_native.is_empty() {
            let section = runmat_execution::executable::linked_native_import_section(
                static_imports
                    .linked_native
                    .iter()
                    .map(|target| target.import)
                    .collect(),
            )
            .map_err(|error| invalid(error.to_string()))?;
            envelope.manifest.optional_sections.push(section);
            envelope
                .manifest
                .optional_sections
                .sort_by(|left, right| left.name.cmp(&right.name));
        }
        let canonical_v3 = Arc::<[u8]>::from(
            envelope
                .canonical_bytes()
                .map_err(|error| invalid(error.to_string()))?,
        );
        let portable_map = unit.source_map().portable_v2().map_err(invalid)?;
        let selected_runtime =
            runmat_runtime::source_context::bind_portable_source_map_v2_with_existing_lease(
                runtime.clone().without_selected_graph_program(),
                &portable_map,
                lease.clone(),
            )
            .map_err(|error| invalid(error.to_string()))?;
        // Authenticate the whole one-pass compiler projection, then select
        // only the closure body for this function-only executable product.
        let admitted = unit.runtime_bytecode(&selected_runtime).map_err(invalid)?;
        let projected = product.project_admitted_bytecode(admitted, compile_inherited)?;
        let body = projected
            .function_registry
            .get_current(execution_function)
            .ok_or_else(|| invalid("pending closure lost its remapped execution body"))?
            .clone();
        let source_owner = unit
            .selected_source_owner(&selected_runtime)
            .map_err(invalid)?;
        #[cfg(not(target_arch = "wasm32"))]
        let (foreign_basis, receipt) = if let Some(prepared) = fresh_parent.as_ref() {
            let (basis, receipt) = prepared
                .issue_for_compiled_unit(Arc::clone(&canonical_v3), &source_owner)
                .map_err(|error| invalid(error.to_string()))?;
            (Some(basis), Some(receipt))
        } else {
            let scoped = self
                .foreign_preparation
                .as_ref()
                .map(|prepared| prepared.scoped_to(&envelope.manifest.interop))
                .transpose()
                .map_err(|error| invalid(error.to_string()))?;
            let basis = if envelope.manifest.interop == runmat_types::InteropManifest::empty() {
                None
            } else {
                scoped
                    .as_ref()
                    .map(ForeignPreparation::selected_reuse_basis)
                    .transpose()
                    .map_err(|error| invalid(error.to_string()))?
            };
            let receipt = crate::foreign_artifacts::selected_unit_interop_receipt(
                Arc::clone(&canonical_v3),
                &source_owner,
                scoped.as_ref(),
            )
            .map_err(invalid)?;
            (basis, receipt)
        };
        #[cfg(target_arch = "wasm32")]
        let foreign_basis = None;
        #[cfg(target_arch = "wasm32")]
        let receipt = None;
        let selected_product =
            SelectedExecutableProduct::admit_with_basis_receipt_and_dynamic_admission(
                Arc::clone(&canonical_v3),
                source_owner.clone(),
                foreign_basis,
                receipt,
                Some(DynamicEvalAdmissionPort::new(Rc::new(self.clone()))),
            )
            .map_err(invalid)?;
        let capture = static_capture_with_linked(
            Rc::clone(&selected_product),
            &static_imports.targets,
            &static_imports.linked_native,
        )
        .map_err(invalid)?;
        let mut selected = FunctionRegistry::default();
        // This registry is private to admission. Its temporary name index is
        // never published; only the validated body moves to the unnamed view.
        selected.insert_replacing_name(body);
        selected
            .attach_current_selected_executable_product(
                std::iter::once(execution_function),
                selected_product,
            )
            .map_err(invalid_attachment)?;
        selected
            .attach_current_selected_static_capture(std::iter::once(execution_function), &capture)
            .map_err(invalid)?;
        let mut active = executing.clone();
        if active.get_current(execution_function).is_none_or(|body| {
            !Arc::ptr_eq(&body.definition_owner, expected_issuing_owner)
                || body.source_id != expected_source
        }) {
            return Err(invalid(
                "pending closure differs from its exact running definition",
            ));
        }
        active.remove(execution_function);
        let selected_body = selected
            .get_current(execution_function)
            .ok_or_else(|| invalid("pending closure lost its selected body"))?
            .clone();
        active
            .insert_selected_unnamed(selected_body)
            .map_err(invalid)?;
        for (binding, view) in preselected.values() {
            active
                .retain_selected_site_view(view, binding)
                .map_err(invalid)?;
        }
        let binding = active
            .exact_bound_binding(execution_function)
            .ok_or_else(|| invalid("pending closure lacks its exact counted definition graph"))?;
        let selected_body = active
            .get_current(execution_function)
            .ok_or_else(|| invalid("pending closure lost its issued body"))?;
        let source_file = selected_body
            .selected_source_owner
            .as_ref()
            .and_then(|owner| {
                selected_body
                    .source_id
                    .and_then(runmat_vm::BytecodeSourceRef::runtime)
                    .and_then(|source| owner.source_ordinal_for(source))
                    .and_then(|ordinal| owner.issued_source_info_for(ordinal))
                    .and_then(|info| info.fullpath_name.as_ref().map(ToString::to_string))
            })
            .or_else(|| selected_body.source_file.clone());
        let publication =
            evaluator.publish_selected_dynamic_for_execution(active, canonical_v3, source_owner)?;
        Ok(AdmittedPendingClosure {
            binding,
            source_file,
            publication: Box::new(DynamicSelectedPublication {
                _publication: publication,
                _selected_sources: preselected
                    .values()
                    .map(|(binding, _)| binding.clone())
                    .collect(),
            }),
        })
    }
}
