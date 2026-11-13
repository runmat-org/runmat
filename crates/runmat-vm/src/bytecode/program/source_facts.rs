use std::{collections::HashMap, rc::Rc, sync::Arc};

use super::{FunctionBytecode, FunctionRegistry};
use runmat_hir::FunctionId;

impl FunctionRegistry {
    /// Explicit source issuer for one selected registry and invocation context.
    /// The same value may be installed as the legacy resolver guard, but exact
    /// retained bindings do not depend on that ambient installation.
    pub fn execution_source_resolver_in(
        self: &Rc<Self>,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Rc<runmat_runtime::user_functions::ExecutionSourceFacts> {
        let by_function = Rc::clone(self);
        let by_source = Rc::clone(self);
        let by_owners = Rc::clone(self);
        let by_origin = Rc::clone(self);
        let by_static_sites = Rc::clone(self);
        let by_selected_class = Rc::clone(self);
        let function_runtime = runtime.clone();
        let static_runtime = runtime.clone();
        let source_runtime = runtime.clone();
        let origin_runtime = runtime.clone();
        Rc::new(
            runmat_runtime::user_functions::ExecutionSourceFacts::new(
                Rc::new(move |function| {
                    by_function.execution_source_fact_in(&function_runtime, FunctionId(function))
                }),
                Rc::new(move |source| {
                    by_source.execution_source_group_facts_in(&source_runtime, source)
                }),
                Rc::new(move |function| {
                    by_owners.execution_definition_owners(FunctionId(function))
                }),
                Rc::new(move |origin| {
                    by_origin.execution_source_fact_for_origin_in(&origin_runtime, origin)
                }),
            )
            .with_static_sites(Rc::new(move |function, info, owner| {
                by_static_sites.execution_static_site_leases(&static_runtime, function, info, owner)
            }))
            .with_selected_class(Rc::new(move |binding| {
                by_selected_class.selected_class_projection_for_binding(binding)
            })),
        )
    }

    /// Read a selected constructor product through its counted definition
    /// lease and live issued source. This is metadata admission, not a new
    /// constructor invocation or class-name lookup.
    pub fn selected_class_projection_for_binding(
        &self,
        binding: &runmat_value::BoundFunctionBinding,
    ) -> Option<Rc<runmat_runtime::class_registry::SelectedClassProjection>> {
        let lease = binding.authority.local_lease(binding.function)?;
        let body = self.get_owned(FunctionId(binding.function), lease)?;
        let method = body.class_method_owner.as_ref()?;
        if method.role != runmat_types::ClassMethodRole::Constructor || method.is_static {
            return None;
        }
        let source = body.selected_source_owner.as_ref()?;
        let product = body.selected_class_product.as_ref()?;
        if !source.is_live()
            || !binding.authority.admits_local(&source.session_lease())
            || !body
                .source_id
                .and_then(super::BytecodeSourceRef::runtime)
                .is_some_and(|source_id| source.owns_source(source_id))
            || product.class() != &method.declaring_class
        {
            return None;
        }
        product.invocation_projection().ok()
    }

    /// Issue site leases only from a body whose selected product, target
    /// owners, and retained registry bodies all agree. A colliding numeric ID
    /// selects by the counted owner, never by the current name/body.
    fn execution_static_site_leases(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        function: usize,
        info: Option<&runmat_runtime::user_functions::SourceFunctionInfo>,
        owner: &Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>> {
        let lease = runmat_value::FunctionDefinitionOwner::issue(owner, function);
        let body = self.get_owned(FunctionId(function), &lease)?;
        if !Arc::ptr_eq(&body.definition_owner, owner) {
            return None;
        }
        match info {
            Some(info) => {
                if info.function != function
                    || body.binding_origin != info.binding_origin
                    || body.source_id?.resolve_in(runtime)? != info.source_id
                {
                    return None;
                }
            }
            None => {
                // A compiler ordinal without an installed source table is not
                // issued source authority. Exact local definitions still permit
                // callbacks, but never selected imports or portable exports.
                let issued_source = body.source_id.is_some_and(|source| match source {
                    super::BytecodeSourceRef::Program(ordinal) => {
                        runtime.program_source_id(ordinal).is_some()
                    }
                    super::BytecodeSourceRef::Runtime(_) => true,
                });
                if issued_source
                    || body.binding_origin.is_some()
                    || body.selected_source_owner.is_some()
                    || body.selected_executable_product.is_some()
                    || body.selected_static_capture.is_some()
                {
                    return None;
                }
            }
        }
        self.static_site_leases_for_body(body)
    }

    /// Core's bound-callable issuer has an immutable selected registry and
    /// exact source records, but must not retain its RuntimeContext inside a
    /// resolver installed on that same context. Prove the issued source from
    /// the body's selected owner, then use the same site/target lease proof as
    /// VM-local issuance. No current-name or numeric-body fallback is used.
    pub fn selected_static_site_leases_for_issued_info(
        &self,
        info: &runmat_runtime::user_functions::SourceFunctionInfo,
        owner: &Arc<runmat_value::FunctionDefinitionOwner>,
    ) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>> {
        let lease = runmat_value::FunctionDefinitionOwner::issue(owner, info.function);
        let body = self.get_owned(FunctionId(info.function), &lease)?;
        let source_owner = body.selected_source_owner.as_ref()?;
        let source = body.source_id?.runtime()?;
        let product = body.selected_executable_product.as_ref()?;
        if !source_owner.is_live()
            || !Arc::ptr_eq(&body.definition_owner, owner)
            || body.binding_origin != info.binding_origin
            || body.source_role != info.role
            || !source_owner.matches_issued_source_id(source, info.source_id)
            || !product.source_owner().same_issuance(source_owner)
        {
            return None;
        }
        self.static_site_leases_for_body(body)
    }

    pub(super) fn static_site_leases_for_body(
        &self,
        body: &FunctionBytecode,
    ) -> Option<Vec<runmat_value::StaticSiteDefinitionLease>> {
        let Some(capture) = body.selected_static_capture.as_ref() else {
            return body
                .selected_executable_product
                .as_ref()
                .is_none_or(|product| {
                    product.static_imports().is_empty()
                        && product.linked_native_imports().is_empty()
                })
                .then(Vec::new);
        };
        let product = body.selected_executable_product.as_ref()?;
        if !Rc::ptr_eq(product, capture.product()) {
            return None;
        }
        let leases = capture.issue_site_leases()?;
        for (edge, lease) in capture.dependencies().iter().zip(&leases) {
            let target =
                self.get_owned(FunctionId(lease.definition.function()), &lease.definition)?;
            if target.binding_origin != Some(lease.target)
                || target
                    .selected_executable_product
                    .as_ref()
                    .is_none_or(|selected| !Rc::ptr_eq(selected, edge.target()))
                || target
                    .selected_source_owner
                    .as_ref()
                    .is_none_or(|owner| !owner.same_issuance(edge.target().source_owner()))
            {
                return None;
            }
            for dependency in lease.dependencies.iter() {
                self.get_owned(FunctionId(dependency.function()), dependency)?;
            }
        }
        Some(leases)
    }

    /// Bodies selected by the immutable execution view. A static overlay can
    /// retain a body after its numeric ID leaves the published registry.
    pub(crate) fn selected_execution_bodies(&self) -> Vec<&FunctionBytecode> {
        if let Some(overlay) = &self.static_overlay {
            let runmat_value::BoundFunctionAuthority::Local {
                definition,
                dependencies,
                ..
            } = &overlay.authority
            else {
                return Vec::new();
            };
            std::iter::once(definition)
                .chain(dependencies.iter())
                .filter_map(|lease| self.get_owned(FunctionId(lease.function()), lease))
                .collect()
        } else {
            self.functions.values().collect()
        }
    }

    /// Check every selected body referring to this issued source, including
    /// bodies retained by an immutable static execution overlay.
    pub fn selected_source_issuance_agrees_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        source: runmat_types::SourceId,
        owner: &runmat_runtime::source_context::SelectedProgramSourceOwner,
    ) -> bool {
        self.selected_execution_bodies().into_iter().all(|body| {
            let Some(reference) = body.source_id.and_then(super::BytecodeSourceRef::runtime) else {
                return true;
            };
            reference.resolve_in(runtime) != Some(source)
                || body
                    .selected_source_owner
                    .as_ref()
                    .is_some_and(|selected| selected.same_issuance(owner))
        })
    }

    /// Capture exact selected-definition name facts while source authority is
    /// installed. The callback later reads only these facts and never
    /// reconstructs a reused numeric ID's path from a newer publication.
    pub fn semantic_name_candidate_facts_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        functions: impl IntoIterator<Item = FunctionId>,
    ) -> HashMap<FunctionId, runmat_runtime::user_functions::SemanticNameCandidate> {
        functions
            .into_iter()
            .filter_map(|id| {
                let body = self.get(id)?;
                Some((
                    id,
                    runmat_runtime::user_functions::SemanticNameCandidate {
                        function: id.0,
                        source: self.execution_source_fact_in(runtime, id),
                        definition_owner: Some(Arc::clone(&body.definition_owner)),
                    },
                ))
            })
            .collect()
    }

    /// Facts from the selected executable body, never from a same-numbered
    /// later publication. The dependency graph is issued on demand only when
    /// a callable value is created.
    pub fn execution_source_fact_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        function: FunctionId,
    ) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
        let body = self.get(function)?;
        source_fact(runtime, body, None)
    }

    /// Resolve an immutable portable origin only when it identifies exactly
    /// one body in this selected registry. Display names and destination-local
    /// function numbers are not origin authority.
    pub fn execution_source_fact_for_origin_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        origin: runmat_execution::FunctionBindingOrigin,
    ) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
        let mut bodies = self
            .selected_execution_bodies()
            .into_iter()
            .filter(|body| body.binding_origin == Some(origin));
        let body = bodies.next()?;
        if bodies.next().is_some() {
            return None;
        }
        source_fact(runtime, body, None)
    }

    pub fn execution_definition_owners(
        &self,
        function: FunctionId,
    ) -> Option<runmat_runtime::user_functions::DefinitionOwnerGraph> {
        self.definition_dependency_owners(function).map(owners_arc)
    }

    /// Source members share one immutable owner graph. A retained view only
    /// enumerates definitions captured by that view, never published replacements.
    pub fn execution_source_group_facts_in(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        source: runmat_types::SourceId,
    ) -> Vec<runmat_runtime::user_functions::SourceFunctionInfo> {
        let Some(source) =
            runmat_runtime::source_context::issued_source_ref_in_context(runtime, source)
        else {
            return Vec::new();
        };
        let bodies = self.execution_source_group(runtime, source);
        let owners = bodies
            .first()
            .and_then(|body| self.definition_dependency_owners(body.function))
            .map(owners_arc);
        bodies
            .into_iter()
            .filter_map(|body| source_fact(runtime, body, owners.clone()))
            .collect()
    }
}

fn owners_arc(
    owners: Vec<(FunctionId, Arc<runmat_value::FunctionDefinitionOwner>)>,
) -> runmat_runtime::user_functions::DefinitionOwnerGraph {
    Arc::new(
        owners
            .into_iter()
            .map(|(id, owner)| (id.0, owner))
            .collect(),
    )
}

fn source_fact(
    runtime: &runmat_runtime::context::RuntimeContext,
    body: &FunctionBytecode,
    definition_owners: Option<runmat_runtime::user_functions::DefinitionOwnerGraph>,
) -> Option<runmat_runtime::user_functions::SourceFunctionInfo> {
    let source_id = body.source_id?.resolve_in(runtime)?;
    let source_file = runmat_runtime::source_context::source_info_in_context(runtime, source_id)
        .and_then(|source| source.fullpath_name.map(|path| path.to_string()))
        .or_else(|| body.source_file.clone());
    Some(runmat_runtime::user_functions::SourceFunctionInfo {
        source_id,
        name: body.display_name.clone(),
        function: body.function.0,
        role: body.source_role,
        binding_origin: body.binding_origin,
        source_file,
        definition_owners,
    })
}
