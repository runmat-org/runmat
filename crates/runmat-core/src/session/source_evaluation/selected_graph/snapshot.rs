use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use runmat_execution::value::{
    GraphBoundOwner, GraphClassDependency, GraphClassMethodDependency, GraphLexicalLayout,
    GraphNodeRef, GraphProgramUnitId, GraphStaticCaller, GraphStaticDependency, PayloadGraph,
    ValueLimits,
};
use runmat_execution::ExecutableSourceMapV2;
use runmat_runtime::execution::value_codec::{
    GraphCallableProof, GraphCaptureProof, GraphSourceIssuer, ValueCodecError,
};
use runmat_runtime::execution::{
    DeferredGraphCapture, DeferredGraphProgram, DeferredGraphSnapshot, DeferredInvocation,
    ExecutionServiceError, GraphResultReceipt, RootedValueSequence,
};
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_runtime::RuntimeError;
use runmat_value::{BoundFunctionBinding, ValueSequence};

use super::selection::{invalid, SelectedGraphUnits};
use super::{select_local_body, SelectedGraphPrimary};
use crate::session::source_evaluation::SessionSourceEvaluator;

/// An invocation-local selector owns only immutable code/source facts and a
/// weak path to the canonical generation. A delayed Future can therefore see
/// a newly written handle cell at spawn without retaining a service cycle or
/// selecting by its display name.
pub(in crate::session) struct SelectedGraphSnapshot {
    primary: Rc<SelectedGraphPrimary>,
    evaluator: Weak<SessionSourceEvaluator>,
}

impl SelectedGraphSnapshot {
    pub(in crate::session) fn program(
        primary: Rc<SelectedGraphPrimary>,
        evaluator: &Rc<SessionSourceEvaluator>,
    ) -> DeferredGraphProgram {
        DeferredGraphProgram::new(Rc::new(Self {
            primary,
            evaluator: Rc::downgrade(evaluator),
        }))
    }
}

impl DeferredGraphSnapshot for SelectedGraphSnapshot {
    fn snapshot(
        &self,
        invocation: &DeferredInvocation,
        limits: ValueLimits,
    ) -> Result<DeferredGraphCapture, ValueCodecError> {
        let preliminary = GraphSourceIssuer::capture_owner(self.primary.source_owner());
        let (prepared, roots) =
            runmat_runtime::execution::value_codec::prepare_deferred_invocation_graph(
                invocation,
                &preliminary,
                limits,
            )?;
        // No evaluator/generation borrow spans the rooted traversal above or
        // the numerical and exception leaf encoding below.
        let evaluator = self
            .evaluator
            .upgrade()
            .ok_or_else(|| invalid("selected graph session has ended"))?;
        let generation = evaluator.current_generation();
        let registry = generation.registry().map_err(generation_error)?;
        let artifacts = generation.artifacts().map_err(generation_error)?;
        let source_owners = generation.source_owners().map_err(generation_error)?;
        drop(generation);

        let mut units = SelectedGraphUnits::new(&self.primary)?;
        let mut views = vec![Rc::clone(&self.primary.selected_registry)];
        let mut node_units = HashMap::<GraphNodeRef, GraphProgramUnitId>::new();
        let mut class_dependencies = Vec::<GraphClassDependency>::new();
        let mut class_dependency_set = HashSet::<GraphClassDependency>::new();
        let mut class_method_dependencies = HashMap::new();
        let mut static_dependencies = Vec::<GraphStaticDependency>::new();
        let mut static_dependency_sites = HashMap::new();
        let mut object_classes =
            HashMap::<GraphBoundOwner, Rc<runmat_value::SelectedObjectClassFacts>>::new();
        for target in self.primary.static_targets() {
            if !target
                .product
                .source_owner()
                .same_issuance(&target.source_owner)
            {
                return Err(invalid("primary static target changed its issued source"));
            }
            let portable = runmat_vm::SelectedStaticProductTarget::new(
                target.import,
                Rc::clone(&target.product),
            )
            .map_err(invalid)?
            .with_dependencies(
                crate::session::source_evaluation::generation::capture_static_product_dependencies(
                    target,
                )
                .map_err(|_| invalid("primary static capture is incomplete"))?,
            )
            .map_err(invalid)?;
            let target_unit = units.add_static_product_target(&artifacts, &portable)?;
            append_static_edge(
                target.import,
                GraphProgramUnitId(0),
                target_unit,
                target.product.full_digest(),
                &mut static_dependencies,
                &mut static_dependency_sites,
            )?;
            append_captured_dependencies(
                &mut units,
                &artifacts,
                &portable,
                &mut static_dependencies,
                &mut static_dependency_sites,
            )?;
            views.push(Rc::new(target.view.clone()));
        }
        for (node, origin, authority) in prepared.bound_authorities() {
            let selected = select_local_body(&self.primary.selected_registry, *origin, authority)
                .or_else(|_| select_local_body(&registry, *origin, authority))?;
            let mut target_unit = None;
            for dependency in selected.dependencies(authority)? {
                let unit = units.add_dependency(&artifacts, &dependency)?;
                if let Some(capture) = dependency.static_capture.as_ref() {
                    if units.unit_for_product(capture.product())? != unit {
                        return Err(invalid("callable static capture changed its selected unit"));
                    }
                    append_product_dependencies(
                        &mut units,
                        &artifacts,
                        capture.dependencies(),
                        &mut static_dependencies,
                        &mut static_dependency_sites,
                    )?;
                }
                if dependency.origin == *origin
                    && dependency
                        .source_owner
                        .same_issuance(&selected.source_owner)
                {
                    target_unit = Some(unit);
                }
            }
            let target_unit = target_unit.ok_or_else(|| {
                invalid("root callable is absent from its exact dependency graph")
            })?;
            let caller = GraphBoundOwner {
                unit: target_unit,
                origin: *origin,
            };
            if let Some(facts) = selected.selected_receiver_facts(authority)? {
                if object_classes
                    .get(&caller)
                    .is_some_and(|prior| prior.as_ref() != facts.as_ref())
                {
                    return Err(invalid(
                        "selected constructor has conflicting receiver facts",
                    ));
                }
                object_classes.insert(caller, facts);
                append_class_method_edges(
                    &units,
                    &selected.view,
                    caller,
                    object_classes[&caller].as_ref(),
                    &mut class_method_dependencies,
                )?;
            }
            for declaration in selected.declaration_dependencies() {
                let declaration_unit = units.add_declaration(&declaration)?;
                let edge = GraphClassDependency {
                    caller,
                    declaration_unit,
                    declaration_digest: runmat_execution::Digest::sha256(&declaration.bytes),
                };
                if class_dependency_set.insert(edge.clone()) {
                    class_dependencies.push(edge);
                }
                for target in &declaration.static_targets {
                    let target_unit = units.add_static_product_target(&artifacts, target)?;
                    append_static_edge(
                        target.import(),
                        declaration_unit,
                        target_unit,
                        target.product().full_digest(),
                        &mut static_dependencies,
                        &mut static_dependency_sites,
                    )?;
                    append_captured_dependencies(
                        &mut units,
                        &artifacts,
                        target,
                        &mut static_dependencies,
                        &mut static_dependency_sites,
                    )?;
                }
            }
            if node_units.insert(*node, target_unit).is_some() {
                return Err(invalid("one graph node repeats its callable authority"));
            }
            views.push(Rc::new(selected.view));
        }
        for (source, owner, requirement) in prepared.exception_frame_sources() {
            units.add_frame_source(source, owner.as_ref(), requirement, &source_owners)?;
        }
        let selected_owners = units.owner_units()?;
        let (layouts, proofs, bindings) =
            selected_callable_facts(&units, &selected_owners, &views)?;
        for (owner, facts) in &object_classes {
            if bindings.get(owner) != Some(facts.constructor()) {
                return Err(invalid(
                    "selected receiver constructor differs from its admitted graph binding",
                ));
            }
        }
        // A constructor reached inside a selected callable may return an
        // object even when no input node carried that receiver. Recover only
        // products in the already-captured immutable definition views, then
        // pair them with the exact unit-qualified binding issued above.
        for (owner, binding) in &bindings {
            let unit = units
                .units()
                .get(usize::from(owner.unit.0))
                .ok_or_else(|| invalid("selected receiver unit is absent"))?;
            for view in &views {
                let Some(lease) = binding.authority.local_lease(binding.function) else {
                    continue;
                };
                let Some(body) = view.get_owned(runmat_hir::FunctionId(binding.function), lease)
                else {
                    continue;
                };
                let Some(method) = body.class_method_owner.as_ref() else {
                    continue;
                };
                if method.role != runmat_types::ClassMethodRole::Constructor {
                    continue;
                }
                let Some(product) = body.selected_class_product.as_ref() else {
                    return Err(invalid("selected constructor lacks its class product"));
                };
                if method.is_static
                    || body.binding_origin != Some(owner.origin)
                    || product.class() != &method.declaring_class
                    || body
                        .selected_source_owner
                        .as_ref()
                        .is_none_or(|source| !source.same_issuance(&unit.source_owner))
                {
                    return Err(invalid("selected constructor differs from its V3 unit"));
                }
                let facts = product
                    .receiver_facts_with_binding(view, binding)
                    .map_err(|error| invalid(error.to_string()))?;
                if object_classes
                    .get(owner)
                    .is_some_and(|prior| prior.as_ref() != facts.as_ref())
                {
                    return Err(invalid("selected constructor has conflicting class facts"));
                }
                object_classes.insert(*owner, facts);
                append_class_method_edges(
                    &units,
                    view,
                    *owner,
                    object_classes[owner].as_ref(),
                    &mut class_method_dependencies,
                )?;
            }
        }
        let mut class_method_dependencies = class_method_dependencies
            .into_values()
            .map(|edge| {
                let canonical = serde_json::to_vec(&edge)
                    .map_err(|_| invalid("selected class method edge cannot be serialized"))?;
                Ok((canonical, edge))
            })
            .collect::<Result<Vec<_>, ValueCodecError>>()?;
        class_method_dependencies.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let class_method_dependencies = class_method_dependencies
            .into_iter()
            .map(|(_, edge)| edge)
            .collect::<Vec<_>>();
        let (issuer, source_map) =
            GraphSourceIssuer::capture_selected_units(&units.source_facts())?;
        let graph = prepared.finish(&issuer, &source_map, &layouts, &proofs, &node_units)?;
        let required_schema = runmat_vm::selected_graph_execution_required_schema(
            &roots,
            &graph,
            |owner| {
                let unit = units
                    .units()
                    .get(usize::from(owner.unit.0))
                    .ok_or("selected graph executable unit is absent")?;
                let binding = bindings
                    .get(&owner)
                    .ok_or("selected graph executable binding is absent")?;
                let lease = binding
                    .authority
                    .local_lease(binding.function)
                    .ok_or("selected graph executable lease is absent")?;
                let mut selected: Option<runmat_execution::value::GraphRequiredSchema> = None;
                for view in &views {
                    let Some(body) =
                        view.get_owned(runmat_hir::FunctionId(binding.function), lease)
                    else {
                        continue;
                    };
                    if body.binding_origin != Some(owner.origin)
                        || body
                            .selected_source_owner
                            .as_ref()
                            .is_none_or(|issued| !issued.same_issuance(&unit.source_owner))
                        || body
                            .selected_executable_product
                            .as_ref()
                            .is_none_or(|product| {
                                product.canonical_v3().as_ref() != unit.bytes.as_ref()
                            })
                    {
                        continue;
                    }
                    let requirement =
                        runmat_vm::selected_graph_code_required_schema(&body.instructions);
                    selected = Some(selected.map_or(requirement, |prior| prior.max(requirement)));
                }
                selected.ok_or_else(|| "selected graph executable body is absent".into())
            },
            || {
                let unit = self
                    .primary
                    .unit
                    .as_ref()
                    .ok_or("selected graph script has no exact executable unit")?;
                Ok(runmat_vm::selected_graph_code_required_schema(
                    &unit.bytecode().instructions,
                ))
            },
        )
        .map_err(invalid)?;
        let selected_units = units
            .units()
            .iter()
            .map(|unit| (unit.source_owner.clone(), Arc::clone(&unit.bytes)))
            .collect();
        let selected_bundle_archives = units.selected_bundle_archives();
        let receipt: Rc<dyn GraphResultReceipt> = Rc::new(SelectedGraphResult {
            source_owner: self.primary.source_owner().clone(),
            required_schema,
            selected_units,
            selected_bundle_archives: selected_bundle_archives.clone(),
            class_dependencies: class_dependencies.clone(),
            class_method_dependencies: class_method_dependencies.clone(),
            static_dependencies: static_dependencies.clone(),
            issuer,
            source_map,
            layouts,
            proofs,
            bindings,
            object_classes,
            limits,
        });
        let mut selected = units.units().iter();
        let primary_unit = Arc::clone(
            &selected
                .next()
                .ok_or_else(|| invalid("selected graph has no primary V3 unit"))?
                .bytes,
        );
        let retained_units = selected.map(|unit| Arc::clone(&unit.bytes)).collect();
        let selected_foreign_policies = units.foreign_policies();
        Ok(DeferredGraphCapture {
            graph,
            required_schema,
            roots,
            primary_unit,
            retained_units,
            class_dependencies,
            class_method_dependencies,
            static_dependencies,
            selected_bundle_archives,
            selected_foreign_policies,
            receipt,
        })
    }
}

fn append_class_method_edges(
    units: &SelectedGraphUnits,
    view: &runmat_vm::FunctionRegistry,
    caller: GraphBoundOwner,
    facts: &runmat_value::SelectedObjectClassFacts,
    edges: &mut HashMap<
        (GraphBoundOwner, runmat_types::ClassMethodOwner),
        GraphClassMethodDependency,
    >,
) -> Result<(), ValueCodecError> {
    for method in facts.methods() {
        let Some(binding) = method.binding.as_ref() else {
            continue;
        };
        let lease = binding
            .authority
            .local_lease(binding.function)
            .ok_or_else(|| invalid("selected class method lacks its counted definition lease"))?;
        let body = view
            .get_owned(runmat_hir::FunctionId(binding.function), lease)
            .ok_or_else(|| invalid("selected class method lacks its exact bound body"))?;
        if body.class_method_owner.as_ref() != Some(&method.owner) {
            return Err(invalid(
                "selected class method differs from its declaration",
            ));
        }
        let origin = body
            .binding_origin
            .ok_or_else(|| invalid("selected class method has no portable origin"))?;
        let product = body
            .selected_executable_product
            .as_ref()
            .ok_or_else(|| invalid("selected class method lacks its selected product"))?;
        if body
            .selected_source_owner
            .as_ref()
            .is_none_or(|owner| !owner.same_issuance(product.source_owner()))
        {
            return Err(invalid("selected class method changed its source issuance"));
        }
        let target = GraphBoundOwner {
            unit: units.unit_for_product(product)?,
            origin,
        };
        let edge = GraphClassMethodDependency {
            caller,
            method: method.owner.clone(),
            target,
            target_unit_digest: product.full_digest(),
            target_body_digest: product
                .body_digest(origin)
                .ok_or_else(|| invalid("selected class method has no body digest"))?,
        };
        match edges.entry((caller, method.owner.clone())) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(edge);
            }
            std::collections::hash_map::Entry::Occupied(slot) if slot.get() == &edge => {}
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(invalid(
                    "selected class method has conflicting exact products",
                ));
            }
        }
    }
    Ok(())
}

fn append_captured_dependencies(
    units: &mut SelectedGraphUnits,
    artifacts: &crate::session::source_evaluation::generation::CallableArtifactIndex,
    root: &runmat_vm::SelectedStaticProductTarget,
    edges: &mut Vec<GraphStaticDependency>,
    sites: &mut HashMap<
        (
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        ),
        GraphStaticDependency,
    >,
) -> Result<(), ValueCodecError> {
    append_product_dependencies(units, artifacts, root.dependencies(), edges, sites)
}

fn append_product_dependencies(
    units: &mut SelectedGraphUnits,
    artifacts: &crate::session::source_evaluation::generation::CallableArtifactIndex,
    dependencies: &[runmat_vm::SelectedStaticProductDependency],
    edges: &mut Vec<GraphStaticDependency>,
    sites: &mut HashMap<
        (
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        ),
        GraphStaticDependency,
    >,
) -> Result<(), ValueCodecError> {
    let mut target_units = Vec::with_capacity(dependencies.len());
    for dependency in dependencies {
        let target = runmat_vm::SelectedStaticProductTarget::new(
            dependency.import(),
            Rc::clone(dependency.target()),
        )
        .map_err(invalid)?;
        let target_unit = units.add_static_product_target(artifacts, &target)?;
        target_units.push(target_unit);
    }
    for (dependency, target_unit) in dependencies.iter().zip(target_units) {
        let caller_unit = units.unit_for_product(dependency.caller())?;
        append_static_edge(
            dependency.import(),
            caller_unit,
            target_unit,
            dependency.target().full_digest(),
            edges,
            sites,
        )?;
    }
    Ok(())
}

fn append_static_edge(
    import: runmat_execution::executable::ExecutableStaticImport,
    caller_unit: GraphProgramUnitId,
    target_unit: GraphProgramUnitId,
    target_unit_digest: runmat_execution::Digest,
    edges: &mut Vec<GraphStaticDependency>,
    sites: &mut HashMap<
        (
            GraphStaticCaller,
            runmat_execution::value::GraphStaticReferenceSite,
        ),
        GraphStaticDependency,
    >,
) -> Result<(), ValueCodecError> {
    let caller = match import.caller {
        runmat_execution::executable::ExecutableStaticCaller::Script => {
            GraphStaticCaller::Script(caller_unit)
        }
        runmat_execution::executable::ExecutableStaticCaller::Function(origin) => {
            GraphStaticCaller::Function(GraphBoundOwner {
                unit: caller_unit,
                origin,
            })
        }
    };
    let edge = GraphStaticDependency {
        caller,
        site: import.site,
        target: GraphBoundOwner {
            unit: target_unit,
            origin: import.target,
        },
        target_unit_digest,
        target_body_digest: import.target_body,
    };
    match sites.get(&(caller, import.site)) {
        Some(prior) if prior == &edge => Ok(()),
        Some(_) => Err(invalid(
            "one static site selects conflicting target products",
        )),
        None => {
            sites.insert((caller, import.site), edge);
            edges.push(edge);
            Ok(())
        }
    }
}

type SelectedCallableFacts = (
    HashMap<GraphBoundOwner, GraphLexicalLayout>,
    HashMap<GraphBoundOwner, GraphCallableProof>,
    HashMap<GraphBoundOwner, BoundFunctionBinding>,
);

fn selected_callable_facts(
    units: &SelectedGraphUnits,
    selected_owners: &[(GraphProgramUnitId, SelectedProgramSourceOwner)],
    views: &[Rc<runmat_vm::FunctionRegistry>],
) -> Result<SelectedCallableFacts, ValueCodecError> {
    let mut layouts = HashMap::new();
    let mut proofs = HashMap::new();
    let mut bindings: HashMap<GraphBoundOwner, BoundFunctionBinding> = HashMap::new();
    for view in views {
        for (position, unit) in units.units().iter().enumerate() {
            let ordinal = GraphProgramUnitId(
                u16::try_from(position).map_err(|_| invalid("graph unit index exceeds u16"))?,
            );
            let selected = runmat_vm::selected_graph_callable_facts_for_owner(
                view,
                ordinal,
                &unit.source_owner,
                selected_owners,
            )
            .map_err(|_| invalid("selected callable facts are inconsistent"))?;
            let (new_layouts, new_proofs, new_bindings) = selected.into_parts();
            for (owner, binding) in new_bindings {
                if let Some(existing) = bindings.get(&owner) {
                    if existing != &binding {
                        return Err(invalid(
                            "one graph callable has conflicting definition leases",
                        ));
                    }
                } else {
                    bindings.insert(owner, binding);
                }
            }
            for (owner, layout) in new_layouts {
                if let Some(existing) = layouts.get(&owner) {
                    if existing != &layout {
                        return Err(invalid("one graph callable has conflicting lexical layout"));
                    }
                } else {
                    layouts.insert(owner, layout);
                }
            }
            for (owner, proof) in new_proofs {
                if let Some(existing) = proofs.get(&owner) {
                    if !same_proof(existing, &proof) {
                        return Err(invalid("one graph callable has conflicting source proof"));
                    }
                } else {
                    proofs.insert(owner, proof);
                }
            }
        }
    }
    Ok((layouts, proofs, bindings))
}

fn same_proof(left: &GraphCallableProof, right: &GraphCallableProof) -> bool {
    let captures = match (left.captures, right.captures) {
        (GraphCaptureProof::Exact(left), GraphCaptureProof::Exact(right)) => left == right,
        (
            GraphCaptureProof::NestedShared {
                vm_copied: left_copied,
                zero_copy: left_zero,
            },
            GraphCaptureProof::NestedShared {
                vm_copied: right_copied,
                zero_copy: right_zero,
            },
        ) => {
            left_copied == right_copied
                && std::mem::discriminant(&left_zero) == std::mem::discriminant(&right_zero)
        }
        (GraphCaptureProof::InstanceMethodReceiver, GraphCaptureProof::InstanceMethodReceiver) => {
            true
        }
        _ => false,
    };
    captures
        && left.qualified_name == right.qualified_name
        && left.source_role == right.source_role
        && left.issued_source_file == right.issued_source_file
        && left.source_file == right.source_file
}

struct SelectedGraphResult {
    source_owner: SelectedProgramSourceOwner,
    required_schema: runmat_execution::value::GraphRequiredSchema,
    selected_units: Vec<(SelectedProgramSourceOwner, Arc<[u8]>)>,
    selected_bundle_archives: Vec<Option<Arc<[u8]>>>,
    class_dependencies: Vec<GraphClassDependency>,
    class_method_dependencies: Vec<GraphClassMethodDependency>,
    static_dependencies: Vec<GraphStaticDependency>,
    issuer: GraphSourceIssuer,
    source_map: ExecutableSourceMapV2,
    layouts: HashMap<GraphBoundOwner, GraphLexicalLayout>,
    proofs: HashMap<GraphBoundOwner, GraphCallableProof>,
    bindings: HashMap<GraphBoundOwner, BoundFunctionBinding>,
    object_classes: HashMap<GraphBoundOwner, Rc<runmat_value::SelectedObjectClassFacts>>,
    limits: ValueLimits,
}

impl GraphResultReceipt for SelectedGraphResult {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same_selected_admission(&self, other: &dyn GraphResultReceipt) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        self.source_owner.same_issuance(&other.source_owner)
            && self.required_schema == other.required_schema
            && self
                .source_owner
                .durable_aliases()
                .eq(other.source_owner.durable_aliases())
            && self.selected_units.len() == other.selected_units.len()
            && self.selected_bundle_archives == other.selected_bundle_archives
            && self.selected_units.iter().zip(&other.selected_units).all(
                |((left_owner, left_bytes), (right_owner, right_bytes))| {
                    left_owner.same_issuance(right_owner)
                        && left_owner
                            .durable_aliases()
                            .eq(right_owner.durable_aliases())
                        && (Arc::ptr_eq(left_bytes, right_bytes)
                            || left_bytes.as_ref() == right_bytes.as_ref())
                },
            )
            && self.class_dependencies == other.class_dependencies
            && self.class_method_dependencies == other.class_method_dependencies
            && self.static_dependencies == other.static_dependencies
            && self.source_map == other.source_map
            && self.layouts == other.layouts
            && self.bindings == other.bindings
            && self.object_classes == other.object_classes
            && self.proofs.len() == other.proofs.len()
            && self.proofs.iter().all(|(owner, proof)| {
                other
                    .proofs
                    .get(owner)
                    .is_some_and(|other_proof| same_proof(proof, other_proof))
            })
            && self.limits == other.limits
    }

    fn check_issuing_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<(), ExecutionServiceError> {
        if !self.source_owner.is_live() {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        if !self.source_owner.is_issued_in(runtime) {
            return Err(ExecutionServiceError::ForeignScope);
        }
        Ok(())
    }

    fn decode_in_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        graph: &PayloadGraph,
    ) -> Result<RootedValueSequence, ExecutionServiceError> {
        self.check_issuing_context(runtime)?;
        if graph.required_schema() > self.required_schema {
            return Err(ExecutionServiceError::Failed(
                "selected graph response requires an unadmitted value schema".into(),
            ));
        }
        let values =
            runmat_runtime::execution::value_codec::decode_payload_graph_with_selected_classes_in_context(
                graph,
                runtime,
                &self.source_map,
                &self.issuer,
                &self.layouts,
                &self.proofs,
                &self.bindings,
                &self.object_classes,
                self.limits,
            )
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        values.with_owned_values(|values| {
            let sequence = ValueSequence::comma_separated(values)
                .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
            RootedValueSequence::new(sequence)
        })
    }

    fn decode_failure_in_context(
        &self,
        runtime: &runmat_runtime::context::RuntimeContext,
        failure: &runmat_execution::ProgramRuntimeFailure,
    ) -> Result<RuntimeError, ExecutionServiceError> {
        self.check_issuing_context(runtime)?;
        runmat_runtime::execution::decode_runtime_failure_with_source_issuer_in_context(
            failure,
            runtime,
            &self.issuer,
            &self.source_map,
        )
        .map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "selected graph failure violates its source contract: {error}"
            ))
        })
    }
}

fn generation_error(error: RuntimeError) -> ValueCodecError {
    ValueCodecError::Invalid {
        path: "$".into(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, ExecutableSourceRecordV2};
    use runmat_runtime::source_context::{
        bind_portable_source_map_v2, lease_source_context_in_context,
    };
    use runmat_types::ProgramSourceId;

    fn runtime() -> runmat_runtime::context::RuntimeContext {
        runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
    }

    fn sample_receipt(
        source_owner: SelectedProgramSourceOwner,
        source_map: ExecutableSourceMapV2,
        bytes: &[u8],
    ) -> SelectedGraphResult {
        SelectedGraphResult {
            issuer: GraphSourceIssuer::capture_owner(&source_owner),
            required_schema: runmat_execution::value::GraphRequiredSchema::V10,
            selected_units: vec![(source_owner.clone(), Arc::from(bytes))],
            selected_bundle_archives: vec![None],
            class_dependencies: Vec::new(),
            class_method_dependencies: Vec::new(),
            static_dependencies: Vec::new(),
            source_owner,
            source_map,
            layouts: HashMap::new(),
            proofs: HashMap::new(),
            bindings: HashMap::new(),
            object_classes: HashMap::new(),
            limits: ValueLimits::default(),
        }
    }

    #[test]
    fn graph_result_receipt_rejects_foreign_fetch_before_decode() {
        let text = "value = 1;";
        let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "source.m".into(),
            content_revision: Digest::sha256(text.as_bytes()),
            display_name: "source.m".into(),
            text: text.into(),
        }])
        .expect("portable source map");
        let owning = bind_portable_source_map_v2(runtime(), &map).expect("issued source map");
        let source_owner = SelectedProgramSourceOwner::capture(&owning, &map).expect("exact owner");
        let receipt = sample_receipt(source_owner.clone(), map.clone(), b"selected V3");
        let identical = sample_receipt(source_owner, map.clone(), b"selected V3");
        let different_bytes =
            sample_receipt(receipt.source_owner.clone(), map.clone(), b"other V3");
        let issued = owning
            .program_source_ref(ProgramSourceId(0))
            .expect("issued source");
        let durable = lease_source_context_in_context(&owning, "source.m", None, text)
            .expect("durable source");
        let aliased = receipt
            .source_owner
            .with_durable_aliases(&owning, [(issued, durable.source_ref())])
            .expect("validated durable alias");
        let different_aliases = sample_receipt(aliased, map.clone(), b"selected V3");
        let foreign = bind_portable_source_map_v2(runtime(), &map).expect("foreign source map");
        let foreign_owner =
            SelectedProgramSourceOwner::capture(&foreign, &map).expect("foreign source owner");
        let different_issuance = sample_receipt(foreign_owner, map, b"selected V3");
        let mut different_foreign_archive = sample_receipt(
            receipt.source_owner.clone(),
            receipt.source_map.clone(),
            b"selected V3",
        );
        different_foreign_archive.selected_bundle_archives =
            vec![Some(Arc::from(b"foreign".as_slice()))];
        assert!(receipt.same_selected_admission(&identical));
        assert!(!receipt.same_selected_admission(&different_bytes));
        assert!(!receipt.same_selected_admission(&different_aliases));
        assert!(!receipt.same_selected_admission(&different_foreign_archive));
        assert!(!receipt.same_selected_admission(&different_issuance));
        assert_eq!(receipt.check_issuing_context(&owning), Ok(()));
        assert_eq!(
            receipt.check_issuing_context(&runtime()),
            Err(ExecutionServiceError::ForeignScope)
        );
        let invalid_failure = runmat_execution::ProgramRuntimeFailure {
            message: String::new(),
            identifier: None,
            span: None,
            builtin: None,
            task_id: None,
            call_frames: Vec::new(),
            call_frames_elided: 0,
            call_stack: Vec::new(),
            phase: None,
        };
        assert!(matches!(
            receipt.decode_failure_in_context(&runtime(), &invalid_failure),
            Err(ExecutionServiceError::ForeignScope)
        ));
        assert!(matches!(
            receipt.decode_failure_in_context(&owning, &invalid_failure),
            Err(ExecutionServiceError::Infrastructure(_))
        ));
    }
}
