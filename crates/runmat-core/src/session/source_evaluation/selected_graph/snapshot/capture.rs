//! One admitted code/source selection for worker invocations and saved callbacks.

use super::*;

impl SelectedGraphSnapshot {
    pub(super) fn capture_prepared(
        &self,
        prepared: runmat_runtime::execution::value_codec::PreparedPayloadGraph<'_>,
        roots: Option<&runmat_execution::value::GraphInvocationRoots>,
        limits: ValueLimits,
    ) -> Result<runmat_runtime::execution::DeferredSelectedCodeCapture, ValueCodecError> {
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
        let mut body_schema = |owner: GraphBoundOwner| -> Result<runmat_execution::value::GraphRequiredSchema, String> {
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
                let Some(body) = view.get_owned(runmat_hir::FunctionId(binding.function), lease)
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
        };
        let selected_value_schema = match roots {
            Some(roots) => runmat_vm::selected_graph_execution_required_schema(
                roots,
                &graph,
                &mut body_schema,
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
            ),
            None => runmat_vm::selected_callable_capture_required_schema(&graph, &mut body_schema),
        }
        .map_err(invalid)?;
        // The selected-code floor is shared with legacy worker admission.
        // Only this new producer chooses the supplied-only callable argument
        // ABI; an older V10/V11 request must not be reinterpreted as V12.
        let required_schema = if roots.is_none_or(|roots| {
            matches!(
                roots,
                runmat_execution::value::GraphInvocationRoots::Callable { .. }
            )
        }) {
            selected_value_schema
                .max(runmat_execution::value::GraphRequiredSchema::CallableArgumentsV12)
        } else {
            selected_value_schema
        };
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
        Ok(runmat_runtime::execution::DeferredSelectedCodeCapture {
            graph,
            required_schema,
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
