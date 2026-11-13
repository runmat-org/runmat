//! Seal compiler-typed external and proved partitioned references to their
//! exact selected parent bodies. Numeric IDs and names are never import proof.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use runmat_execution::executable::{
    ExecutableStaticCaller, ExecutableStaticImport, ExecutableStaticImportKind,
    LinkedNativeStaticImport,
};
use runmat_execution::value::GraphStaticReferenceSite;
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_value::{BoundFunctionBinding, FunctionDefinitionLease, FunctionDefinitionOwner};
use runmat_vm::{
    instruction_reference_sites, BytecodeSourceRef, FunctionReferenceKind, FunctionRegistry,
    SelectedExecutableProduct, SelectedLinkedNativeSite, SelectedStaticCapture,
    SelectedStaticProductDependency, SelectedStaticSiteOwner,
};

/// One immutable, invocation-local source selection. It keeps a definition
/// lease only until script/parallel graph capture finishes; no persistent
/// executable product points back into the selected registry.
#[derive(Clone, Debug)]
pub(crate) struct SelectedStaticTarget {
    pub import: ExecutableStaticImport,
    pub function: runmat_hir::FunctionId,
    pub definition: FunctionDefinitionLease,
    pub binding: BoundFunctionBinding,
    pub view: FunctionRegistry,
    pub product: Rc<SelectedExecutableProduct>,
    pub source_owner: SelectedProgramSourceOwner,
}

#[derive(Debug)]
pub(in crate::session) struct SelectedStaticImports {
    pub imports: Vec<ExecutableStaticImport>,
    pub targets: Vec<SelectedStaticTarget>,
    pub linked_native: Vec<SelectedLinkedNativeTarget>,
}

#[derive(Debug)]
pub(in crate::session) struct SelectedLinkedNativeTarget {
    pub import: LinkedNativeStaticImport,
    pub target: runmat_runtime::user_functions::SelectedExternalSourceTarget,
}

fn add_static_edge(
    edges: &mut Vec<SelectedStaticProductDependency>,
    owners: &mut Vec<SelectedStaticSiteOwner>,
    by_site: &mut HashMap<(usize, ExecutableStaticCaller, GraphStaticReferenceSite), usize>,
    edge: SelectedStaticProductDependency,
    owner: SelectedStaticSiteOwner,
) -> Result<(), &'static str> {
    let key = (
        Rc::as_ptr(edge.caller()) as usize,
        edge.import().caller,
        edge.import().site,
    );
    if let Some(existing) = by_site.get(&key).map(|index| &edges[*index]) {
        if existing.import() != edge.import() || !Rc::ptr_eq(existing.target(), edge.target()) {
            return Err("selected static site has contradictory captured products");
        }
        let prior = &owners[by_site[&key]];
        if prior.function() != owner.function()
            || !std::sync::Arc::ptr_eq(prior.definition(), owner.definition())
            || !prior.same_dependency_owners(&owner)
        {
            return Err("selected static site has contradictory definition owners");
        }
    } else {
        by_site.insert(key, edges.len());
        edges.push(edge);
        owners.push(owner);
    }
    Ok(())
}

/// Reuse one complete, immutable product-graph proof for ordinary and dynamic
/// publication. Product nodes never retain this graph or a definition lease.
pub(in crate::session) fn static_capture(
    product: Rc<SelectedExecutableProduct>,
    targets: &[SelectedStaticTarget],
) -> Result<Rc<SelectedStaticCapture>, &'static str> {
    static_capture_with_linked(product, targets, &[])
}

pub(in crate::session) fn static_capture_with_linked(
    product: Rc<SelectedExecutableProduct>,
    targets: &[SelectedStaticTarget],
    linked_targets: &[SelectedLinkedNativeTarget],
) -> Result<Rc<SelectedStaticCapture>, &'static str> {
    let mut edges = Vec::new();
    let mut owners = Vec::new();
    let mut by_site = HashMap::new();
    let mut linked = Vec::<SelectedLinkedNativeSite>::new();
    let mut linked_by_site = HashMap::new();
    for target in linked_targets {
        let site = SelectedLinkedNativeSite::new(
            Rc::clone(&product),
            target.import,
            target.target.clone(),
        )?;
        add_linked_site(&mut linked, &mut linked_by_site, site)?;
    }
    for target in targets {
        let body = target
            .view
            .get_owned(target.function, &target.definition)
            .ok_or("selected static target has no exact leased definition")?;
        if body.binding_origin != Some(target.import.target)
            || !target
                .binding
                .authority
                .admits_local(&target.source_owner.session_lease())
            || target
                .binding
                .authority
                .local_definition()
                .is_none_or(|lease| lease != &target.definition)
            || body
                .selected_executable_product
                .as_ref()
                .is_none_or(|selected| !Rc::ptr_eq(selected, &target.product))
            || body
                .selected_source_owner
                .as_ref()
                .is_none_or(|owner| !owner.same_issuance(&target.source_owner))
            || body
                .source_id
                .and_then(BytecodeSourceRef::runtime)
                .is_none_or(|source| !target.source_owner.owns_source(source))
        {
            return Err("selected static target changed its leased code/source product");
        }
        let dependencies = target
            .binding
            .authority
            .local_dependencies()
            .ok_or("selected static target has no local dependency graph")?;
        for dependency in dependencies {
            target
                .view
                .get_owned(runmat_hir::FunctionId(dependency.function()), dependency)
                .ok_or("selected static target dependency has retired")?;
        }
        let edge = SelectedStaticProductDependency::new(
            Rc::clone(&product),
            target.import,
            Rc::clone(&target.product),
        )?;
        let owner = SelectedStaticSiteOwner::new(
            edge.clone(),
            target.function,
            std::sync::Arc::clone(&body.definition_owner),
            dependencies
                .iter()
                .map(|lease| (lease.function(), std::sync::Arc::clone(lease.owner_arc())))
                .collect(),
        )?;
        add_static_edge(&mut edges, &mut owners, &mut by_site, edge, owner)?;
        let captured = body.selected_static_capture.as_ref();
        if let Some(captured) = captured {
            if !Rc::ptr_eq(captured.product(), &target.product) {
                return Err("selected static target changed its captured product");
            }
            if captured.dependencies().len() != captured.site_owners().len() {
                return Err("selected static target lacks transitive definition owners");
            }
            for (inherited, owner) in captured.dependencies().iter().zip(captured.site_owners()) {
                add_static_edge(
                    &mut edges,
                    &mut owners,
                    &mut by_site,
                    inherited.clone(),
                    owner.clone(),
                )?;
            }
            for inherited in captured.linked_native() {
                add_linked_site(&mut linked, &mut linked_by_site, inherited.clone())?;
            }
        } else if !target.product.static_imports().is_empty()
            || !target.product.linked_native_imports().is_empty()
        {
            return Err("selected static target lacks its transitive captured imports");
        }
    }
    SelectedStaticCapture::admit_with_linked(product, edges, linked)?.with_site_owners(owners)
}

fn add_linked_site(
    linked: &mut Vec<SelectedLinkedNativeSite>,
    by_site: &mut HashMap<(usize, ExecutableStaticCaller, GraphStaticReferenceSite), usize>,
    site: SelectedLinkedNativeSite,
) -> Result<(), &'static str> {
    let import = site.import();
    let key = (
        Rc::as_ptr(site.caller()) as usize,
        import.caller,
        import.site,
    );
    if let Some(existing) = by_site.get(&key) {
        if !linked[*existing].same_target(&site) {
            return Err("selected linked-native site has contradictory loaded code");
        }
    } else {
        by_site.insert(key, linked.len());
        linked.push(site);
    }
    Ok(())
}

struct SelectionInputs<'a> {
    inherited: &'a FunctionRegistry,
    included_current_ids: &'a HashSet<runmat_hir::FunctionId>,
    partitioned_reused_ids: &'a HashSet<runmat_hir::FunctionId>,
    preselected: &'a HashMap<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>,
    linked_native: &'a HashMap<
        runmat_hir::FunctionId,
        runmat_runtime::user_functions::SelectedExternalSourceTarget,
    >,
}

pub(in crate::session) fn selected_static_imports(
    unit: &crate::ExecutableUnit,
    inherited: &FunctionRegistry,
    included_current_ids: &HashSet<runmat_hir::FunctionId>,
    partitioned_reused_ids: &HashSet<runmat_hir::FunctionId>,
) -> Result<SelectedStaticImports, String> {
    selected_static_imports_with_preselected(
        unit,
        inherited,
        included_current_ids,
        partitioned_reused_ids,
        &HashMap::new(),
    )
}

/// Only the eval compiler may supply invocation-local aliases. Their exact
/// binding/view was selected by the canonical no-invoke loader before HIR;
/// aliases are never inserted into the durable registry name table.
pub(in crate::session) fn selected_static_imports_with_preselected(
    unit: &crate::ExecutableUnit,
    inherited: &FunctionRegistry,
    included_current_ids: &HashSet<runmat_hir::FunctionId>,
    partitioned_reused_ids: &HashSet<runmat_hir::FunctionId>,
    preselected: &HashMap<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>,
) -> Result<SelectedStaticImports, String> {
    selected_static_imports_with_preselected_and_linked(
        unit,
        inherited,
        included_current_ids,
        partitioned_reused_ids,
        preselected,
        &HashMap::new(),
    )
}

pub(in crate::session) fn selected_static_imports_with_preselected_and_linked(
    unit: &crate::ExecutableUnit,
    inherited: &FunctionRegistry,
    included_current_ids: &HashSet<runmat_hir::FunctionId>,
    partitioned_reused_ids: &HashSet<runmat_hir::FunctionId>,
    preselected: &HashMap<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>,
    linked_native: &HashMap<
        runmat_hir::FunctionId,
        runmat_runtime::user_functions::SelectedExternalSourceTarget,
    >,
) -> Result<SelectedStaticImports, String> {
    selected_static_imports_with_preselected_mode(
        unit,
        inherited,
        included_current_ids,
        partitioned_reused_ids,
        preselected,
        linked_native,
        true,
    )
}

/// The eval script is already executing when its closure is issued. Its
/// instructions are not part of the function-only selected product and must
/// not contribute imports to that product's sealed static section.
pub(in crate::session) fn selected_pending_closure_imports_with_preselected_and_linked(
    unit: &crate::ExecutableUnit,
    inherited: &FunctionRegistry,
    function: runmat_hir::FunctionId,
    partitioned_siblings: &HashSet<runmat_hir::FunctionId>,
    preselected: &HashMap<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>,
    linked_native: &HashMap<
        runmat_hir::FunctionId,
        runmat_runtime::user_functions::SelectedExternalSourceTarget,
    >,
) -> Result<SelectedStaticImports, String> {
    if partitioned_siblings
        .iter()
        .any(|function| !preselected.contains_key(function))
    {
        return Err("pending Local dependency lacks its exact selected binding".into());
    }
    selected_static_imports_with_preselected_mode(
        unit,
        inherited,
        &HashSet::from([function]),
        partitioned_siblings,
        preselected,
        linked_native,
        false,
    )
}

fn selected_static_imports_with_preselected_mode(
    unit: &crate::ExecutableUnit,
    inherited: &FunctionRegistry,
    included_current_ids: &HashSet<runmat_hir::FunctionId>,
    partitioned_reused_ids: &HashSet<runmat_hir::FunctionId>,
    preselected: &HashMap<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>,
    linked_native: &HashMap<
        runmat_hir::FunctionId,
        runmat_runtime::user_functions::SelectedExternalSourceTarget,
    >,
    include_script: bool,
) -> Result<SelectedStaticImports, String> {
    if !included_current_ids
        .iter()
        .all(|id| unit.functions().functions.contains_key(id))
    {
        return Err("current definition is absent from the compiled unit".into());
    }
    if !partitioned_reused_ids
        .iter()
        .all(|id| unit.functions().functions.contains_key(id) && !included_current_ids.contains(id))
    {
        return Err("partitioned definition is not a proved excluded compiled body".into());
    }
    let mut selected = SelectedStaticImports {
        imports: Vec::new(),
        targets: Vec::new(),
        linked_native: Vec::new(),
    };
    let mut selected_views =
        HashMap::<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>::new();
    let inputs = SelectionInputs {
        inherited,
        included_current_ids,
        partitioned_reused_ids,
        preselected,
        linked_native,
    };
    if include_script {
        collect(
            Some(ExecutableStaticCaller::Script),
            &unit.bytecode().instructions,
            &inputs,
            &mut selected_views,
            &mut selected,
        )?;
    }
    for (id, body) in &unit.functions().functions {
        if !included_current_ids.contains(id) {
            // Non-current bodies belong to their separately selected V3.
            continue;
        }
        collect(
            body.binding_origin.map(ExecutableStaticCaller::Function),
            &body.instructions,
            &inputs,
            &mut selected_views,
            &mut selected,
        )?;
    }
    // FunctionRegistry bodies are hash-indexed; the wire section and the
    // invocation-local target list must have one stable site order.
    selected.targets.sort_by_key(|target| {
        let caller = match target.import.caller {
            ExecutableStaticCaller::Script => None,
            ExecutableStaticCaller::Function(origin) => Some((origin.program, origin.function.0)),
        };
        (caller, target.import.site)
    });
    selected.imports = selected
        .targets
        .iter()
        .map(|target| target.import)
        .collect();
    selected.linked_native.sort_by_key(|target| {
        let caller = match target.import.caller {
            ExecutableStaticCaller::Script => None,
            ExecutableStaticCaller::Function(origin) => Some((origin.program, origin.function.0)),
        };
        (caller, target.import.site)
    });
    Ok(selected)
}

fn collect(
    caller: Option<ExecutableStaticCaller>,
    instructions: &[runmat_vm::Instr],
    inputs: &SelectionInputs<'_>,
    selected_views: &mut HashMap<runmat_hir::FunctionId, (BoundFunctionBinding, FunctionRegistry)>,
    selected_imports: &mut SelectedStaticImports,
) -> Result<(), String> {
    for (pc, instruction) in instructions.iter().enumerate() {
        for (site, function, kind) in instruction_reference_sites(instruction, pc)? {
            let import_kind = match kind {
                FunctionReferenceKind::External => ExecutableStaticImportKind::ExplicitExternal,
                FunctionReferenceKind::Local if inputs.included_current_ids.contains(&function) => {
                    continue;
                }
                FunctionReferenceKind::Local
                    if inputs.partitioned_reused_ids.contains(&function) =>
                {
                    ExecutableStaticImportKind::PartitionedDefinition
                }
                FunctionReferenceKind::Local => {
                    return Err(
                        "local reference is outside its proved current or reused partition".into(),
                    );
                }
            };
            let caller = caller.ok_or("selected static caller lacks its portable origin")?;
            if let Some(target) = inputs.linked_native.get(&function) {
                if kind != FunctionReferenceKind::External {
                    return Err("linked native target is not an external reference".into());
                }
                let import = target
                    .linked_import(caller, site)
                    .ok_or("linked native target source or loaded code has retired")?;
                selected_imports
                    .linked_native
                    .push(SelectedLinkedNativeTarget {
                        import,
                        target: target.clone(),
                    });
                continue;
            }
            // External sites were emitted from the pre-merge inherited
            // registry. Partitioned sites have a separately proved reused-ID
            // exclusion, and must also prove the exact inherited body.
            // Neither may select a projected eval-local body merely because
            // its numeric ID collides.
            let selected_pair = if kind == FunctionReferenceKind::External
                || (kind == FunctionReferenceKind::Local
                    && inputs.partitioned_reused_ids.contains(&function))
            {
                inputs.preselected.get(&function)
            } else {
                None
            };
            let selected_function = selected_pair
                .map(|(binding, _)| runmat_hir::FunctionId(binding.function))
                .unwrap_or(function);
            let selected = match selected_pair {
                Some((binding, view)) => {
                    let lease = binding
                        .authority
                        .local_definition()
                        .ok_or("preselected static target has no exact definition lease")?;
                    view.get_owned(selected_function, lease)
                }
                None => inputs.inherited.get_current(function),
            }
            .ok_or("typed static import lacks its selected inherited definition")?;
            let origin = selected
                .binding_origin
                .ok_or("typed static target lacks its portable origin")?;
            let source_owner = selected
                .selected_source_owner
                .as_ref()
                .ok_or("typed static target lacks its selected source owner")?;
            let source = selected
                .source_id
                .and_then(BytecodeSourceRef::runtime)
                .ok_or("typed static target lacks an issued runtime source")?;
            if !source_owner.is_live() || !source_owner.owns_source(source) {
                return Err("typed static target source issuance has retired".into());
            }
            let target_product = selected
                .selected_executable_product
                .as_ref()
                .ok_or("typed static target lacks its exact selected V3 product")?;
            if !target_product.source_owner().same_issuance(source_owner) {
                return Err("typed static target changed its selected product".into());
            }
            let target_body = target_product
                .body_digest(origin)
                .ok_or("typed static target is absent from its selected product")?;
            let import = ExecutableStaticImport {
                caller,
                site,
                target: origin,
                kind: import_kind,
                target_content: target_product.content_digest(),
                target_body,
            };
            let definition =
                FunctionDefinitionOwner::issue(&selected.definition_owner, selected_function.0);
            let (binding, view) = match selected_views.get(&function) {
                Some((binding, view)) => (binding.clone(), view.clone()),
                None => {
                    let (binding, view) = match selected_pair {
                        Some((binding, view)) => (binding.clone(), view.clone()),
                        None => {
                            let binding = inputs
                                .inherited
                                .exact_bound_binding(function)
                                .ok_or("typed static target has no exact bound dependency graph")?;
                            let view = inputs
                                .inherited
                                .owned_view(function, &binding.authority)
                                .ok_or("typed static target has no exact owned view")?;
                            (binding, view)
                        }
                    };
                    selected_views.insert(function, (binding.clone(), view.clone()));
                    (binding, view)
                }
            };
            if binding
                .authority
                .local_definition()
                .is_none_or(|root| root != &definition)
            {
                return Err("typed static target changed its definition lease".into());
            }
            selected_imports.imports.push(import);
            selected_imports.targets.push(SelectedStaticTarget {
                import,
                function: selected_function,
                definition,
                binding,
                view,
                product: Rc::clone(target_product),
                source_owner: source_owner.clone(),
            });
        }
    }
    Ok(())
}
