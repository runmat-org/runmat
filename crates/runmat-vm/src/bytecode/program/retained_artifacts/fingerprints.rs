use std::collections::{HashMap, HashSet};

use runmat_execution::executable::{
    ExecutableStaticCaller, ExecutableStaticImport, ExecutableStaticImportKind,
    SelectedBodySemanticDigest,
};
use runmat_execution::{ExecutableSourceMapV2, FunctionBindingOrigin};
use runmat_hir::FunctionId;

use super::{
    function_remap, FunctionRegistry, RetainedCallableStageError, SelectedFingerprintInvariant,
};

fn selected_graph(
    function: FunctionId,
    invariant: SelectedFingerprintInvariant,
) -> RetainedCallableStageError {
    RetainedCallableStageError::SelectedFingerprintGraph {
        function,
        invariant,
    }
}

#[path = "fingerprints/source_identity.rs"]
mod source_identity;
use source_identity::{project_definition_source, DefinitionSourceDomain};

/// Normalize session-local numbers to portable function ordinals before
/// comparing bodies from independently packaged executable units.
pub(super) fn normalized(
    registry: &FunctionRegistry,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    normalized_in(registry, DefinitionSourceDomain::Runtime, false)
}

/// Fingerprint the exact selected execution view, including leased static
/// definitions. The temporary registry is a projection of already-selected
/// bodies, not a new source or definition authority.
pub(super) fn normalized_selected(
    registry: &FunctionRegistry,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    let bodies = selected_bodies(registry)?;
    normalized_in(
        &FunctionRegistry::new(bodies),
        DefinitionSourceDomain::Runtime,
        true,
    )
}

/// Normalize one leased body without collapsing a second equal-origin body
/// issued by a different selected unit. The caller must retain the exact
/// definition and source owner while comparing this result with admitted V3.
pub(super) fn normalized_selected_body(
    registry: &FunctionRegistry,
    function: FunctionId,
) -> Result<(FunctionBindingOrigin, String), RetainedCallableStageError> {
    normalized_selected_bodies(registry, &[function])?
        .remove(&function)
        .ok_or(RetainedCallableStageError::IncompleteDefinitionGraph)
}

pub(super) fn normalized_selected_bodies(
    registry: &FunctionRegistry,
    functions: &[FunctionId],
) -> Result<HashMap<FunctionId, (FunctionBindingOrigin, String)>, RetainedCallableStageError> {
    let selected = FunctionRegistry::new(selected_bodies(registry)?);
    let mut fingerprints = HashMap::with_capacity(functions.len());
    for &function in functions {
        let body = selected.functions.get(&function).ok_or_else(|| {
            selected_graph(function, SelectedFingerprintInvariant::MissingSelectedBody)
        })?;
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin { function })?;
        let fingerprint = fingerprint_body(&selected, body, DefinitionSourceDomain::Runtime, true)?;
        if fingerprints
            .insert(function, (origin, fingerprint))
            .is_some()
        {
            return Err(selected_graph(
                function,
                SelectedFingerprintInvariant::RepeatedRequestedBody,
            ));
        }
    }
    Ok(fingerprints)
}

fn selected_bodies(
    registry: &FunctionRegistry,
) -> Result<HashMap<FunctionId, super::super::FunctionBytecode>, RetainedCallableStageError> {
    let mut bodies = HashMap::new();
    for body in registry.selected_execution_bodies() {
        if bodies.insert(body.function, body.clone()).is_some() {
            return Err(selected_graph(
                body.function,
                SelectedFingerprintInvariant::DuplicateSelectedBody,
            ));
        }
    }
    Ok(bodies)
}

pub(super) fn normalized_admitted(
    registry: &FunctionRegistry,
    source_map: &ExecutableSourceMapV2,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    normalized_in(
        registry,
        DefinitionSourceDomain::Admitted(source_map),
        false,
    )
}

/// Normalize one admitted unit with its sealed, site-qualified external
/// references. Imported targets are identities in other selected products,
/// never members inferred from this registry's numeric ID space.
pub(super) fn normalized_admitted_with_imports(
    registry: &FunctionRegistry,
    source_map: &ExecutableSourceMapV2,
    imports: &[ExecutableStaticImport],
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    if imports.is_empty() {
        return normalized_admitted(registry, source_map);
    }
    let mut fingerprints = HashMap::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        let fingerprint = fingerprint_body_with_imports(
            registry,
            body,
            DefinitionSourceDomain::Admitted(source_map),
            false,
            imports,
            true,
        )?;
        if fingerprints.insert(origin, fingerprint).is_some() {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    Ok(fingerprints)
}

pub(super) fn normalized_selected_bodies_with_imports(
    registry: &FunctionRegistry,
    functions: &[FunctionId],
    imports: &[ExecutableStaticImport],
) -> Result<HashMap<FunctionId, (FunctionBindingOrigin, String)>, RetainedCallableStageError> {
    if imports.is_empty() {
        return normalized_selected_bodies(registry, functions);
    }
    let selected = FunctionRegistry::new(selected_bodies(registry)?);
    let mut fingerprints = HashMap::with_capacity(functions.len());
    for &function in functions {
        let body = selected.functions.get(&function).ok_or_else(|| {
            selected_graph(function, SelectedFingerprintInvariant::MissingSelectedBody)
        })?;
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin { function })?;
        let fingerprint = fingerprint_body_with_imports(
            &selected,
            body,
            DefinitionSourceDomain::Runtime,
            true,
            imports,
            true,
        )?;
        if fingerprints
            .insert(function, (origin, fingerprint))
            .is_some()
        {
            return Err(selected_graph(
                function,
                SelectedFingerprintInvariant::RepeatedRequestedBody,
            ));
        }
    }
    Ok(fingerprints)
}

/// One selected overlay projection serves bodies from several exact products.
/// Each body uses only its own sealed site imports; equal origins in other
/// units cannot silently substitute a different selected product.
pub(super) fn normalized_selected_bodies_by_product(
    registry: &FunctionRegistry,
    inputs: &[(FunctionId, &[ExecutableStaticImport])],
) -> Result<HashMap<FunctionId, (FunctionBindingOrigin, String)>, RetainedCallableStageError> {
    let selected = FunctionRegistry::new(selected_bodies(registry)?);
    let mut fingerprints = HashMap::with_capacity(inputs.len());
    for &(function, imports) in inputs {
        let body = selected
            .functions
            .get(&function)
            .ok_or(RetainedCallableStageError::IncompleteDefinitionGraph)?;
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin { function })?;
        let fingerprint = fingerprint_body_with_imports(
            &selected,
            body,
            DefinitionSourceDomain::Runtime,
            true,
            imports,
            true,
        )?;
        if fingerprints
            .insert(function, (origin, fingerprint))
            .is_some()
        {
            return Err(RetainedCallableStageError::IncompleteDefinitionGraph);
        }
    }
    Ok(fingerprints)
}

pub(super) fn admitted_semantic_body_digests(
    registry: &FunctionRegistry,
    source_map: &ExecutableSourceMapV2,
    imports: &[ExecutableStaticImport],
) -> Result<HashMap<FunctionBindingOrigin, SelectedBodySemanticDigest>, RetainedCallableStageError>
{
    let mut digests = HashMap::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        let fingerprint = fingerprint_body_with_imports(
            registry,
            body,
            DefinitionSourceDomain::Admitted(source_map),
            false,
            imports,
            false,
        )?;
        let digest = SelectedBodySemanticDigest::from_normalized_fingerprint(&fingerprint)
            .map_err(|_| RetainedCallableStageError::IncompleteDefinitionGraph)?;
        if digests.insert(origin, digest).is_some() {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    Ok(digests)
}

fn fingerprint_body_with_imports(
    registry: &FunctionRegistry,
    body: &super::super::FunctionBytecode,
    source_domain: DefinitionSourceDomain<'_>,
    selected: bool,
    imports: &[ExecutableStaticImport],
    include_product_links: bool,
) -> Result<String, RetainedCallableStageError> {
    let origin = body
        .binding_origin
        .ok_or(RetainedCallableStageError::MissingOrigin {
            function: body.function,
        })?;
    let caller = ExecutableStaticCaller::Function(origin);
    let body_imports = imports
        .iter()
        .filter(|import| import.caller == caller)
        .collect::<Vec<_>>();
    if body_imports.is_empty() {
        return fingerprint_body(registry, body, source_domain, selected);
    }
    let mut local_ids = HashSet::from([body.function]);
    if let Some(parent) = body.lexical_parent {
        let parent_body = registry.functions.get(&parent).ok_or_else(|| {
            selected_graph(
                body.function,
                SelectedFingerprintInvariant::MissingLexicalParent(parent),
            )
        })?;
        let parent_origin = parent_body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin { function: parent })?;
        if body.portable_lexical_parent != Some(parent_origin.function)
            || origin.program != parent_origin.program
        {
            return Err(selected_graph(
                body.function,
                SelectedFingerprintInvariant::MismatchedLexicalParent(parent),
            ));
        }
        local_ids.insert(parent);
    } else if body.portable_lexical_parent.is_some() {
        return Err(selected_graph(
            body.function,
            SelectedFingerprintInvariant::UnexpectedPortableLexicalParent,
        ));
    }
    let mut imported_sites = HashMap::new();
    for import in &body_imports {
        if imported_sites.insert(import.site, *import).is_some() {
            return Err(selected_graph(
                body.function,
                SelectedFingerprintInvariant::DuplicateImportSite(import.site),
            ));
        }
    }
    let mut observed_imports = HashSet::new();
    for (pc, instruction) in body.instructions.iter().enumerate() {
        for (site, function, kind) in function_remap::instruction_reference_sites(instruction, pc)
            .map_err(|_| {
            selected_graph(
                body.function,
                SelectedFingerprintInvariant::InvalidReferenceSites(pc),
            )
        })? {
            match imported_sites.get(&site) {
                Some(import)
                    if (kind == function_remap::FunctionReferenceKind::External
                        && matches!(
                            import.kind,
                            ExecutableStaticImportKind::ExplicitExternal
                                | ExecutableStaticImportKind::LinkedNative
                        ))
                        || (kind == function_remap::FunctionReferenceKind::Local
                            && import.kind
                                == ExecutableStaticImportKind::PartitionedDefinition) =>
                {
                    observed_imports.insert(site);
                }
                Some(_) => {
                    return Err(selected_graph(
                        body.function,
                        SelectedFingerprintInvariant::ImportKindMismatch(site),
                    ))
                }
                None if kind == function_remap::FunctionReferenceKind::Local => {
                    local_ids.insert(function);
                }
                None => {
                    return Err(selected_graph(
                        body.function,
                        SelectedFingerprintInvariant::UnimportedExternalReference(site),
                    ))
                }
            }
        }
    }
    if let Some(site) = imported_sites
        .keys()
        .filter(|site| !observed_imports.contains(*site))
        .min()
    {
        return Err(selected_graph(
            body.function,
            SelectedFingerprintInvariant::UnobservedImportSite(*site),
        ));
    }
    if let Some(local) = local_ids
        .iter()
        .filter(|function| !registry.functions.contains_key(*function))
        .min()
    {
        return Err(selected_graph(
            body.function,
            SelectedFingerprintInvariant::MissingLocalDefinition(*local),
        ));
    }
    let mut local_ids = local_ids.into_iter().collect::<Vec<_>>();
    local_ids.sort_by_key(|function| {
        let origin = registry.functions[function]
            .binding_origin
            .expect("local origins were validated");
        (origin.program, origin.function)
    });
    let local_origins = local_ids
        .iter()
        .map(|function| {
            registry.functions[function]
                .binding_origin
                .expect("local origins were validated")
        })
        .collect::<Vec<_>>();
    if local_origins.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(RetainedCallableStageError::DuplicateOrigin);
    }
    let remap = local_ids
        .iter()
        .enumerate()
        .map(|(ordinal, function)| (*function, FunctionId(ordinal)))
        .collect::<HashMap<_, _>>();
    let mut normalized = body.clone();
    function_remap::remap_closed_function(&mut normalized, &remap).map_err(|_| {
        selected_graph(
            body.function,
            SelectedFingerprintInvariant::ClosedFunctionRemap,
        )
    })?;
    let replacements = body_imports
        .iter()
        .enumerate()
        .map(|(ordinal, import)| (import.site, FunctionId(local_ids.len() + ordinal)))
        .collect::<HashMap<_, _>>();
    function_remap::remap_reference_sites(&mut normalized.instructions, &replacements).map_err(
        |_| {
            selected_graph(
                body.function,
                SelectedFingerprintInvariant::ImportedSiteRemap,
            )
        },
    )?;
    #[cfg(feature = "native-accel")]
    {
        normalized.reseal_fusion_stage_after_owned_remap();
        normalized.validate_fusion_stage_projection().map_err(|_| {
            selected_graph(
                body.function,
                SelectedFingerprintInvariant::FusionStageProjection,
            )
        })?;
    }
    let selected_domain = if selected {
        DefinitionSourceDomain::Selected(
            body.selected_source_owner
                .as_ref()
                .ok_or(RetainedCallableStageError::ForeignSourceOwner)?,
        )
    } else {
        source_domain
    };
    let source = project_definition_source(normalized.source_id, selected_domain)?;
    normalized.source_id = None;
    let definition = super::super::semantic_definition_fingerprint(&normalized)?;
    if include_product_links {
        let imported_links = body_imports
            .iter()
            .map(|import| {
                (
                    import.site,
                    import.target,
                    import.kind,
                    import.target_content,
                    import.target_body,
                )
            })
            .collect::<Vec<_>>();
        Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
            definition,
            source,
            local_origins,
            imported_links,
        ))?)
        .to_string())
    } else {
        let imported_sites = body_imports
            .iter()
            .map(|import| (import.site, import.target, import.kind))
            .collect::<Vec<_>>();
        Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
            definition,
            source,
            local_origins,
            imported_sites,
        ))?)
        .to_string())
    }
}

fn normalized_in(
    registry: &FunctionRegistry,
    source_domain: DefinitionSourceDomain<'_>,
    selected: bool,
) -> Result<HashMap<FunctionBindingOrigin, String>, RetainedCallableStageError> {
    let mut origins = HashSet::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        if !origins.insert(origin) {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    let mut fingerprints = HashMap::with_capacity(registry.functions.len());
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin {
                function: body.function,
            })?;
        let fingerprint = fingerprint_body(registry, body, source_domain, selected)?;
        if fingerprints.insert(origin, fingerprint).is_some() {
            return Err(RetainedCallableStageError::DuplicateOrigin);
        }
    }
    Ok(fingerprints)
}

fn fingerprint_body(
    registry: &FunctionRegistry,
    body: &super::super::FunctionBytecode,
    source_domain: DefinitionSourceDomain<'_>,
    selected: bool,
) -> Result<String, RetainedCallableStageError> {
    let (portable_ids, referenced_origins) = direct_origin_ids(registry, body.function)?;
    if selected
        && referenced_origins.iter().collect::<HashSet<_>>().len() != referenced_origins.len()
    {
        return Err(RetainedCallableStageError::DuplicateOrigin);
    }
    let mut normalized = body.clone();
    function_remap::remap_closed_function(&mut normalized, &portable_ids).map_err(|_| {
        selected_graph(
            body.function,
            SelectedFingerprintInvariant::ClosedFunctionRemap,
        )
    })?;
    let selected_domain = if selected {
        DefinitionSourceDomain::Selected(
            body.selected_source_owner
                .as_ref()
                .ok_or(RetainedCallableStageError::ForeignSourceOwner)?,
        )
    } else {
        source_domain
    };
    let source = project_definition_source(normalized.source_id, selected_domain)?;
    // An ordinal is only a position in this admitted envelope. The exact
    // validated source record supplies immutable source identity below.
    normalized.source_id = None;
    let definition = super::super::semantic_definition_fingerprint(&normalized)?;
    Ok(runmat_execution::Digest::sha256(serde_json::to_vec(&(
        definition,
        source,
        referenced_origins,
    ))?)
    .to_string())
}

/// Project one immutable body and its direct semantic references into a dense
/// origin-ordered identity domain. Each referenced body's own fingerprint is
/// checked independently, so unrelated or merely transitive definitions must
/// not make this body's fingerprint quadratic or packaging-dependent.
fn direct_origin_ids(
    registry: &FunctionRegistry,
    root: FunctionId,
) -> Result<(HashMap<FunctionId, FunctionId>, Vec<FunctionBindingOrigin>), RetainedCallableStageError>
{
    let body = registry
        .functions
        .get(&root)
        .ok_or_else(|| selected_graph(root, SelectedFingerprintInvariant::MissingRootBody))?;
    let mut referenced = HashSet::from([root]);
    if let Some(parent) = body.lexical_parent {
        let parent_body = registry.functions.get(&parent).ok_or_else(|| {
            selected_graph(
                root,
                SelectedFingerprintInvariant::MissingLexicalParent(parent),
            )
        })?;
        let child_origin = body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin { function: root })?;
        let parent_origin = parent_body
            .binding_origin
            .ok_or(RetainedCallableStageError::MissingOrigin { function: parent })?;
        if body.portable_lexical_parent != Some(parent_origin.function)
            || child_origin.program != parent_origin.program
        {
            return Err(selected_graph(
                root,
                SelectedFingerprintInvariant::MismatchedLexicalParent(parent),
            ));
        }
        referenced.insert(parent);
    } else if body.portable_lexical_parent.is_some() {
        return Err(selected_graph(
            root,
            SelectedFingerprintInvariant::UnexpectedPortableLexicalParent,
        ));
    }
    for instruction in &body.instructions {
        function_remap::visit_instruction(instruction, |dependency| {
            referenced.insert(dependency);
        });
    }
    if let Some(missing) = referenced
        .iter()
        .filter(|function| !registry.functions.contains_key(*function))
        .min()
    {
        return Err(selected_graph(
            root,
            SelectedFingerprintInvariant::MissingLocalDefinition(*missing),
        ));
    }
    let mut referenced = referenced.into_iter().collect::<Vec<_>>();
    referenced.sort_by_key(|function| {
        registry.functions[function]
            .binding_origin
            .map(|origin| (origin.program, origin.function))
    });
    let referenced_origins = referenced
        .iter()
        .map(|function| {
            registry.functions[function]
                .binding_origin
                .expect("all registry origins were validated")
        })
        .collect();
    let portable_ids = referenced
        .into_iter()
        .enumerate()
        .map(|(portable, function)| (function, FunctionId(portable)))
        .collect();
    Ok((portable_ids, referenced_origins))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::FunctionBytecode;
    use runmat_types::{FunctionSourceRole, ProgramFunctionId};

    #[test]
    fn nested_parent_is_a_direct_origin_and_must_match_its_portable_owner() {
        let program = runmat_execution::Digest::sha256("nested source");
        let parent = FunctionId(41);
        let nested = FunctionId(42);
        let origin = |function| FunctionBindingOrigin { program, function };
        let parent_body = FunctionBytecode {
            function: parent,
            portable_function: ProgramFunctionId(1),
            binding_origin: Some(origin(ProgramFunctionId(1))),
            ..Default::default()
        };
        let nested_body = FunctionBytecode {
            function: nested,
            portable_function: ProgramFunctionId(2),
            binding_origin: Some(origin(ProgramFunctionId(2))),
            source_role: FunctionSourceRole::Nested,
            lexical_parent: Some(parent),
            portable_lexical_parent: Some(ProgramFunctionId(1)),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (parent, parent_body),
            (nested, nested_body.clone()),
        ]));
        let (ids, origins) = direct_origin_ids(&registry, nested).expect("complete parent origin");
        assert!(ids.contains_key(&parent));
        assert!(origins.contains(&origin(ProgramFunctionId(1))));

        let mut wrong = nested_body;
        wrong.portable_lexical_parent = Some(ProgramFunctionId(9));
        let invalid = FunctionRegistry::new(HashMap::from([
            (parent, registry.functions[&parent].clone()),
            (nested, wrong),
        ]));
        assert!(matches!(
            direct_origin_ids(&invalid, nested),
            Err(RetainedCallableStageError::SelectedFingerprintGraph {
                function,
                invariant: SelectedFingerprintInvariant::MismatchedLexicalParent(found),
            }) if function == nested && found == parent
        ));
    }
}
