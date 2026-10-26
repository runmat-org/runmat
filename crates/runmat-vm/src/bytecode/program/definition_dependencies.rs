use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::executable::{ExecutableStaticCaller, ExecutableStaticImportKind};
use runmat_hir::FunctionId;
use runmat_value::FunctionDefinitionOwner;

use super::{
    function_remap::{self, FunctionReferenceKind},
    super_constructor_dependencies, FunctionRegistry, SuperConstructorDependencyError,
};

pub(super) fn owners(
    registry: &FunctionRegistry,
    function: FunctionId,
) -> Result<Vec<(FunctionId, Arc<FunctionDefinitionOwner>)>, SuperConstructorDependencyError> {
    owners_with_capture(registry, function, None)
}

/// Bootstrap local owners only after the complete immutable import graph has
/// been validated. This view does not issue a callable or any counted lease.
pub(super) fn owners_with_capture(
    registry: &FunctionRegistry,
    function: FunctionId,
    selected_capture: Option<&super::SelectedStaticCapture>,
) -> Result<Vec<(FunctionId, Arc<FunctionDefinitionOwner>)>, SuperConstructorDependencyError> {
    let mut super_constructors = None;
    let mut visited = HashSet::new();
    let mut owners = Vec::new();
    let mut pending = Vec::new();
    pending.push(function);
    while let Some(function) = pending.pop() {
        if !visited.insert(function) {
            continue;
        }
        let body = registry
            .get(function)
            .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
        owners.push((function, Arc::clone(&body.definition_owner)));
        // Every selected source, including a separately admitted dependency,
        // owns its local definitions and class declaration. localfunctions()
        // may issue those handles after the file is retired even when no
        // static call edge references a particular local.
        if let Some(source) = body.source_id {
            pending.extend(registry.functions_for_source_ref(source).iter().copied());
        }
        if let Some(parent) = body.lexical_parent {
            pending.push(parent);
        }
        // A selected import is keyed by the compiler's instruction site.
        // Its counted target lease is issued separately with the bound
        // callable; pushing its numeric ID here could retain a newer body
        // with the same session-local number instead.
        let mut imports = HashMap::new();
        if let Some(capture) = selected_capture.or(body.selected_static_capture.as_deref()) {
            let product = body
                .selected_executable_product
                .as_ref()
                .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
            if !Rc::ptr_eq(product, capture.product()) {
                return Err(SuperConstructorDependencyError::MissingOrigin);
            }
            for edge in capture.dependencies() {
                if !Rc::ptr_eq(edge.caller(), product) {
                    continue;
                }
                let import = edge.import();
                if import.caller
                    == ExecutableStaticCaller::Function(
                        body.binding_origin
                            .ok_or(SuperConstructorDependencyError::MissingOrigin)?,
                    )
                    && imports.insert(import.site, import).is_some()
                {
                    return Err(SuperConstructorDependencyError::MissingOrigin);
                }
            }
            for site in capture.linked_native() {
                if !Rc::ptr_eq(site.caller(), product) {
                    continue;
                }
                let import = site.import();
                if import.caller
                    == ExecutableStaticCaller::Function(
                        body.binding_origin
                            .ok_or(SuperConstructorDependencyError::MissingOrigin)?,
                    )
                    && imports
                        .insert(import.site, import.fingerprint_projection())
                        .is_some()
                {
                    return Err(SuperConstructorDependencyError::MissingOrigin);
                }
            }
        } else if let (Some(product), Some(origin)) = (
            body.selected_executable_product.as_ref(),
            body.binding_origin,
        ) {
            if product
                .static_imports()
                .iter()
                .any(|import| import.caller == ExecutableStaticCaller::Function(origin))
                || product
                    .linked_native_imports()
                    .iter()
                    .any(|import| import.caller == ExecutableStaticCaller::Function(origin))
            {
                return Err(SuperConstructorDependencyError::MissingOrigin);
            }
        }
        let mut observed = HashSet::new();
        for (pc, instruction) in body.instructions.iter().enumerate() {
            for (site, dependency, kind) in
                function_remap::instruction_reference_sites(instruction, pc)
                    .map_err(|_| SuperConstructorDependencyError::MissingOrigin)?
            {
                if let Some(import) = imports.get(&site) {
                    let matches_provenance = matches!(
                        (kind, import.kind),
                        (
                            FunctionReferenceKind::External,
                            ExecutableStaticImportKind::ExplicitExternal
                        ) | (
                            FunctionReferenceKind::External,
                            ExecutableStaticImportKind::LinkedNative
                        ) | (
                            FunctionReferenceKind::Local,
                            ExecutableStaticImportKind::PartitionedDefinition
                        )
                    );
                    if !matches_provenance {
                        return Err(SuperConstructorDependencyError::MissingOrigin);
                    }
                    observed.insert(site);
                } else if kind == FunctionReferenceKind::External {
                    return Err(SuperConstructorDependencyError::MissingOrigin);
                } else {
                    pending.push(dependency);
                }
            }
            if let Some((requesting_class, super_class)) =
                super_constructor_dependencies::instruction_classes(instruction)
            {
                let index = super_constructors.get_or_insert_with(|| {
                    super_constructor_dependencies::SuperConstructorIndex::new(registry)
                });
                if body.class_method_owner.as_ref().is_none_or(|owner| {
                    owner.is_static || &owner.declaring_class != requesting_class
                }) {
                    return Err(SuperConstructorDependencyError::CallingOwnerMismatch);
                }
                if let Some(target) = index.target(super_class, body.binding_origin)? {
                    pending.push(target);
                } else if let Some(declaration) =
                    index.declaration(super_class, body.binding_origin)?
                {
                    pending.push(declaration);
                }
            }
        }
        if observed.len() != imports.len() {
            return Err(SuperConstructorDependencyError::MissingOrigin);
        }
    }
    Ok(owners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{FunctionBytecode, Instr};
    use std::collections::HashMap;

    #[test]
    fn static_helper_and_bound_closure_are_retained_transitively() {
        let first = FunctionBytecode {
            function: FunctionId(1),
            instructions: vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)],
            ..Default::default()
        };
        let second = FunctionBytecode {
            function: FunctionId(2),
            instructions: vec![Instr::CreateSemanticClosure(
                FunctionId(3),
                "nested".into(),
                0,
                runmat_types::FunctionSourceRole::Anonymous,
            )],
            ..Default::default()
        };
        let third = FunctionBytecode {
            function: FunctionId(3),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), first),
            (FunctionId(2), second),
            (FunctionId(3), third),
        ]));
        assert_eq!(
            registry
                .definition_dependency_owners(FunctionId(1))
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn unresolved_static_callee_cannot_issue_incomplete_bundle() {
        let caller = FunctionBytecode {
            function: FunctionId(1),
            instructions: vec![Instr::CallSemanticFunctionMulti(FunctionId(2), 0, 1)],
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(1), caller)]));
        assert!(registry
            .definition_dependency_owners(FunctionId(1))
            .is_none());
    }

    #[test]
    fn nested_definition_retains_its_parent_even_without_an_instruction_edge() {
        let parent = FunctionBytecode {
            function: FunctionId(1),
            ..Default::default()
        };
        let nested = FunctionBytecode {
            function: FunctionId(2),
            lexical_parent: Some(FunctionId(1)),
            source_role: runmat_types::FunctionSourceRole::Nested,
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), parent),
            (FunctionId(2), nested.clone()),
        ]));
        let owners = registry
            .definition_dependency_owners(FunctionId(2))
            .expect("complete lexical graph");
        assert_eq!(owners.len(), 2);

        let incomplete = FunctionRegistry::new(HashMap::from([(FunctionId(2), nested)]));
        assert!(incomplete
            .definition_dependency_owners(FunctionId(2))
            .is_none());
    }

    #[test]
    fn long_dependency_chain_and_cycle_are_bounded_by_the_worklist() {
        const COUNT: usize = 4096;
        let functions = (0..COUNT)
            .map(|index| {
                let dependency = (index + 1) % COUNT;
                (
                    FunctionId(index),
                    FunctionBytecode {
                        function: FunctionId(index),
                        instructions: vec![Instr::CallSemanticFunctionMulti(
                            FunctionId(dependency),
                            0,
                            1,
                        )],
                        ..Default::default()
                    },
                )
            })
            .collect();
        let registry = FunctionRegistry::new(functions);
        assert_eq!(
            registry
                .definition_dependency_owners(FunctionId(0))
                .unwrap()
                .len(),
            COUNT
        );
    }
}
