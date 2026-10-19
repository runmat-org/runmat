//! Typed constructor edges in an admitted bytecode registry.
//!
//! Super calls encode class identities rather than a numeric function ID.
//! Resolve a compiled constructor owner in the admitted registry. Prefer the
//! caller's product; a separately admitted parent must be the sole typed
//! owner of that class in the selected graph. A no-body parent instead needs
//! an admitted linked declaration or a typed intrinsic identity.

use runmat_execution::FunctionBindingOrigin;
use runmat_hir::FunctionId;
use runmat_runtime::context::{RetainedClassDeclaration, RetainedConstructorDeclaration};
use runmat_types::{ClassIdentity, ClassMethodOwner};

use super::{FunctionRegistry, Instr};

#[derive(Debug, Clone)]
pub struct RetainedSuperConstructorTarget {
    pub requesting_class: ClassIdentity,
    pub caller_origin: FunctionBindingOrigin,
    pub function: FunctionId,
    pub owner: ClassMethodOwner,
    pub origin: FunctionBindingOrigin,
    pub caller_declaration: RetainedClassDeclaration,
    pub target_declaration: RetainedClassDeclaration,
}

#[path = "super_constructor_dependencies/implicit.rs"]
mod implicit;
pub use implicit::RetainedImplicitSuperConstructorTarget;
#[path = "super_constructor_dependencies/index.rs"]
mod index;
pub(crate) use index::same_class_registration;
pub(super) use index::SuperConstructorIndex;

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum SuperConstructorDependencyError {
    #[error("super constructor call disagrees with its compiled class owner")]
    CallingOwnerMismatch,
    #[error("super constructor has multiple compiled definitions")]
    AmbiguousTarget,
    #[error("compiled super constructor lacks an authenticated origin")]
    MissingOrigin,
    #[error("compiled super constructor lacks a matching immutable class declaration")]
    MissingDeclaration,
    #[error("compiled super constructor has conflicting class declarations")]
    ConflictingDeclaration,
}

pub(crate) fn instruction_classes(instruction: &Instr) -> Option<(&ClassIdentity, &ClassIdentity)> {
    match instruction {
        Instr::CallSuperConstructorMulti {
            current_class,
            super_class,
            ..
        }
        | Instr::CallSuperConstructorExpandMultiOutput {
            current_class,
            super_class,
            ..
        } => Some((current_class, super_class)),
        _ => None,
    }
}

impl FunctionRegistry {
    /// Typed external parents required by this admitted unit but not supplied
    /// by one of its own constructor definitions. The source loader may join
    /// only exact already-admitted products for these classes; names alone do
    /// not identify a callable definition.
    pub fn unresolved_super_constructor_classes(
        &self,
    ) -> Result<Vec<ClassIdentity>, SuperConstructorDependencyError> {
        let index = SuperConstructorIndex::new(self);
        let mut unresolved = Vec::new();
        for body in self.functions.values() {
            for instruction in &body.instructions {
                let Some((requesting_class, super_class)) = instruction_classes(instruction) else {
                    continue;
                };
                if body.class_method_owner.as_ref().is_none_or(|owner| {
                    owner.is_static || &owner.declaring_class != requesting_class
                }) {
                    return Err(SuperConstructorDependencyError::CallingOwnerMismatch);
                }
                if index.target(super_class, body.binding_origin)?.is_none()
                    && index
                        .declaration(super_class, body.binding_origin)?
                        .is_none()
                    && !runmat_runtime::class_registry::has_intrinsic_implicit_constructor(
                        super_class,
                    )
                    && !unresolved.contains(super_class)
                {
                    unresolved.push(super_class.clone());
                }
            }
        }
        Ok(unresolved)
    }

    /// Super edges owned by one selected executable body. This avoids a
    /// same-class constructor in a different admitted product inheriting the
    /// first product's callable graph.
    pub fn retained_super_constructor_targets_for(
        &self,
        caller: FunctionId,
    ) -> Result<Vec<RetainedSuperConstructorTarget>, SuperConstructorDependencyError> {
        let body = self
            .get(caller)
            .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
        if !body
            .instructions
            .iter()
            .any(|instruction| instruction_classes(instruction).is_some())
        {
            return Ok(Vec::new());
        }
        let selected = selected_super_owner_ids(self, caller)?;
        let index = SuperConstructorIndex::from_ids(self, selected.iter().copied());
        retained_targets_for_body(self, body, &index, &selected)
    }

    /// Exact source identities for compiled superclass calls. This does not
    /// authorize a same-named function or invent a body for an external class.
    pub fn retained_super_constructor_targets(
        &self,
    ) -> Result<Vec<RetainedSuperConstructorTarget>, SuperConstructorDependencyError> {
        let index = SuperConstructorIndex::new(self);
        let mut targets = Vec::new();
        let selected = self.functions.keys().copied().collect::<Vec<_>>();
        for body in self.functions.values() {
            for target in retained_targets_for_body(self, body, &index, &selected)? {
                if !targets
                    .iter()
                    .any(|existing: &RetainedSuperConstructorTarget| {
                        existing.requesting_class == target.requesting_class
                            && existing.owner.declaring_class == target.owner.declaring_class
                            && existing.caller_origin == target.caller_origin
                    })
                {
                    targets.push(target);
                }
            }
        }
        Ok(targets)
    }
}

/// Preserve typed graph failures while using the canonical dependency scan.
fn selected_super_owner_ids(
    registry: &FunctionRegistry,
    caller: FunctionId,
) -> Result<Vec<FunctionId>, SuperConstructorDependencyError> {
    registry
        .definition_dependency_owners_checked(caller)
        .map(|owners| owners.into_iter().map(|(id, _)| id).collect())
}

fn retained_targets_for_body(
    registry: &FunctionRegistry,
    body: &super::FunctionBytecode,
    index: &SuperConstructorIndex,
    selected: &[FunctionId],
) -> Result<Vec<RetainedSuperConstructorTarget>, SuperConstructorDependencyError> {
    let mut targets = Vec::new();
    for instruction in &body.instructions {
        let Some((requesting_class, super_class)) = instruction_classes(instruction) else {
            continue;
        };
        if body
            .class_method_owner
            .as_ref()
            .is_none_or(|owner| owner.is_static || &owner.declaring_class != requesting_class)
        {
            return Err(SuperConstructorDependencyError::CallingOwnerMismatch);
        }
        let Some(target_id) = index.target(super_class, body.binding_origin)? else {
            continue;
        };
        let target_body = registry
            .get(target_id)
            .expect("target was selected from the exact definition graph");
        let origin = target_body
            .binding_origin
            .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
        let caller_origin = body
            .binding_origin
            .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
        let owner = target_body
            .class_method_owner
            .clone()
            .expect("target was selected from a typed constructor owner");
        let (caller_declaration, target_declaration) = admitted_class_declarations(
            registry,
            selected,
            caller_origin,
            origin,
            requesting_class,
            super_class,
            &owner,
        )?;
        if !targets
            .iter()
            .any(|existing: &RetainedSuperConstructorTarget| {
                existing.requesting_class == *requesting_class
                    && existing.owner.declaring_class == *super_class
                    && existing.caller_origin == caller_origin
            })
        {
            targets.push(RetainedSuperConstructorTarget {
                requesting_class: requesting_class.clone(),
                caller_origin,
                function: target_id,
                owner,
                origin,
                caller_declaration,
                target_declaration,
            });
        }
    }
    Ok(targets)
}

fn admitted_class_declarations(
    registry: &FunctionRegistry,
    selected: &[FunctionId],
    caller_origin: FunctionBindingOrigin,
    target_origin: FunctionBindingOrigin,
    requesting_class: &ClassIdentity,
    super_class: &ClassIdentity,
    owner: &ClassMethodOwner,
) -> Result<(RetainedClassDeclaration, RetainedClassDeclaration), SuperConstructorDependencyError> {
    let mut caller_declaration = None;
    let mut target_declaration = None;
    for body in selected.iter().filter_map(|id| registry.get(*id)) {
        let Some(body_origin) = body.binding_origin else {
            continue;
        };
        for instruction in &body.instructions {
            let Instr::RegisterClass {
                name,
                super_class: declared_parent,
                methods,
                ..
            } = instruction
            else {
                continue;
            };
            if body_origin.program == caller_origin.program && name == requesting_class {
                if declared_parent.as_ref() != Some(super_class) {
                    return Err(SuperConstructorDependencyError::ConflictingDeclaration);
                }
                caller_declaration = Some(RetainedClassDeclaration {
                    class: name.clone(),
                    parent: declared_parent.clone(),
                    constructor: None,
                });
            }
            if body_origin.program != target_origin.program || name != super_class {
                continue;
            }
            let Some(method) = methods.iter().find(|method| method.name == owner.method) else {
                continue;
            };
            if method.is_static || method.is_abstract {
                return Err(SuperConstructorDependencyError::ConflictingDeclaration);
            }
            let candidate = RetainedClassDeclaration {
                class: name.clone(),
                parent: declared_parent.clone(),
                constructor: Some(RetainedConstructorDeclaration {
                    name: method.name.clone(),
                    is_static: method.is_static,
                    is_abstract: method.is_abstract,
                    access: method.access,
                }),
            };
            if target_declaration
                .as_ref()
                .is_some_and(|prior: &RetainedClassDeclaration| {
                    prior.parent != candidate.parent
                        || prior.constructor.as_ref().is_some_and(|prior| {
                            prior.name != method.name || prior.access != method.access
                        })
                })
            {
                return Err(SuperConstructorDependencyError::ConflictingDeclaration);
            }
            target_declaration = Some(candidate);
        }
    }
    Ok((
        caller_declaration.ok_or(SuperConstructorDependencyError::MissingDeclaration)?,
        target_declaration.ok_or(SuperConstructorDependencyError::MissingDeclaration)?,
    ))
}

#[cfg(test)]
#[path = "super_constructor_dependencies/implicit_tests.rs"]
mod implicit_tests;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_execution::Digest;
    use runmat_types::{ClassMethodRole, MemberAccess, MethodName, ProgramFunctionId};

    use super::*;
    use crate::bytecode::{instr::BytecodeClassMethod, FunctionBytecode};

    fn owner(class: &str) -> ClassMethodOwner {
        ClassMethodOwner {
            declaring_class: ClassIdentity::new(class).expect("class"),
            method: MethodName::from(class),
            is_static: false,
            role: ClassMethodRole::Constructor,
        }
    }

    fn origin(function: u32) -> FunctionBindingOrigin {
        FunctionBindingOrigin {
            program: Digest::sha256("super-graph"),
            function: ProgramFunctionId(function),
        }
    }

    fn product_origin(product: &str, function: u32) -> FunctionBindingOrigin {
        FunctionBindingOrigin {
            program: Digest::sha256(product),
            function: ProgramFunctionId(function),
        }
    }

    fn derived_body(id: usize, product: &str, access: MemberAccess) -> FunctionBytecode {
        FunctionBytecode {
            function: FunctionId(id),
            class_method_owner: Some(owner("Derived")),
            binding_origin: Some(product_origin(product, 1)),
            instructions: vec![
                declaration("Derived", Some("Base"), Vec::new()),
                declaration(
                    "Base",
                    None,
                    vec![BytecodeClassMethod {
                        name: MethodName::from("Base"),
                        function_name: "Base".into(),
                        role: runmat_types::ClassMethodRole::Constructor,
                        is_static: false,
                        is_abstract: false,
                        is_sealed: false,
                        access,
                    }],
                ),
                Instr::CallSuperConstructorMulti {
                    current_class: ClassIdentity::new("Derived").expect("derived"),
                    super_class: ClassIdentity::new("Base").expect("base"),
                    arg_count: 0,
                    out_count: 1,
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn same_classes_in_two_products_keep_distinct_caller_and_access() {
        let registry = FunctionRegistry::new(HashMap::from([
            (
                FunctionId(1),
                derived_body(1, "earlier", MemberAccess::Protected),
            ),
            (
                FunctionId(2),
                FunctionBytecode {
                    function: FunctionId(2),
                    class_method_owner: Some(owner("Base")),
                    binding_origin: Some(product_origin("earlier", 2)),
                    ..Default::default()
                },
            ),
            (
                FunctionId(3),
                derived_body(3, "later", MemberAccess::Public),
            ),
            (
                FunctionId(4),
                FunctionBytecode {
                    function: FunctionId(4),
                    class_method_owner: Some(owner("Base")),
                    binding_origin: Some(product_origin("later", 2)),
                    ..Default::default()
                },
            ),
        ]));
        let earlier = registry
            .retained_super_constructor_targets_for(FunctionId(1))
            .expect("earlier owner");
        let later = registry
            .retained_super_constructor_targets_for(FunctionId(3))
            .expect("later owner");
        assert_eq!(earlier.len(), 1);
        assert_eq!(later.len(), 1);
        assert_eq!(earlier[0].origin, product_origin("earlier", 2));
        assert_eq!(
            earlier[0]
                .target_declaration
                .constructor
                .as_ref()
                .unwrap()
                .access,
            MemberAccess::Protected
        );
        assert_eq!(later[0].origin, product_origin("later", 2));
        assert_eq!(
            later[0]
                .target_declaration
                .constructor
                .as_ref()
                .unwrap()
                .access,
            MemberAccess::Public
        );
        assert_eq!(
            registry.retained_super_constructor_targets().unwrap().len(),
            2
        );
    }

    #[test]
    fn separately_admitted_parent_is_retained_by_exact_origin_and_owner() {
        let derived = FunctionBytecode {
            function: FunctionId(1),
            display_name: "Derived.Derived".into(),
            source_id: Some(crate::bytecode::source_ref::BytecodeSourceRef::Program(
                runmat_types::ProgramSourceId(1),
            )),
            class_method_owner: Some(owner("Derived")),
            binding_origin: Some(product_origin("derived-file", 1)),
            instructions: vec![
                declaration("Derived", Some("Base"), Vec::new()),
                Instr::CallSuperConstructorMulti {
                    current_class: ClassIdentity::new("Derived").expect("derived"),
                    super_class: ClassIdentity::new("Base").expect("base"),
                    arg_count: 0,
                    out_count: 1,
                },
            ],
            ..Default::default()
        };
        let base = FunctionBytecode {
            function: FunctionId(2),
            display_name: "Base.Base".into(),
            source_id: Some(crate::bytecode::source_ref::BytecodeSourceRef::Program(
                runmat_types::ProgramSourceId(2),
            )),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(product_origin("base-file", 2)),
            ..Default::default()
        };
        let base_declaration = FunctionBytecode {
            function: FunctionId(3),
            source_id: base.source_id,
            binding_origin: Some(product_origin("base-file", 3)),
            instructions: vec![declaration(
                "Base",
                None,
                vec![BytecodeClassMethod {
                    name: MethodName::from("Base"),
                    function_name: "Base".into(),
                    role: runmat_types::ClassMethodRole::Constructor,
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Protected,
                }],
            )],
            ..Default::default()
        };
        let mut registry = FunctionRegistry::new(HashMap::from([
            (derived.function, derived.clone()),
            (base.function, base.clone()),
            (base_declaration.function, base_declaration),
        ]));
        let targets = registry
            .retained_super_constructor_targets_for(derived.function)
            .expect("cross-file selected graph");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].caller_origin, product_origin("derived-file", 1));
        assert_eq!(targets[0].origin, product_origin("base-file", 2));
        assert_eq!(
            targets[0]
                .target_declaration
                .constructor
                .as_ref()
                .unwrap()
                .access,
            MemberAccess::Protected
        );
        let owners = registry
            .definition_dependency_owners(derived.function)
            .expect("cross-file dependency owners");
        assert_eq!(owners.len(), 3);
        let issuer = std::sync::Arc::new(());
        let authority = runmat_value::BoundFunctionAuthority::local(
            std::sync::Arc::downgrade(&issuer),
            runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, owners[0].0 .0),
            owners[1..]
                .iter()
                .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
                .collect(),
            runmat_value::LocalExport::Ephemeral,
        );
        registry.insert_replacing_name(FunctionBytecode {
            binding_origin: Some(product_origin("base-replacement", 2)),
            ..base
        });
        let retained = registry
            .owned_view(derived.function, &authority)
            .expect("old selected owner graph");
        assert_eq!(
            retained
                .get(FunctionId(2))
                .and_then(|body| body.binding_origin),
            Some(product_origin("base-file", 2)),
            "the selected dependency lease must not resolve to the replacement origin"
        );
        let retained_owners = retained
            .definition_dependency_owners(derived.function)
            .expect("retained selected owner graph");
        assert_eq!(
            retained_owners
                .iter()
                .map(|(id, _)| *id)
                .collect::<std::collections::HashSet<_>>(),
            std::collections::HashSet::from([derived.function, FunctionId(2), FunctionId(3),]),
        );
        assert_eq!(
            retained
                .get(FunctionId(3))
                .and_then(|body| body.binding_origin),
            Some(product_origin("base-file", 3)),
        );
        let retained_index =
            SuperConstructorIndex::from_ids(&retained, retained_owners.iter().map(|(id, _)| *id));
        assert_eq!(
            retained_index
                .declaration(
                    &ClassIdentity::from("Base"),
                    Some(product_origin("derived-file", 1))
                )
                .expect("old admitted declaration"),
            Some(FunctionId(3)),
        );
        let selected = selected_super_owner_ids(&retained, derived.function)
            .expect("retained selected owner IDs");
        assert!(
            selected.iter().any(|id| {
                retained.get(*id).is_some_and(|body| {
                    body.binding_origin == Some(product_origin("derived-file", 1))
                        && body.instructions.iter().any(|instruction| {
                            matches!(instruction, Instr::RegisterClass { name, super_class, .. }
                            if name == &ClassIdentity::from("Derived")
                                && super_class.as_ref() == Some(&ClassIdentity::from("Base")))
                        })
                })
            }),
            "retained caller declaration is absent from the selected owner graph"
        );
        assert!(
            selected.iter().any(|id| {
                retained.get(*id).is_some_and(|body| {
                    body.binding_origin.is_some_and(|origin| {
                        origin.program == product_origin("base-file", 2).program
                    }) && body.instructions.iter().any(|instruction| {
                        matches!(instruction, Instr::RegisterClass { name, methods, .. }
                        if name == &ClassIdentity::from("Base")
                            && methods.iter().any(|method| method.name == MethodName::from("Base")))
                    })
                })
            }),
            "retained parent method declaration is absent from the selected owner graph"
        );
        let retained_target = retained
            .retained_super_constructor_targets_for(derived.function)
            .expect("old exact parent target");
        assert_eq!(retained_target.len(), 1);
        assert_eq!(retained_target[0].origin, product_origin("base-file", 2));
    }

    #[test]
    fn multiple_external_parent_products_refuse_ambiguous_selection() {
        let derived = derived_body(1, "derived-file", MemberAccess::Public);
        let external_base = |id, product| FunctionBytecode {
            function: FunctionId(id),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(product_origin(product, 2)),
            ..Default::default()
        };
        let mut derived = derived;
        derived.instructions.remove(1);
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived),
            (FunctionId(2), external_base(2, "base-one")),
            (FunctionId(3), external_base(3, "base-two")),
        ]));
        assert!(matches!(
            registry.retained_super_constructor_targets_for(FunctionId(1)),
            Err(SuperConstructorDependencyError::AmbiguousTarget)
        ));
        assert!(registry
            .definition_dependency_owners(FunctionId(1))
            .is_none());
    }

    #[test]
    fn constructor_without_matching_immutable_declaration_is_rejected() {
        let mut derived = derived_body(1, "product", MemberAccess::Protected);
        derived.instructions.remove(1);
        let base = FunctionBytecode {
            function: FunctionId(2),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(product_origin("product", 2)),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived),
            (FunctionId(2), base),
        ]));
        assert!(matches!(
            registry.retained_super_constructor_targets_for(FunctionId(1)),
            Err(SuperConstructorDependencyError::MissingDeclaration)
        ));
    }

    #[test]
    fn retained_view_uses_old_base_body_after_same_id_publication() {
        let mut earlier = derived_body(1, "earlier", MemberAccess::Protected);
        earlier.display_name = "Derived".into();
        let base = FunctionBytecode {
            function: FunctionId(2),
            display_name: "Base".into(),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(product_origin("earlier", 2)),
            ..Default::default()
        };
        let mut registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), earlier),
            (FunctionId(2), base),
        ]));
        let owners = registry
            .definition_dependency_owners(FunctionId(1))
            .expect("old complete owner graph");
        let issuer = std::sync::Arc::new(());
        let authority = runmat_value::BoundFunctionAuthority::local(
            std::sync::Arc::downgrade(&issuer),
            runmat_value::FunctionDefinitionOwner::issue(&owners[0].1, owners[0].0 .0),
            owners[1..]
                .iter()
                .map(|(id, owner)| runmat_value::FunctionDefinitionOwner::issue(owner, id.0))
                .collect(),
            runmat_value::LocalExport::Ephemeral,
        );
        let mut new_derived = derived_body(1, "later", MemberAccess::Public);
        new_derived.display_name = "Derived".into();
        registry.insert_replacing_name(new_derived);
        registry.insert_replacing_name(FunctionBytecode {
            function: FunctionId(2),
            display_name: "Base".into(),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(product_origin("later", 2)),
            ..Default::default()
        });
        let retained = registry
            .owned_view(FunctionId(1), &authority)
            .expect("retained old caller and base");
        let old = retained
            .retained_super_constructor_targets_for(FunctionId(1))
            .expect("old selected target");
        assert_eq!(old.len(), 1);
        assert_eq!(old[0].origin, product_origin("earlier", 2));
        assert_eq!(
            old[0]
                .target_declaration
                .constructor
                .as_ref()
                .unwrap()
                .access,
            MemberAccess::Protected
        );
        let current = registry
            .retained_super_constructor_targets_for(FunctionId(1))
            .expect("new selected target");
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].origin, product_origin("later", 2));
        assert_eq!(
            current[0]
                .target_declaration
                .constructor
                .as_ref()
                .unwrap()
                .access,
            MemberAccess::Public
        );
    }

    fn declaration(name: &str, parent: Option<&str>, methods: Vec<BytecodeClassMethod>) -> Instr {
        Instr::RegisterClass {
            name: ClassIdentity::new(name).expect("class"),
            super_class: parent.map(|parent| ClassIdentity::new(parent).expect("parent")),
            is_sealed: false,
            is_abstract: false,
            properties: Vec::new(),
            methods,
            events: Vec::new(),
            enumerations: Vec::new(),
        }
    }

    #[test]
    fn superclass_edge_selects_exact_typed_origin_and_retains_body() {
        let derived = FunctionBytecode {
            function: FunctionId(1),
            class_method_owner: Some(owner("Derived")),
            binding_origin: Some(origin(1)),
            instructions: vec![
                declaration("Derived", Some("Base"), Vec::new()),
                declaration(
                    "Base",
                    None,
                    vec![BytecodeClassMethod {
                        name: MethodName::from("Base"),
                        function_name: "Base".into(),
                        role: runmat_types::ClassMethodRole::Constructor,
                        is_static: false,
                        is_abstract: false,
                        is_sealed: false,
                        access: MemberAccess::Protected,
                    }],
                ),
                Instr::CallSuperConstructorMulti {
                    current_class: ClassIdentity::new("Derived").expect("derived"),
                    super_class: ClassIdentity::new("Base").expect("base"),
                    arg_count: 0,
                    out_count: 1,
                },
            ],
            ..Default::default()
        };
        let base = FunctionBytecode {
            function: FunctionId(2),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(origin(2)),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived),
            (FunctionId(2), base),
            (
                FunctionId(3),
                FunctionBytecode {
                    function: FunctionId(3),
                    class_method_owner: Some(owner("Base")),
                    binding_origin: Some(FunctionBindingOrigin {
                        program: Digest::sha256("later-super-graph"),
                        function: ProgramFunctionId(2),
                    }),
                    ..Default::default()
                },
            ),
        ]));
        let targets = registry
            .retained_super_constructor_targets()
            .expect("typed targets");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].requesting_class.display_name(), "Derived");
        assert_eq!(targets[0].caller_origin, origin(1));
        assert_eq!(targets[0].owner.declaring_class.display_name(), "Base");
        assert_eq!(targets[0].origin, origin(2));
        assert_eq!(
            targets[0]
                .target_declaration
                .constructor
                .as_ref()
                .unwrap()
                .access,
            MemberAccess::Protected
        );
        assert_eq!(
            registry
                .definition_dependency_owners(FunctionId(1))
                .expect("complete graph")
                .len(),
            2
        );
    }

    #[test]
    fn ambiguous_or_foreign_owner_cannot_be_selected() {
        let derived = FunctionBytecode {
            function: FunctionId(1),
            class_method_owner: Some(owner("Other")),
            instructions: vec![Instr::CallSuperConstructorMulti {
                current_class: ClassIdentity::new("Derived").expect("derived"),
                super_class: ClassIdentity::new("Base").expect("base"),
                arg_count: 0,
                out_count: 1,
            }],
            ..Default::default()
        };
        let base = FunctionBytecode {
            function: FunctionId(2),
            class_method_owner: Some(owner("Base")),
            binding_origin: Some(origin(2)),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived.clone()),
            (FunctionId(2), base.clone()),
        ]));
        assert!(matches!(
            registry.retained_super_constructor_targets(),
            Err(SuperConstructorDependencyError::CallingOwnerMismatch)
        ));
        assert!(registry
            .definition_dependency_owners(FunctionId(1))
            .is_none());

        let registry = FunctionRegistry::new(HashMap::from([
            (
                FunctionId(1),
                FunctionBytecode {
                    class_method_owner: Some(owner("Derived")),
                    ..derived
                },
            ),
            (FunctionId(2), base.clone()),
            (
                FunctionId(3),
                FunctionBytecode {
                    function: FunctionId(3),
                    ..base
                },
            ),
        ]));
        assert!(matches!(
            registry.retained_super_constructor_targets(),
            Err(SuperConstructorDependencyError::AmbiguousTarget)
        ));
        assert!(registry
            .definition_dependency_owners(FunctionId(1))
            .is_none());
    }

    #[test]
    fn compiled_base_without_portable_origin_is_not_an_executable_binding() {
        let derived = FunctionBytecode {
            function: FunctionId(1),
            class_method_owner: Some(owner("Derived")),
            instructions: vec![Instr::CallSuperConstructorExpandMultiOutput {
                current_class: ClassIdentity::new("Derived").expect("derived"),
                super_class: ClassIdentity::new("Base").expect("base"),
                specs: Vec::new(),
                out_count: 1,
            }],
            ..Default::default()
        };
        let base = FunctionBytecode {
            function: FunctionId(2),
            class_method_owner: Some(owner("Base")),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), derived),
            (FunctionId(2), base),
        ]));
        assert!(matches!(
            registry.retained_super_constructor_targets(),
            Err(SuperConstructorDependencyError::MissingOrigin)
        ));
    }
}
