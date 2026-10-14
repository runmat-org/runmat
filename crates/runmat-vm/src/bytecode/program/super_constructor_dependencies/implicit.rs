use super::*;

#[derive(Debug, Clone)]
pub struct RetainedImplicitSuperConstructorTarget {
    pub caller_origin: FunctionBindingOrigin,
    pub caller_declaration: RetainedClassDeclaration,
    pub parent_declaration: Option<RetainedClassDeclaration>,
}

impl RetainedImplicitSuperConstructorTarget {
    pub fn into_binding(
        self,
    ) -> Result<
        runmat_runtime::context::RetainedSuperConstructorBinding,
        runmat_runtime::RuntimeError,
    > {
        match self.parent_declaration {
            Some(parent) => runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_linked_implicit(
                &self.caller_declaration,
                &parent,
                self.caller_origin,
            ),
            None => runmat_runtime::context::RetainedSuperConstructorBinding::from_admitted_intrinsic(
                &self.caller_declaration,
                self.caller_origin,
            ),
        }
    }
}

impl FunctionRegistry {
    /// Exact no-body superclass calls from one selected owner graph. An
    /// intrinsic parent is canonical; a user parent requires its admitted
    /// RegisterClass declaration in this same retained graph.
    pub fn retained_implicit_super_constructor_targets_for(
        &self,
        caller: FunctionId,
    ) -> Result<Vec<RetainedImplicitSuperConstructorTarget>, SuperConstructorDependencyError> {
        self.retained_implicit_super_constructor_targets_with_linked(caller, &[])
    }

    /// A script-only admitted class with no methods has no FunctionBytecode to
    /// carry its RegisterClass instruction. The prepared-unit owner supplies
    /// that exact linked declaration; arbitrary class-name lookup cannot.
    pub(crate) fn retained_implicit_super_constructor_targets_with_linked(
        &self,
        caller: FunctionId,
        linked: &[RetainedClassDeclaration],
    ) -> Result<Vec<RetainedImplicitSuperConstructorTarget>, SuperConstructorDependencyError> {
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
        let selected = super::selected_super_owner_ids(self, caller)?;
        let index = SuperConstructorIndex::from_ids(self, selected.iter().copied());
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
            if index.target(super_class, body.binding_origin)?.is_some() {
                continue;
            }
            let caller_origin = body
                .binding_origin
                .ok_or(SuperConstructorDependencyError::MissingOrigin)?;
            let caller_declaration_id =
                index
                    .declaration(requesting_class, Some(caller_origin))?
                    .ok_or(SuperConstructorDependencyError::MissingDeclaration)?;
            let caller_body = self
                .get(caller_declaration_id)
                .ok_or(SuperConstructorDependencyError::MissingDeclaration)?;
            if caller_body.binding_origin.map(|origin| origin.program)
                != Some(caller_origin.program)
            {
                return Err(SuperConstructorDependencyError::MissingDeclaration);
            }
            let caller_declaration = caller_body
                .instructions
                .iter()
                .filter_map(|instruction| match instruction {
                    Instr::RegisterClass {
                        name,
                        super_class: parent,
                        ..
                    } if name == requesting_class && parent.as_ref() == Some(super_class) => {
                        Some(RetainedClassDeclaration {
                            class: name.clone(),
                            parent: parent.clone(),
                            constructor: None,
                        })
                    }
                    _ => None,
                })
                .next()
                .ok_or(SuperConstructorDependencyError::MissingDeclaration)?;
            let parent_declaration =
                if runmat_runtime::class_registry::has_intrinsic_implicit_constructor(super_class) {
                    None
                } else {
                    let declaration = index.declaration(super_class, body.binding_origin)?;
                    let parent = if let Some(declaration) = declaration {
                        let parent_body = self
                            .get(declaration)
                            .ok_or(SuperConstructorDependencyError::MissingDeclaration)?;
                        parent_body
                            .instructions
                            .iter()
                            .find_map(|instruction| match instruction {
                                Instr::RegisterClass {
                                    name,
                                    super_class: parent,
                                    ..
                                } if name == super_class => Some(RetainedClassDeclaration {
                                    class: name.clone(),
                                    parent: parent.clone(),
                                    constructor: None,
                                }),
                                _ => None,
                            })
                            .ok_or(SuperConstructorDependencyError::MissingDeclaration)?
                    } else {
                        linked
                            .iter()
                            .find(|declaration| declaration.class == *super_class)
                            .cloned()
                            .ok_or(SuperConstructorDependencyError::MissingDeclaration)?
                    };
                    Some(parent)
                };
            if !targets
                .iter()
                .any(|existing: &RetainedImplicitSuperConstructorTarget| {
                    existing.caller_origin == caller_origin
                        && existing.caller_declaration.class == caller_declaration.class
                })
            {
                targets.push(RetainedImplicitSuperConstructorTarget {
                    caller_origin,
                    caller_declaration,
                    parent_declaration,
                });
            }
        }
        Ok(targets)
    }
}
