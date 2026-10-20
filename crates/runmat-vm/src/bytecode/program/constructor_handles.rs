//! Typed constructor-handle selection from the current admitted registry.
//! This does not add a class-name alias to ordinary function lookup.

use runmat_hir::FunctionId;
use runmat_types::{ClassIdentity, ClassMethodOwner, ClassMethodRole};

use super::FunctionRegistry;
use crate::bytecode::Instr;

impl FunctionRegistry {
    pub fn resolve_session_constructor_handle(
        &self,
        name: &str,
    ) -> Result<Option<FunctionId>, &'static str> {
        let Ok(class) = ClassIdentity::new(name) else {
            return Ok(None);
        };
        let mut selected = None;
        for (id, body) in &self.functions {
            let Some(owner) = body.class_method_owner.as_ref().filter(|owner| {
                owner.declaring_class == class
                    && owner.role == ClassMethodRole::Constructor
                    && !owner.is_static
            }) else {
                continue;
            };
            let declared = body.selected_class_product.as_ref().is_some_and(|product| {
                product.class() == &class
                    && declares_constructor(product.declarations(), &class, owner, body)
            }) || body.binding_origin.is_some_and(|origin| {
                self.functions.values().any(|candidate| {
                    candidate
                        .binding_origin
                        .is_some_and(|declaration| declaration.program == origin.program)
                        && declares_constructor(&candidate.instructions, &class, owner, body)
                })
            });
            if !declared {
                continue;
            }
            if selected.replace(*id).is_some() {
                return Err("multiple admitted constructor definitions match one handle");
            }
        }
        Ok(selected)
    }
}

fn declares_constructor(
    instructions: &[Instr],
    class: &ClassIdentity,
    owner: &ClassMethodOwner,
    body: &super::FunctionBytecode,
) -> bool {
    instructions.iter().any(|instruction| {
        let Instr::RegisterClass { name, methods, .. } = instruction else {
            return false;
        };
        name == class
            && methods.iter().any(|method| {
                method.name == owner.method
                    && method.function_name == body.display_name
                    && !method.is_static
                    && !method.is_abstract
            })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use runmat_execution::{Digest, FunctionBindingOrigin};
    use runmat_types::{ClassMethodOwner, MethodName, ProgramFunctionId};

    use super::*;
    use crate::bytecode::{
        instr::BytecodeClassMethod, remap_session_entrypoint, Bytecode, FunctionBytecode,
        SessionCallableBinding,
    };

    fn constructor(id: usize, class_name: &str, product: &str) -> FunctionBytecode {
        let class = ClassIdentity::new(class_name).expect("class identity");
        let method = MethodName::from(
            class
                .qualified_name()
                .0
                .last()
                .expect("class segment")
                .0
                .as_str(),
        );
        let display_name = format!("{class_name}.{}", method.display_name());
        FunctionBytecode {
            function: FunctionId(id),
            display_name: display_name.clone(),
            binding_origin: Some(FunctionBindingOrigin {
                program: Digest::sha256(product),
                function: ProgramFunctionId(id as u32),
            }),
            class_method_owner: Some(ClassMethodOwner {
                declaring_class: class.clone(),
                method: method.clone(),
                is_static: false,
                role: ClassMethodRole::Constructor,
            }),
            instructions: vec![Instr::RegisterClass {
                name: class,
                super_class: None,
                is_sealed: false,
                is_abstract: false,
                properties: Vec::new(),
                methods: vec![BytecodeClassMethod {
                    name: method,
                    function_name: display_name,
                    role: runmat_types::ClassMethodRole::Constructor,
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: runmat_types::MemberAccess::Public,
                }],
                events: Vec::new(),
                enumerations: Vec::new(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn selected_constructor_handle_uses_declared_typed_body_without_name_alias() {
        for class in ["LocalRevision", "pkg.LocalRevision"] {
            let id = FunctionId(7);
            let registry = FunctionRegistry::new(HashMap::from([(
                id,
                constructor(7, class, "selected-product"),
            )]));
            assert_eq!(registry.resolve_name(class), None);
            assert_eq!(
                registry.resolve_session_constructor_handle(class),
                Ok(Some(id))
            );
            let source_handle = if class.contains('.') {
                Instr::CreateExternalFunctionHandle(class.into())
            } else {
                Instr::CreateFunctionHandle(class.into())
            };
            let mut bytecode = Bytecode::with_instructions(vec![source_handle], 0);
            bytecode.function_registry = registry;
            remap_session_entrypoint(
                &mut bytecode,
                &HashMap::new(),
                &[SessionCallableBinding::constructor(
                    0,
                    id,
                    class.into(),
                    crate::FunctionReferenceKind::External,
                )],
            )
            .expect("typed constructor handle seals");
            assert!(matches!(
                &bytecode.instructions[0],
                Instr::CreateExternalBoundFunctionHandle(function, name, _)
                    if *function == id && name == class
            ));
            bytecode.instructions[0] = Instr::CreateFunctionHandle(class.into());
            remap_session_entrypoint(
                &mut bytecode,
                &HashMap::new(),
                &[SessionCallableBinding::constructor(
                    0,
                    id,
                    class.into(),
                    crate::FunctionReferenceKind::Local,
                )],
            )
            .expect("the same spelling can select a current constructor");
            assert!(matches!(
                &bytecode.instructions[0],
                Instr::CreateBoundFunctionHandle(function, name, _) if *function == id && name == class
            ));
        }
    }

    #[test]
    fn constructor_handle_rejects_missing_declaration_and_ambiguous_bodies() {
        let mut missing = constructor(7, "LocalRevision", "selected-product");
        missing.instructions.clear();
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(7), missing)]));
        assert_eq!(
            registry.resolve_session_constructor_handle("LocalRevision"),
            Ok(None)
        );

        let registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(7), constructor(7, "LocalRevision", "one")),
            (FunctionId(8), constructor(8, "LocalRevision", "two")),
        ]));
        assert!(registry
            .resolve_session_constructor_handle("LocalRevision")
            .is_err());
    }
}
