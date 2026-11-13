//! Exact typed class-declaration and constructor selection in one admitted registry.

use std::collections::HashMap;

use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_hir::FunctionId;
use runmat_runtime::source_context::SelectedProgramSourceOwner;
use runmat_types::{ClassIdentity, ClassMethodRole};

use super::SuperConstructorDependencyError;
use crate::bytecode::{BytecodeSourceRef, FunctionRegistry, Instr};

struct IndexedDeclaration<'a> {
    function: FunctionId,
    program: Option<Digest>,
    source: Option<BytecodeSourceRef>,
    owner: Option<&'a SelectedProgramSourceOwner>,
    instruction: &'a Instr,
}

pub(in crate::bytecode::program) struct SuperConstructorIndex<'a> {
    by_class: HashMap<ClassIdentity, Vec<(FunctionId, Option<Digest>)>>,
    declarations: HashMap<ClassIdentity, Vec<IndexedDeclaration<'a>>>,
}

impl<'a> SuperConstructorIndex<'a> {
    pub(in crate::bytecode::program) fn new(registry: &'a FunctionRegistry) -> Self {
        Self::from_ids(registry, registry.functions.keys().copied())
    }

    pub(in crate::bytecode::program) fn from_ids(
        registry: &'a FunctionRegistry,
        ids: impl IntoIterator<Item = FunctionId>,
    ) -> Self {
        let mut by_class = HashMap::new();
        let mut declarations = HashMap::new();
        for id in ids {
            let Some(body) = registry.get(id) else {
                continue;
            };
            for instruction in &body.instructions {
                if let Instr::RegisterClass { name, .. } = instruction {
                    declarations
                        .entry(name.clone())
                        .or_insert_with(Vec::new)
                        .push(IndexedDeclaration {
                            function: id,
                            program: body.binding_origin.map(|origin| origin.program),
                            source: body.source_id,
                            owner: body.selected_source_owner.as_ref(),
                            instruction,
                        });
                }
            }
            let Some(owner) = body
                .class_method_owner
                .as_ref()
                .filter(|owner| owner.role == ClassMethodRole::Constructor && !owner.is_static)
            else {
                continue;
            };
            by_class
                .entry(owner.declaring_class.clone())
                .or_insert_with(Vec::new)
                .push((id, body.binding_origin.map(|origin| origin.program)));
        }
        Self {
            by_class,
            declarations,
        }
    }

    pub(in crate::bytecode::program) fn declaration(
        &self,
        class: &ClassIdentity,
        caller_origin: Option<FunctionBindingOrigin>,
    ) -> Result<Option<FunctionId>, SuperConstructorDependencyError> {
        let candidates = self.declarations.get(class);
        if let Some(origin) = caller_origin {
            let same_product = candidates
                .into_iter()
                .flatten()
                .filter(|candidate| candidate.program == Some(origin.program))
                .collect::<Vec<_>>();
            let first = Self::unique_declaration(&same_product)?;
            if first.is_some() {
                return Ok(first);
            }
        }
        let matches = candidates.into_iter().flatten().collect::<Vec<_>>();
        Self::unique_declaration(&matches)
    }

    fn unique_declaration(
        candidates: &[&IndexedDeclaration<'_>],
    ) -> Result<Option<FunctionId>, SuperConstructorDependencyError> {
        let Some(first) = candidates.first() else {
            return Ok(None);
        };
        for candidate in candidates.iter().skip(1) {
            if first.program.is_none() || first.program != candidate.program {
                return Err(SuperConstructorDependencyError::AmbiguousTarget);
            }
            let same_owner = match (first.owner, candidate.owner) {
                (Some(left), Some(right)) => left.same_issuance(right),
                (None, None) => true,
                _ => false,
            };
            if first.source != candidate.source
                || !same_owner
                || !same_class_registration(first.instruction, candidate.instruction)
            {
                return Err(SuperConstructorDependencyError::ConflictingDeclaration);
            }
        }
        Ok(Some(first.function))
    }

    pub(in crate::bytecode::program) fn target(
        &self,
        super_class: &ClassIdentity,
        caller_origin: Option<FunctionBindingOrigin>,
    ) -> Result<Option<FunctionId>, SuperConstructorDependencyError> {
        let candidates = self.by_class.get(super_class);
        if let Some(origin) = caller_origin {
            let mut same_product = candidates
                .into_iter()
                .flatten()
                .filter(|(_, program)| *program == Some(origin.program));
            let first = same_product.next().map(|(id, _)| *id);
            if same_product.next().is_some() {
                return Err(SuperConstructorDependencyError::AmbiguousTarget);
            }
            if first.is_some() {
                return Ok(first);
            }
        }
        // Distinct constructors remain ambiguous even when declarations agree.
        let mut matches = candidates.into_iter().flatten();
        let first = matches.next().map(|(id, _)| *id);
        if matches.next().is_some() {
            Err(SuperConstructorDependencyError::AmbiguousTarget)
        } else {
            Ok(first)
        }
    }
}

pub(crate) fn same_class_registration(left: &Instr, right: &Instr) -> bool {
    let (
        Instr::RegisterClass {
            name: left_name,
            super_class: left_parent,
            is_sealed: left_sealed,
            is_abstract: left_abstract,
            properties: left_properties,
            methods: left_methods,
            events: left_events,
            enumerations: left_enumerations,
        },
        Instr::RegisterClass {
            name: right_name,
            super_class: right_parent,
            is_sealed: right_sealed,
            is_abstract: right_abstract,
            properties: right_properties,
            methods: right_methods,
            events: right_events,
            enumerations: right_enumerations,
        },
    ) = (left, right)
    else {
        return false;
    };
    left_name == right_name
        && left_parent == right_parent
        && left_sealed == right_sealed
        && left_abstract == right_abstract
        && left_properties == right_properties
        && left_methods == right_methods
        && left_events == right_events
        && left_enumerations == right_enumerations
}
