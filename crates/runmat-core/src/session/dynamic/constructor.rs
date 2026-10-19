use runmat_runtime::{build_runtime_error, RuntimeError};

/// Package class files spell their classdef locally (`Class`) while the
/// selected source identity is qualified (`pkg.Class`). Validate that lexical
/// declaration before giving the primary AST its typed selected identity.
/// This is source admission, not a runtime name fallback.
pub(super) fn qualify_requested_source_classdef(
    phase: &runmat_runtime::user_functions::DynamicFunctionLoadPhase,
    body: &mut [runmat_parser::Stmt],
) -> Result<(), RuntimeError> {
    let requested = match phase {
        runmat_runtime::user_functions::DynamicFunctionLoadPhase::RegisteredConstructorSource(
            requested,
        )
        | runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
            class: Some(requested),
            ..
        } => requested,
        _ => return Ok(()),
    };
    let mut declarations = body.iter().filter_map(|stmt| {
        if let runmat_parser::Stmt::ClassDef { name, .. } = stmt {
            Some(name)
        } else {
            None
        }
    });
    let Some(declared) = declarations.next() else {
        return Err(source_mismatch());
    };
    if declarations.next().is_some() {
        return Err(source_mismatch());
    }
    let declared_identity =
        runmat_types::ClassIdentity::new(declared.as_str()).map_err(|_| source_mismatch())?;
    let source_local_class = declared_identity.qualified_name().0.len() == 1
        && declared_identity.qualified_name().0.last() == requested.qualified_name().0.last();
    if declared_identity != *requested && !source_local_class {
        return Err(source_mismatch());
    }
    super::super::compile::qualify_companion_classdefs(body, requested.display_name());
    Ok(())
}

/// A cold `@name` may select a class file before that class is registered.
/// Its source declaration must still match the selected typed path identity
/// before we issue a constructor handle from the compiled body.
pub(super) fn qualify_handle_source_classdef(
    name: &str,
    body: &mut [runmat_parser::Stmt],
) -> Result<(), RuntimeError> {
    if !body
        .iter()
        .any(|stmt| matches!(stmt, runmat_parser::Stmt::ClassDef { .. }))
    {
        return Ok(());
    }
    let class = runmat_types::ClassIdentity::new(name).map_err(|_| source_mismatch())?;
    qualify_requested_source_classdef(
        &runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
            class: Some(class),
            selected_source: None,
        },
        body,
    )
}

fn source_mismatch() -> RuntimeError {
    build_runtime_error("selected source does not define the registered class constructor")
        .with_identifier("RunMat:ConstructorSourceMismatch")
        .build()
}

/// A registered class may ask the source loader for its unresolved constructor,
/// but a same-named ordinary function (or another class's method) cannot
/// replace the class construction path. Check before publication or invocation.
pub(super) fn require_requested_source_constructor(
    phase: &runmat_runtime::user_functions::DynamicFunctionLoadPhase,
    owner: Option<&runmat_types::ClassMethodOwner>,
) -> Result<(), RuntimeError> {
    let requested = match phase {
        runmat_runtime::user_functions::DynamicFunctionLoadPhase::RegisteredConstructorSource(
            requested,
        )
        | runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
            class: Some(requested),
            ..
        } => requested,
        _ => return Ok(()),
    };
    if owner.is_some_and(|owner| {
        owner.role == runmat_types::ClassMethodRole::Constructor
            && !owner.is_static
            && &owner.declaring_class == requested
    }) {
        return Ok(());
    }
    Err(source_mismatch())
}

/// The constructor branch is selected only from the exact admitted function,
/// never from the requested source spelling or a method-name heuristic.
pub(super) fn admitted_constructor_owner(
    unit: &crate::ExecutableUnit,
    bound: &runmat_vm::RuntimeBoundExecutableUnit,
    origin: runmat_execution::FunctionBindingOrigin,
) -> Result<Option<runmat_types::ClassMethodOwner>, RuntimeError> {
    let Some(function) = bound.entrypoint_function() else {
        return Ok(None);
    };
    let owner = registered_constructor_owner(bound.registry(), function, origin)?;
    let metadata = unit.mir().functions.get(&function).ok_or_else(|| {
        build_runtime_error("admitted constructor lacks its MIR declaration")
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build()
    })?;
    let body = bound.registry().get(function).ok_or_else(|| {
        build_runtime_error("admitted constructor lacks its compiled body")
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build()
    })?;
    if metadata.class_method_owner != body.class_method_owner {
        return Err(
            build_runtime_error("admitted method owner disagrees with its compiled body")
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build(),
        );
    }
    let Some(owner) = owner else {
        return Ok(None);
    };
    if !matches!(
        metadata.kind,
        runmat_hir::FunctionKind::ClassMethod {
            is_static: false,
            role: runmat_types::ClassMethodRole::Constructor,
        }
    ) || metadata.class_method_owner.as_ref() != Some(&owner)
    {
        return Err(
            build_runtime_error("admitted constructor owner disagrees with its binding")
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build(),
        );
    }
    Ok(Some(owner))
}

/// Cached products retain the same admitted function registry and immutable
/// origin checked on the cold path. Rebinding changes only runtime-local
/// source leases, not this typed method declaration.
pub(super) fn retained_constructor_owner(
    admitted: &runmat_vm::AdmittedExecutableUnit,
    origin: runmat_execution::FunctionBindingOrigin,
) -> Result<Option<runmat_types::ClassMethodOwner>, RuntimeError> {
    let Some(function) = admitted.entrypoint_function() else {
        return Ok(None);
    };
    registered_constructor_owner(admitted.registry(), function, origin)
}

fn registered_constructor_owner(
    registry: &runmat_vm::FunctionRegistry,
    function: runmat_hir::FunctionId,
    origin: runmat_execution::FunctionBindingOrigin,
) -> Result<Option<runmat_types::ClassMethodOwner>, RuntimeError> {
    let body = registry.get(function).ok_or_else(|| {
        build_runtime_error("admitted constructor lacks its compiled body")
            .with_identifier("RunMat:ConstructorOwnerInvalid")
            .build()
    })?;
    let Some(owner) = body.class_method_owner.as_ref() else {
        return Ok(None);
    };
    if owner.role != runmat_types::ClassMethodRole::Constructor {
        return Ok(None);
    }
    if owner.is_static || body.function != function || body.binding_origin != Some(origin) {
        return Err(
            build_runtime_error("admitted constructor owner disagrees with its binding")
                .with_identifier("RunMat:ConstructorOwnerInvalid")
                .build(),
        );
    }
    Ok(Some(owner.clone()))
}

#[cfg(test)]
mod source_selection_tests {
    use super::*;
    use runmat_runtime::user_functions::DynamicFunctionLoadPhase;
    use runmat_types::{ClassIdentity, ClassMethodOwner, ClassMethodRole, MethodName};

    #[test]
    fn registered_source_requires_the_exact_nonstatic_constructor_owner() {
        let requested = ClassIdentity::new("SourceCtor").expect("class identity");
        let phase = DynamicFunctionLoadPhase::RegisteredConstructorSource(requested.clone());
        let owner = ClassMethodOwner {
            declaring_class: requested.clone(),
            method: MethodName::from("SourceCtor"),
            is_static: false,
            role: ClassMethodRole::Constructor,
        };
        assert!(require_requested_source_constructor(&phase, Some(&owner)).is_ok());
        for other in [
            None,
            Some(ClassMethodOwner {
                declaring_class: ClassIdentity::new("Other").expect("class identity"),
                ..owner.clone()
            }),
            Some(ClassMethodOwner {
                role: ClassMethodRole::Ordinary,
                ..owner.clone()
            }),
            Some(ClassMethodOwner {
                is_static: true,
                ..owner.clone()
            }),
        ] {
            let error = require_requested_source_constructor(&phase, other.as_ref())
                .expect_err("unrelated source cannot shadow registered class");
            assert_eq!(error.identifier(), Some("RunMat:ConstructorSourceMismatch"));
        }
        assert!(require_requested_source_constructor(
            &DynamicFunctionLoadPhase::AfterSemantic,
            None,
        )
        .is_ok());
    }

    #[test]
    fn package_source_classdef_is_qualified_only_after_exact_declaration_check() {
        let phase = DynamicFunctionLoadPhase::RegisteredConstructorSource(
            ClassIdentity::new("pkg.SeedDerived").expect("class identity"),
        );
        let mut source = runmat_parser::parse_with_options(
            "classdef SeedDerived < handle\n methods\n function obj = SeedDerived()\n end\n end\nend",
            runmat_parser::ParserOptions::default(),
        )
        .expect("package class source");
        qualify_requested_source_classdef(&phase, &mut source.body)
            .expect("selected package class source");
        assert!(matches!(
            source.body.as_slice(),
            [runmat_parser::Stmt::ClassDef { name, .. }] if name == "pkg.SeedDerived"
        ));

        let mut unrelated = runmat_parser::parse_with_options(
            "classdef Other < handle\nend",
            runmat_parser::ParserOptions::default(),
        )
        .expect("unrelated class source");
        let error = qualify_requested_source_classdef(&phase, &mut unrelated.body)
            .expect_err("unrelated declaration cannot inherit selected package identity");
        assert_eq!(error.identifier(), Some("RunMat:ConstructorSourceMismatch"));
    }
}
