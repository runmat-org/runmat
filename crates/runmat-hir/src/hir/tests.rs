use super::*;

fn span() -> Span {
    Span { start: 0, end: 0 }
}

#[test]
fn assembly_owns_core_items() {
    let module = ModuleId(0);
    let function = FunctionId(0);
    let binding = BindingId(0);
    let entrypoint = EntrypointId(0);

    let assembly = HirAssembly {
        bound_function_modifiers: Default::default(),
        statement_result_bindings: Default::default(),
        primary_source: None,
        modules: vec![HirModule {
            id: module,
            name: QualifiedName(vec![SymbolName("demo".into())]),
            source_id: ProgramSourceId(0),
            source_unit: SourceUnitKind::ScriptFile,
            imports: vec![],
            top_level_functions: vec![],
            classes: vec![],
            synthetic_entry_function: Some(function),
            script_sections: vec![],
        }],
        functions: vec![HirFunction {
            id: function,
            module,
            parent: None,
            enclosing_class: None,
            name: FunctionName("demo_entry".into()),
            kind: FunctionKind::SyntheticEntrypoint,
            params: vec![],
            outputs: vec![],
            abi: FunctionAbi {
                fixed_inputs: vec![],
                varargin: None,
                fixed_outputs: vec![],
                varargout: None,
                implicit_nargin: None,
                implicit_nargout: None,
            },
            locals: vec![binding],
            captures: vec![],
            argument_validations: vec![],
            modifiers: FunctionModifiers::default(),
            body: HirBlock { statements: vec![] },
            span: span(),
        }],
        classes: vec![],
        bindings: vec![HirBinding {
            id: binding,
            owner: BindingOwner::Function(function),
            name: BindingName("x".into()),
            role: BindingRole::Local,
            storage: BindingStorage::Lexical,
            workspace_visibility: WorkspaceVisibility::TopLevel,
            declared_span: span(),
        }],
        entrypoints: vec![HirEntrypoint {
            id: entrypoint,
            name: Some(EntrypointName("demo".into())),
            target: function,
            origin: EntrypointOrigin::SourcePath,
            policy: EntrypointPolicy {
                workspace_export: WorkspaceExportPolicy::ExportTopLevelBindings,
                top_level_await: false,
            },
        }],
    };

    assert_eq!(assembly.modules[0].synthetic_entry_function, Some(function));
    assert_eq!(assembly.entrypoints[0].target, function);
    assert_eq!(assembly.functions[0].locals, vec![binding]);
    assert!(matches!(
        assembly.bindings[0].workspace_visibility,
        WorkspaceVisibility::TopLevel
    ));
}

#[test]
fn function_abi_can_reuse_shared_input_output_binding() {
    let binding = BindingId(0);
    let abi = FunctionAbi {
        fixed_inputs: vec![binding],
        varargin: None,
        fixed_outputs: vec![binding],
        varargout: None,
        implicit_nargin: Some(BindingId(1)),
        implicit_nargout: Some(BindingId(2)),
    };

    assert_eq!(abi.fixed_inputs[0], abi.fixed_outputs[0]);
    assert_eq!(abi.implicit_nargin, Some(BindingId(1)));
    assert_eq!(abi.implicit_nargout, Some(BindingId(2)));
}

#[test]
fn fixed_output_bindings_exclude_only_variadic_declaration() {
    let mut abi = FunctionAbi {
        fixed_inputs: vec![],
        varargin: None,
        fixed_outputs: vec![BindingId(0), BindingId(1)],
        varargout: None,
        implicit_nargin: None,
        implicit_nargout: None,
    };
    assert_eq!(
        abi.fixed_output_bindings().copied().collect::<Vec<_>>(),
        abi.fixed_outputs
    );
    abi.varargout = Some(BindingId(1));
    assert_eq!(
        abi.fixed_output_bindings().copied().collect::<Vec<_>>(),
        vec![BindingId(0)]
    );
    abi.fixed_outputs.remove(0);
    assert_eq!(abi.fixed_output_bindings().count(), 0);
    assert_eq!(abi.fixed_outputs, vec![BindingId(1)]);
}

#[test]
fn captures_reference_original_binding_identity() {
    let parent = FunctionId(0);
    let child = FunctionId(1);
    let binding = BindingId(0);

    let capture = CapturedBinding {
        binding,
        from_function: parent,
    };

    let child_function = HirFunction {
        id: child,
        module: ModuleId(0),
        parent: Some(parent),
        enclosing_class: None,
        name: FunctionName("inner".into()),
        kind: FunctionKind::Named,
        params: vec![],
        outputs: vec![],
        abi: FunctionAbi {
            fixed_inputs: vec![],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        locals: vec![],
        captures: vec![capture],
        argument_validations: vec![],
        modifiers: FunctionModifiers::default(),
        body: HirBlock { statements: vec![] },
        span: span(),
    };

    assert_eq!(child_function.parent, Some(parent));
    assert_eq!(child_function.captures[0].binding, binding);
    assert_eq!(child_function.captures[0].from_function, parent);
}

#[test]
fn mutation_captures_creation_and_shape_context() {
    let binding = BindingId(0);
    let mutation = PlaceMutation {
        place: HirPlace::Binding(binding),
        kind: PlaceMutationKind::BindOrAssign,
        creation_policy: AssignmentCreationPolicy::CreateBinding,
        shape_policy: AssignmentShapePolicy::MatlabCompatible,
    };
    assert!(matches!(mutation.place, HirPlace::Binding(id) if id == binding));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::CreateBinding
    ));
    assert!(matches!(
        mutation.shape_policy,
        AssignmentShapePolicy::MatlabCompatible
    ));
}

#[test]
fn def_path_is_distinct_from_local_table_ids() {
    let function_id = FunctionId(0);
    let path = DefPath {
        package: PackageName("pkg".into()),
        module: QualifiedName(vec![SymbolName("pkg".into()), SymbolName("demo".into())]),
        item: vec![DefPathSegment::Function(SymbolName("f".into()))],
    };

    assert_eq!(function_id, FunctionId(0));
    assert_eq!(path.package.0, "pkg");
    assert!(matches!(path.item[0], DefPathSegment::Function(_)));
}

#[test]
fn imported_callable_identity_prefers_qualified_display_name() {
    let path = DefPath {
        package: PackageName("pkg".into()),
        module: QualifiedName(vec![
            SymbolName("pkg".into()),
            SymbolName("demo".into()),
            SymbolName("f".into()),
        ]),
        item: vec![DefPathSegment::Function(SymbolName("f".into()))],
    };
    let identity = CallableIdentity::Imported(path);
    assert_eq!(identity.display_name().as_deref(), Some("pkg.demo.f"));

    let empty_module_path = DefPath {
        package: PackageName("pkg".into()),
        module: QualifiedName(vec![]),
        item: vec![DefPathSegment::Function(SymbolName("f".into()))],
    };
    let empty_module_identity = CallableIdentity::Imported(empty_module_path);
    assert_eq!(empty_module_identity.display_name(), None);
}

#[test]
fn qualified_name_display_name_rejects_empty_segments() {
    let malformed = QualifiedName(vec![
        SymbolName("pkg".into()),
        SymbolName("".into()),
        SymbolName("remote".into()),
    ]);
    assert_eq!(malformed.display_name(), None);
}

#[test]
fn callable_identity_display_name_rejects_empty_name_fields() {
    assert_eq!(
        CallableIdentity::Builtin(BuiltinId(String::new())).display_name(),
        None
    );
    assert_eq!(
        CallableIdentity::Method(MethodId(String::new())).display_name(),
        None
    );
    assert_eq!(
        CallableIdentity::DynamicName(SymbolName(String::new())).display_name(),
        None
    );
}

#[test]
fn callable_name_fallback_policies_require_well_formed_external_names() {
    let dynamic = CallableIdentity::DynamicName(SymbolName("sqrt".into()));
    let imported = CallableIdentity::Imported(DefPath {
        package: PackageName("Point".into()),
        module: QualifiedName(vec![
            SymbolName("Point".into()),
            SymbolName("origin".into()),
        ]),
        item: vec![DefPathSegment::Function(SymbolName("origin".into()))],
    });
    let imported_missing_item = CallableIdentity::Imported(DefPath {
        package: PackageName("Point".into()),
        module: QualifiedName(vec![
            SymbolName("Point".into()),
            SymbolName("origin".into()),
        ]),
        item: vec![],
    });
    let imported_empty_item_name = CallableIdentity::Imported(DefPath {
        package: PackageName("Point".into()),
        module: QualifiedName(vec![
            SymbolName("Point".into()),
            SymbolName("origin".into()),
        ]),
        item: vec![DefPathSegment::Function(SymbolName("".into()))],
    });
    let imported_mismatched_item = CallableIdentity::Imported(DefPath {
        package: PackageName("Point".into()),
        module: QualifiedName(vec![
            SymbolName("Point".into()),
            SymbolName("origin".into()),
        ]),
        item: vec![DefPathSegment::Function(SymbolName("different".into()))],
    });
    let method = CallableIdentity::Method(MethodId("deal".into()));
    let whitespace_dynamic = CallableIdentity::DynamicName(SymbolName("   ".into()));
    let whitespace_method = CallableIdentity::Method(MethodId("   ".into()));
    let single_external =
        CallableIdentity::ExternalName(QualifiedName(vec![SymbolName("sqrt".into())]));
    let whitespace_external = CallableIdentity::ExternalName(QualifiedName(vec![
        SymbolName("OverIdx".into()),
        SymbolName("   ".into()),
    ]));
    let qualified_external = CallableIdentity::ExternalName(QualifiedName(vec![
        SymbolName("OverIdx".into()),
        SymbolName("plus".into()),
    ]));
    let malformed_external = CallableIdentity::ExternalName(QualifiedName(vec![
        SymbolName("OverIdx".into()),
        SymbolName("".into()),
        SymbolName("plus".into()),
    ]));

    assert!(
        CallableFallbackPolicy::RuntimeNameResolution.allows_semantic_name_resolution_for(&dynamic)
    );
    assert!(CallableFallbackPolicy::RuntimeNameResolution
        .allows_semantic_name_resolution_for(&imported));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_semantic_name_resolution_for(&imported_missing_item));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_semantic_name_resolution_for(&imported_empty_item_name));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_semantic_name_resolution_for(&imported_mismatched_item));
    assert!(
        CallableFallbackPolicy::RuntimeNameResolution.allows_semantic_name_resolution_for(&method)
    );
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_semantic_name_resolution_for(&whitespace_dynamic));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_semantic_name_resolution_for(&whitespace_method));
    assert!(!CallableFallbackPolicy::ExternalBoundary.allows_semantic_name_resolution_for(&dynamic));
    assert!(
        !CallableFallbackPolicy::ExternalBoundary.allows_semantic_name_resolution_for(&imported)
    );
    assert!(!CallableFallbackPolicy::ExternalBoundary.allows_semantic_name_resolution_for(&method));
    assert!(!CallableFallbackPolicy::ExternalBoundary
        .allows_semantic_name_resolution_for(&single_external));
    assert!(CallableFallbackPolicy::ExternalBoundary
        .allows_semantic_name_resolution_for(&qualified_external));
    assert!(!CallableFallbackPolicy::ExternalBoundary
        .allows_semantic_name_resolution_for(&whitespace_external));
    assert!(!CallableFallbackPolicy::ExternalBoundary
        .allows_semantic_name_resolution_for(&malformed_external));

    assert!(CallableFallbackPolicy::RuntimeNameResolution.allows_vm_name_fallback_for(&dynamic));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_vm_name_fallback_for(&whitespace_dynamic));
    assert!(CallableFallbackPolicy::RuntimeNameResolution.allows_vm_name_fallback_for(&imported));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_vm_name_fallback_for(&imported_missing_item));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_vm_name_fallback_for(&imported_empty_item_name));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution
        .allows_vm_name_fallback_for(&imported_mismatched_item));
    assert!(!CallableFallbackPolicy::RuntimeNameResolution.allows_vm_name_fallback_for(&method));
    assert!(!CallableFallbackPolicy::ExternalBoundary.allows_vm_name_fallback_for(&dynamic));
    assert!(!CallableFallbackPolicy::ExternalBoundary.allows_vm_name_fallback_for(&single_external));
    assert!(
        CallableFallbackPolicy::ExternalBoundary.allows_vm_name_fallback_for(&qualified_external)
    );
    assert!(
        !CallableFallbackPolicy::ExternalBoundary.allows_vm_name_fallback_for(&whitespace_external)
    );
    assert!(
        !CallableFallbackPolicy::ExternalBoundary.allows_vm_name_fallback_for(&malformed_external)
    );

    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.vm_fallback_name_for(&dynamic),
        Some("sqrt".into())
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.vm_fallback_name_for(&dynamic),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.vm_fallback_name_for(&whitespace_dynamic),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.vm_fallback_name_for(&imported),
        Some("Point.origin".into())
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.vm_fallback_name_for(&imported_missing_item),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution
            .vm_fallback_name_for(&imported_empty_item_name),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution
            .vm_fallback_name_for(&imported_mismatched_item),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.resolution_name_for(&imported_missing_item),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution
            .resolution_name_for(&imported_empty_item_name),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution
            .resolution_name_for(&imported_mismatched_item),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.vm_fallback_name_for(&method),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.resolution_name_for(&method),
        Some("deal".into())
    );
    assert_eq!(
        CallableFallbackPolicy::RuntimeNameResolution.resolution_name_for(&whitespace_method),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.vm_fallback_name_for(&single_external),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.resolution_name_for(&single_external),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.vm_fallback_name_for(&qualified_external),
        Some("OverIdx.plus".into())
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.resolution_name_for(&qualified_external),
        Some("OverIdx.plus".into())
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.vm_fallback_name_for(&whitespace_external),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.resolution_name_for(&whitespace_external),
        None
    );
    assert_eq!(
        CallableFallbackPolicy::ExternalBoundary.vm_fallback_name_for(&malformed_external),
        None
    );
}
