use super::*;
use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_hir::{
    EntrypointId, FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, Span,
    WorkspaceExportPolicy,
};
use runmat_mir::{MirAssembly, MirBody, MirFunctionMetadata};
use runmat_types::{
    ClassIdentity, ClassMethodOwner, ClassMethodRole, FunctionSourceRole, MethodName,
    ProgramFunctionId, ProgramSourceId,
};
use runmat_vm::{
    Bytecode, FunctionBytecode, FunctionRegistry, VmAssemblyLayout, VmEntrypointLayout, VmFrameAbi,
    VmFunctionLayout,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[test]
fn preferred_source_name_uses_only_registry_alias_of_selected_body() {
    let mut session = crate::RunMatSession::with_options(false, false).unwrap();
    let mut unit = futures::executor::block_on(session.compile_executable_unit(
        crate::ExecutableSource::new(
            "registry-alias@1",
            "pulse.m",
            "function y=pulse(x)\ny=x+1;\nend\nfunction y=other(x)\ny=x+2;\nend\n",
        ),
        None,
    ))
    .unwrap();
    let selected = unit.native_function_id("pulse").unwrap();
    let other = unit.native_function_id("other").unwrap();
    let registry = Rc::make_mut(&mut unit.functions);
    registry.names.retain(|_, function| *function != selected);
    registry.names.insert("loads.pulse".into(), selected);
    registry.names.insert("zzz.pulse".into(), selected);
    registry.names.insert("aaa.other".into(), other);
    let super::super::portable_entrypoint::PortableEntrypoint::Callable {
        name,
        compiled_function,
        ..
    } = super::super::portable_entrypoint::resolve(&unit, Some("pulse")).unwrap()
    else {
        panic!("selected function")
    };
    assert_eq!(name, "loads.pulse");
    assert_eq!(compiled_function, selected);
    Rc::make_mut(&mut unit.functions)
        .names
        .retain(|_, function| *function != selected);
    assert!(
        super::super::portable_entrypoint::resolve(&unit, Some("pulse"))
            .err()
            .expect("unrelated aliases cannot select a body")
            .contains("no exact admitted registry name")
    );
}

fn body(function: FunctionId) -> MirBody {
    MirBody {
        function,
        abi: FunctionAbi {
            fixed_inputs: Vec::new(),
            varargin: None,
            fixed_outputs: Vec::new(),
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        locals: Vec::new(),
        blocks: Vec::new(),
        catch_scopes: BTreeMap::new(),
    }
}

fn declaration(name: &str, kind: FunctionKind) -> MirFunctionMetadata {
    MirFunctionMetadata {
        source: ProgramSourceId(0),
        name: FunctionName(name.into()),
        parent: None,
        enclosing_class: None,
        class_method_owner: None,
        kind,
        source_role: FunctionSourceRole::Local,
        argument_validations: Vec::new(),
        captures: Vec::new(),
        modifiers: FunctionModifiers::default(),
        span: Span { start: 0, end: 1 },
    }
}

#[test]
fn constructor_class_entrypoint_uses_exact_typed_owner_only() {
    let constructor = FunctionId(1);
    let ordinary = FunctionId(2);
    let duplicate = FunctionId(3);
    let class = ClassIdentity::new("pkg.Seed").unwrap();
    let mut mir = MirAssembly::default();
    let mut constructor_metadata = declaration(
        "pkg.Seed.Seed",
        FunctionKind::ClassMethod {
            is_static: false,
            role: ClassMethodRole::Constructor,
        },
    );
    constructor_metadata.class_method_owner = Some(ClassMethodOwner {
        declaring_class: class.clone(),
        method: MethodName::from("Seed"),
        is_static: false,
        role: ClassMethodRole::Constructor,
    });
    mir.functions
        .insert(constructor, constructor_metadata.clone());
    let mut ordinary_metadata = declaration(
        "pkg.Seed.bump",
        FunctionKind::ClassMethod {
            is_static: false,
            role: ClassMethodRole::Ordinary,
        },
    );
    ordinary_metadata.class_method_owner = Some(ClassMethodOwner {
        declaring_class: class.clone(),
        method: MethodName::from("bump"),
        is_static: false,
        role: ClassMethodRole::Ordinary,
    });
    mir.functions.insert(ordinary, ordinary_metadata);

    assert_eq!(
        native_function_id_in_mir(&mir, "pkg.Seed"),
        Some(constructor)
    );
    assert_eq!(
        native_function_id_in_mir(&mir, "pkg.Seed.Seed"),
        Some(constructor)
    );
    assert_eq!(
        native_function_id_in_mir(&mir, "pkg.Seed.bump"),
        Some(ordinary)
    );
    assert_eq!(native_function_id_in_mir(&mir, "bump"), None);
    assert_eq!(native_function_id_in_mir(&mir, "pkg.Other"), None);
    assert_eq!(
        constructor_function_id_in_mir(&mir, &class, Some(ProgramSourceId(0))),
        Some(constructor)
    );

    mir.functions.get_mut(&constructor).unwrap().source = ProgramSourceId(1);
    assert_eq!(
        constructor_function_id_in_mir(&mir, &class, Some(ProgramSourceId(0))),
        None,
        "a companion constructor cannot replace the primary class source"
    );
    mir.functions.get_mut(&constructor).unwrap().source = ProgramSourceId(0);

    mir.functions.insert(duplicate, constructor_metadata);
    assert_eq!(native_function_id_in_mir(&mir, "pkg.Seed"), None);
    assert_eq!(
        constructor_function_id_in_mir(&mir, &class, Some(ProgramSourceId(0))),
        None,
        "ambiguous primary constructors cannot select an arbitrary body"
    );
    assert_eq!(
        native_function_id_in_mir(&mir, "pkg.Seed.Seed"),
        Some(constructor)
    );

    mir.functions.remove(&duplicate);
    let constructor_metadata = mir.functions.get_mut(&constructor).unwrap();
    constructor_metadata.kind = FunctionKind::ClassMethod {
        is_static: true,
        role: ClassMethodRole::Constructor,
    };
    constructor_metadata
        .class_method_owner
        .as_mut()
        .unwrap()
        .is_static = true;
    assert_eq!(native_function_id_in_mir(&mir, "pkg.Seed"), None);
    assert_eq!(
        constructor_function_id_in_mir(&mir, &class, Some(ProgramSourceId(0))),
        None,
        "static methods cannot own a source constructor entrypoint"
    );
    assert_eq!(
        native_function_id_in_mir(&mir, "pkg.Seed.Seed"),
        Some(constructor)
    );
}

fn function(id: usize, program: Digest, portable: u32) -> (FunctionId, FunctionBytecode) {
    let function = FunctionId(id);
    (
        function,
        FunctionBytecode {
            function,
            portable_function: ProgramFunctionId(portable),
            binding_origin: Some(FunctionBindingOrigin {
                program,
                function: ProgramFunctionId(portable),
            }),
            ..FunctionBytecode::default()
        },
    )
}

#[test]
fn compiled_owners_exclude_inherited_mir_bodies_and_declarations() {
    let current = Digest::sha256(b"current-program");
    let inherited = Digest::sha256(b"inherited-program");
    let entrypoint = FunctionId(0);
    let inherited_function = FunctionId(7);
    let current_function = FunctionId(8);
    let mut mir = MirAssembly::default();
    mir.entrypoints.push(entrypoint);
    mir.bodies.insert(entrypoint, body(entrypoint));
    mir.functions.insert(
        entrypoint,
        declaration("entrypoint", FunctionKind::SyntheticEntrypoint),
    );
    mir.bodies
        .insert(inherited_function, body(inherited_function));
    mir.functions.insert(
        inherited_function,
        declaration("inherited", FunctionKind::Named),
    );
    mir.bodies.insert(current_function, body(current_function));
    mir.functions.insert(
        current_function,
        declaration("current", FunctionKind::Named),
    );
    let mut bytecode = Bytecode::empty();
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([
        function(entrypoint.0, inherited, 9),
        function(inherited_function.0, inherited, 0),
        function(current_function.0, current, 1),
    ]));
    bytecode.bound_functions = bytecode.function_registry.functions.clone();
    bytecode.layout = Some(VmAssemblyLayout {
        functions: HashMap::from([(
            entrypoint,
            VmFunctionLayout {
                function: entrypoint,
                parent: None,
                display_name: "entrypoint".into(),
                source_role: FunctionSourceRole::Local,
                private_owner_scope: String::new(),
                frame_abi: VmFrameAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                binding_slots: HashMap::new(),
                mir_local_slots: HashMap::new(),
                captures: Vec::new(),
                local_count: 0,
                resume_points: BTreeMap::new(),
            },
        )]),
        entrypoints: HashMap::from([(
            EntrypointId(0),
            VmEntrypointLayout {
                entrypoint: EntrypointId(0),
                target: entrypoint,
                workspace_export: WorkspaceExportPolicy::HostDefined,
                exports: Vec::new(),
            },
        )]),
        storage_bindings: HashMap::new(),
    });

    let owners = compiled_function_owners(current, &mir, &bytecode).unwrap();
    assert_eq!(
        owners.assembly,
        BTreeSet::from([entrypoint, current_function])
    );
    assert_eq!(owners.registered_bodies, BTreeSet::from([current_function]));
    let analysis = runmat_mir::analysis::analyze_assembly(&mir);
    let analysis_owners = owners.analysis_owners(&analysis, &bytecode).unwrap();
    assert_eq!(analysis_owners[&entrypoint], ProgramFunctionId(0));
    assert_eq!(analysis_owners[&current_function], ProgramFunctionId(1));
    let projected = analysis
        .project_product_functions(&analysis_owners)
        .unwrap();
    assert!(projected.function_for_local(inherited_function).is_none());
    assert_eq!(
        projected
            .function_for_local(current_function)
            .unwrap()
            .function,
        ProgramFunctionId(1)
    );
    let old = bytecode.function_registry.functions[&entrypoint].clone();
    bytecode
        .install_regions_for_product(&[], &analysis_owners, &owners.registered_bodies)
        .unwrap();
    bytecode
        .install_parallel_regions_for_product(
            &mir,
            &runmat_types::ParallelManifest::empty(),
            &analysis_owners,
            &owners.registered_bodies,
        )
        .unwrap();
    assert_eq!(
        bytecode.function_registry.functions[&entrypoint].binding_origin,
        old.binding_origin
    );
    assert_eq!(
        bytecode.bound_functions[&entrypoint].binding_origin,
        old.binding_origin
    );
}

#[test]
fn compiled_owner_requires_current_function_origin() {
    let current = Digest::sha256(b"current-program");
    let function = FunctionId(4);
    let mut mir = MirAssembly::default();
    mir.bodies.insert(function, body(function));
    mir.functions
        .insert(function, declaration("unstamped", FunctionKind::Named));
    let mut bytecode = Bytecode::empty();
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([(
        function,
        FunctionBytecode {
            function,
            ..FunctionBytecode::default()
        },
    )]));

    assert!(compiled_function_owners(current, &mir, &bytecode)
        .unwrap_err()
        .contains("no immutable binding origin"));
}

#[test]
fn unregistered_named_entrypoint_is_not_compilation_authority() {
    let current = Digest::sha256(b"current-program");
    let entrypoint = FunctionId(3);
    let mut mir = MirAssembly::default();
    mir.entrypoints.push(entrypoint);
    mir.bodies.insert(entrypoint, body(entrypoint));
    mir.functions
        .insert(entrypoint, declaration("named", FunctionKind::Named));

    assert!(compiled_function_owners(current, &mir, &Bytecode::empty())
        .unwrap_err()
        .contains("is not synthetic"));
}

#[test]
fn named_entrypoint_requires_current_program_origin() {
    let current = Digest::sha256(b"current-program");
    let entrypoint = FunctionId(3);
    let mut mir = MirAssembly::default();
    mir.entrypoints.push(entrypoint);
    mir.bodies.insert(entrypoint, body(entrypoint));
    mir.functions
        .insert(entrypoint, declaration("named", FunctionKind::Named));
    let mut bytecode = Bytecode::empty();
    bytecode.function_registry = FunctionRegistry::new(HashMap::from([function(
        entrypoint.0,
        Digest::sha256(b"foreign-program"),
        0,
    )]));

    assert!(compiled_function_owners(current, &mir, &bytecode)
        .unwrap_err()
        .contains("different immutable program"));
}
