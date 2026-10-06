use super::*;
use runmat_execution::{Digest, FunctionBindingOrigin};
use runmat_hir::{FunctionAbi, FunctionId, FunctionKind, FunctionModifiers, FunctionName, Span};
use runmat_mir::{MirAssembly, MirBody, MirFunctionMetadata};
use runmat_types::{FunctionSourceRole, ProgramFunctionId, ProgramSourceId};
use runmat_vm::{Bytecode, FunctionBytecode, FunctionRegistry};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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
        function(inherited_function.0, inherited, 0),
        function(current_function.0, current, 1),
    ]));

    assert_eq!(
        compiled_function_owners(current, &mir, &bytecode).unwrap(),
        BTreeSet::from([entrypoint, current_function])
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
