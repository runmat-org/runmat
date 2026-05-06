use runmat_hir::{
    lower, BindingRole, BindingStorage, CommandArgument, FunctionKind, HirCallableRef, HirExprKind,
    HirStmtKind, LoweringContext, OutputTarget, RequestedOutputCount, SourceUnitKind,
    WorkspaceVisibility,
};

fn lower_semantic(src: &str) -> runmat_hir::HirAssembly {
    let ast = runmat_parser::parse(src).unwrap();
    lower(&ast, &LoweringContext::empty()).unwrap().assembly
}

#[test]
fn script_lowers_to_module_entrypoint_and_workspace_bindings() {
    let assembly = lower_semantic("x = 1; y = x + 2;");

    assert_eq!(assembly.modules.len(), 1);
    assert!(matches!(
        assembly.modules[0].source_unit,
        SourceUnitKind::ScriptFile
    ));
    let entry_function = assembly.modules[0].synthetic_entry_function.unwrap();
    assert_eq!(assembly.entrypoints[0].target, entry_function);

    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry_function)
        .unwrap();
    assert!(matches!(function.kind, FunctionKind::SyntheticEntrypoint));
    assert_eq!(function.body.statements.len(), 2);
    assert!(assembly.bindings.iter().any(|binding| {
        binding.name.0 == "x"
            && matches!(binding.workspace_visibility, WorkspaceVisibility::TopLevel)
    }));
}

#[test]
fn top_level_function_lowers_to_module_owned_function_with_bindings() {
    let assembly = lower_semantic("function y = f(x); y = x; end");

    assert!(assembly.modules[0].synthetic_entry_function.is_none());
    assert_eq!(assembly.modules[0].top_level_functions.len(), 1);
    let function_id = assembly.modules[0].top_level_functions[0];
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == function_id)
        .unwrap();

    assert_eq!(function.name.0, "f");
    assert_eq!(function.params.len(), 1);
    assert_eq!(function.outputs.len(), 1);
    assert_eq!(function.abi.fixed_inputs, function.params);
    assert_eq!(function.abi.fixed_outputs, function.outputs);
    assert!(function.abi.implicit_nargin.is_some());
    assert!(function.abi.implicit_nargout.is_some());
    assert!(assembly
        .bindings
        .iter()
        .any(|binding| binding.id == function.params[0]
            && matches!(binding.role, BindingRole::Parameter)));
}

#[test]
fn shared_input_output_name_reuses_one_binding() {
    let assembly = lower_semantic("function x = bump(x); x = x + 1; end");
    let function = &assembly.functions[0];

    assert_eq!(function.params, function.outputs);
    assert_eq!(function.abi.fixed_inputs, function.abi.fixed_outputs);
}

#[test]
fn imports_attach_to_module_and_preserve_source_statement() {
    let assembly = lower_semantic("import pkg.*; x = 1;");

    assert_eq!(assembly.modules[0].imports.len(), 1);
    assert_eq!(assembly.modules[0].imports[0].path.0[0].0, "pkg");
    assert!(assembly.modules[0].imports[0].wildcard);

    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    assert!(matches!(
        function.body.statements[0].kind,
        HirStmtKind::Import(_)
    ));
}

#[test]
fn local_function_call_resolves_to_function_id() {
    let assembly = lower_semantic("x = f(1); function y = f(a); y = a; end");
    let function_id = assembly.modules[0].top_level_functions[0];
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let entry_function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::Assign(_, expr, _) = &entry_function.body.statements[0].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected call expression");
    };
    assert!(matches!(call.callee, HirCallableRef::Function(id) if id == function_id));
}

#[test]
fn nested_function_records_capture_of_parent_binding() {
    let assembly = lower_semantic(
        "function y = outer(a); acc = 0; function bump(x); acc = acc + x; end; bump(a); y = acc; end",
    );
    let outer = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "outer")
        .unwrap();
    let bump = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "bump")
        .unwrap();
    let acc = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "acc")
        .unwrap();

    assert_eq!(bump.parent, Some(outer.id));
    assert!(bump
        .captures
        .iter()
        .any(|capture| capture.binding == acc.id && capture.from_function == outer.id));
}

#[test]
fn anonymous_function_lowers_to_real_function_with_capture() {
    let assembly = lower_semantic("function f = outer(a); f = @(x) x + a; end");
    let outer = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "outer")
        .unwrap();
    let anon = assembly
        .functions
        .iter()
        .find(|function| matches!(function.kind, FunctionKind::Anonymous))
        .unwrap();
    let a = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "a")
        .unwrap();

    assert_eq!(anon.parent, Some(outer.id));
    assert!(anon
        .captures
        .iter()
        .any(|capture| capture.binding == a.id && capture.from_function == outer.id));
}

#[test]
fn class_method_lowers_to_function_referenced_by_class() {
    let assembly = lower_semantic(
        "classdef C\n properties\n p\n end\n methods\n function y = f(obj, x); y = x; end\n end\n end",
    );

    assert_eq!(assembly.classes.len(), 1);
    let class = &assembly.classes[0];
    assert_eq!(class.name.0[0].0, "C");
    assert_eq!(class.properties[0].name.0, "p");
    assert_eq!(class.methods.len(), 1);
    let method_function = assembly
        .functions
        .iter()
        .find(|function| function.id == class.methods[0].function)
        .unwrap();
    assert_eq!(method_function.enclosing_class, Some(class.id));
    assert!(matches!(
        method_function.kind,
        FunctionKind::ClassMethod { is_static: false }
    ));
}

#[test]
fn varargin_varargout_populate_function_abi() {
    let assembly = lower_semantic("function varargout = f(x, varargin); varargout = varargin; end");
    let function = &assembly.functions[0];

    assert_eq!(function.abi.fixed_inputs.len(), 2);
    assert_eq!(function.abi.varargin, Some(function.abi.fixed_inputs[1]));
    assert_eq!(function.abi.fixed_outputs.len(), 1);
    assert_eq!(function.abi.varargout, Some(function.abi.fixed_outputs[0]));
}

#[test]
fn multi_output_assignment_records_requested_outputs_and_discard() {
    let assembly = lower_semantic("x = 1; [~, idx] = max(x);");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::MultiAssign(targets, expr, _) = &function.body.statements[1].kind else {
        panic!("expected multi-assignment");
    };
    assert!(matches!(targets.targets[0], OutputTarget::Discard));
    assert!(matches!(targets.targets[1], OutputTarget::Place(_)));
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected call");
    };
    assert!(matches!(
        call.requested_outputs,
        RequestedOutputCount::Exactly(2)
    ));
}

#[test]
fn global_and_persistent_declarations_set_binding_storage() {
    let assembly = lower_semantic("global g; persistent p; g = 1; p = 2;");

    let global = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "g")
        .unwrap();
    let persistent = assembly
        .bindings
        .iter()
        .find(|binding| binding.name.0 == "p")
        .unwrap();

    assert!(matches!(global.storage, BindingStorage::Global));
    assert!(matches!(persistent.storage, BindingStorage::Persistent));
}

#[test]
fn command_syntax_lowers_to_semantic_command_call() {
    let assembly = lower_semantic("disp hello");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::ExprStmt(expr, _) = &function.body.statements[0].kind else {
        panic!("expected expression statement");
    };
    let HirExprKind::CommandCall(call) = &expr.kind else {
        panic!("expected semantic command call");
    };
    match &call.command {
        HirCallableRef::Builtin(id) => assert_eq!(id.0, "disp"),
        HirCallableRef::Unresolved(name) => assert_eq!(name.0[0].0, "disp"),
        other => panic!("unexpected command callee: {other:?}"),
    }
    assert!(matches!(call.args[0], CommandArgument::Word(ref word) if word.0 == "hello"));
}

#[test]
fn await_and_spawn_lower_to_explicit_semantic_forms() {
    let assembly = lower_semantic("f = fetch(); t = spawn(f); y = await(t);");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::Assign(_, spawn_expr, _) = &function.body.statements[1].kind else {
        panic!("expected spawn assignment");
    };
    assert!(matches!(spawn_expr.kind, HirExprKind::Spawn(_)));

    let HirStmtKind::Assign(_, await_expr, _) = &function.body.statements[2].kind else {
        panic!("expected await assignment");
    };
    assert!(matches!(await_expr.kind, HirExprKind::Await(_)));
}
