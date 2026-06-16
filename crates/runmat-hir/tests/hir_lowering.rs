use runmat_hir::{
    lower, AssignmentCreationPolicy, BindingRole, BindingStorage, CallKind, CommandArgument,
    FunctionKind, HirCallableRef, HirExprKind, HirPlace, HirStmtKind, IndexComponent, IndexKind,
    IndexResultContext, IndexingSemantics, LoweringContext, MemberAccess, OutputTarget,
    PlaceMutationKind, RequestedOutputCount, SourceUnitKind, WorkspaceVisibility,
};
use std::collections::HashMap;

fn lower_result(src: &str) -> runmat_hir::LoweringResult {
    let ast = runmat_parser::parse(src).unwrap();
    lower(&ast, &LoweringContext::empty()).unwrap()
}

fn lower_semantic(src: &str) -> runmat_hir::HirAssembly {
    lower_result(src).assembly
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
fn function_modifiers_lower_to_semantic_hir() {
    let assembly = lower_semantic("isolated async function y = f(x); y = x; end");
    let function = &assembly.functions[0];

    assert!(function.modifiers.isolated);
    assert!(function.modifiers.is_async);
}

#[test]
fn isolated_function_cannot_capture_outer_binding() {
    let ast = runmat_parser::parse(
        "function y = outer(x); isolated function z = inner(); z = x; end; y = 1; end",
    )
    .unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();

    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:IsolatedLexicalCaptureUnsupported")
    );
}

#[test]
fn mixed_script_arguments_block_default_kind_lowers() {
    let ast = runmat_parser::parse(
        r#"
        function y = typed(x)
            arguments
                x (1,1) double
            end
            y = x * 2;
        end
        r = typed(3);
        "#,
    )
    .unwrap();

    lower(&ast, &LoweringContext::empty()).expect("default arguments block kind lowers");
}

#[test]
fn arguments_block_unsupported_trailing_syntax_has_stable_identifier() {
    let ast = runmat_parser::parse(
        r#"
        function y = typed(x)
            arguments
                x (1,1) double < 10
            end
            y = x;
        end
        "#,
    )
    .unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();

    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn arguments_block_advanced_kind_has_stable_identifier() {
    let ast = runmat_parser::parse(
        r#"
        function y = typed(x, varargin)
            arguments (Repeating)
                varargin double
            end
            y = x;
        end
        "#,
    )
    .unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();

    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn arguments_block_name_value_declaration_has_stable_identifier() {
    let ast = runmat_parser::parse(
        r#"
        function y = typed(opts)
            arguments
                opts.Name (1,1) double = 1
            end
            y = opts.Name;
        end
        "#,
    )
    .unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();

    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:FunctionArgumentValidationUnsupported")
    );
}

#[test]
fn class_cannot_inherit_from_itself_identifier_contract() {
    let ast = runmat_parser::parse("classdef A < A; end").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();

    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassSelfInheritanceInvalid")
    );
}

#[test]
fn class_duplicate_property_identifier_contract() {
    let source = r#"
classdef A
    properties
        x
        x
    end
end
"#;
    let ast = runmat_parser::parse(source).unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassMemberDuplicate")
    );
}

#[test]
fn class_property_method_name_conflict_identifier_contract() {
    let source = r#"
classdef A
    properties
        x
    end
    methods
        function y = x(obj)
            y = 1;
        end
    end
end
"#;
    let ast = runmat_parser::parse(source).unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:ClassMemberNameConflict")
    );
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
fn name_value_call_arguments_lower_to_name_value_pairs() {
    let assembly =
        lower_semantic(r#"x = f(1, Mode="fast", Count=2); function y = f(varargin); y = 0; end"#);
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
    assert_eq!(call.args.len(), 5);
    assert!(matches!(call.args[0].kind, HirExprKind::Number(ref value) if value == "1"));
    assert!(matches!(call.args[1].kind, HirExprKind::String(ref value) if value.0 == "Mode"));
    assert!(matches!(call.args[2].kind, HirExprKind::String(ref value) if value.0 == "\"fast\""));
    assert!(matches!(call.args[3].kind, HirExprKind::String(ref value) if value.0 == "Count"));
    assert!(matches!(call.args[4].kind, HirExprKind::Number(ref value) if value == "2"));
}

#[test]
fn name_value_binding_call_lowers_to_dynamic_dispatch_not_indexing() {
    let assembly = lower_semantic("h = @sin; y = h(Name=1);");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let entry_function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::Assign(_, expr, _) = &entry_function.body.statements[1].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("name-value binding call should dispatch dynamically, not lower as indexing");
    };
    assert!(matches!(call.callee, HirCallableRef::DynamicExpr(_)));
    assert_eq!(call.args.len(), 2);
    assert!(matches!(call.args[0].kind, HirExprKind::String(ref value) if value.0 == "Name"));
    assert!(matches!(call.args[1].kind, HirExprKind::Number(ref value) if value == "1"));
}

#[test]
fn name_value_brace_value_lowers_as_single_value_not_argument_expansion() {
    let assembly =
        lower_semantic(r#"C = {7}; x = f(Name=C{:}); function y = f(varargin); y = 0; end"#);
    let function_id = assembly.modules[0].top_level_functions[0];
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let entry_function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::Assign(_, expr, _) = &entry_function.body.statements[1].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected call expression");
    };
    assert!(matches!(call.callee, HirCallableRef::Function(id) if id == function_id));
    assert_eq!(call.args.len(), 2);
    assert!(matches!(call.args[0].kind, HirExprKind::String(ref value) if value.0 == "Name"));
    let HirExprKind::Index(_, indexing) = &call.args[1].kind else {
        panic!("expected brace-index value expression");
    };
    assert!(matches!(indexing.kind, IndexKind::Brace));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::ReadCommaList
    ));
}

#[test]
fn unbound_builtin_identifier_lowers_to_zero_arg_builtin_call() {
    let assembly = lower_semantic("x = rand;");
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
        panic!("expected bare builtin identifier to become a call");
    };
    assert!(matches!(&call.callee, HirCallableRef::Builtin(id) if id.0 == "rand"));
    assert!(call.args.is_empty());
    assert_eq!(call.requested_outputs, RequestedOutputCount::One);
}

#[test]
fn local_binding_shadows_bare_builtin_identifier() {
    let assembly = lower_semantic("rand = 7; x = rand;");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let entry_function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::Assign(_, expr, _) = &entry_function.body.statements[1].kind else {
        panic!("expected assignment");
    };
    assert!(
        matches!(expr.kind, HirExprKind::Binding(_)),
        "local binding must keep precedence over builtin fallback"
    );
}

#[test]
fn bare_builtin_identifier_statement_preserves_requested_output_count() {
    let assembly = lower_semantic("rand;");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let entry_function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::ExprStmt(expr, suppressed) = &entry_function.body.statements[0].kind else {
        panic!("expected expression statement");
    };
    assert!(*suppressed);
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected bare builtin identifier to become a call");
    };
    assert!(matches!(&call.callee, HirCallableRef::Builtin(id) if id.0 == "rand"));
    assert!(
        call.bare_identifier,
        "bare builtin identifier calls should retain source syntax"
    );
    assert_eq!(call.requested_outputs, RequestedOutputCount::Zero);
}

#[test]
fn bare_builtin_identifier_after_external_visibility_uses_workspace_first_call() {
    let assembly = lower_semantic("run(\"setup.m\"); x = rand;");
    let expr = assembly
        .functions
        .iter()
        .flat_map(|function| function.body.statements.iter())
        .find_map(|stmt| match &stmt.kind {
            HirStmtKind::Assign(_, expr, _) => Some(expr),
            _ => None,
        })
        .expect("expected assignment after run");
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected bare builtin identifier to become a call");
    };
    assert!(matches!(&call.callee, HirCallableRef::Builtin(id) if id.0 == "rand"));
    assert!(
        call.bare_identifier,
        "bare builtin identifier calls should retain source syntax"
    );
    assert_eq!(
        call.workspace_first_name
            .as_ref()
            .map(|name| name.0.as_str()),
        Some("rand")
    );
}

#[test]
fn shadowed_workspace_loading_identifier_does_not_expose_external_bindings() {
    let ast = runmat_parser::parse("load = 7; load; x = missing_name;").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn user_function_named_workspace_loader_does_not_expose_external_bindings() {
    let ast =
        runmat_parser::parse("run(); x = missing_name; function run(); marker = 1; end").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn recursive_function_call_resolves_to_self_function_id() {
    let assembly = lower_semantic("function y = fact(n); y = fact(n - 1); end");
    let function_id = assembly.modules[0].top_level_functions[0];
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == function_id)
        .unwrap();

    let HirStmtKind::Assign(_, expr, _) = &function.body.statements[0].kind else {
        panic!("expected recursive assignment");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected recursive call expression");
    };
    assert!(matches!(call.callee, HirCallableRef::Function(id) if id == function_id));
}

#[test]
fn scoped_private_function_alias_resolves_only_for_owner_scope() {
    let mut ast = runmat_parser::parse(
        r#"
        leak = helper(1);
        function y = entry(x)
            a = helper(x);
            h = @helper;
            b = h(x);
            y = a + b;
        end
        function y = helper(x)
            y = x + 1;
        end
        "#,
    )
    .unwrap();
    for stmt in &mut ast.body {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            if name == "entry" {
                *name = "pkg.entry".to_string();
            } else if name == "helper" {
                *name = "pkg.__private__.helper".to_string();
            }
        }
    }
    let mut owners = HashMap::new();
    owners.insert("pkg.__private__.helper".to_string(), "pkg".to_string());
    let mut pkg_aliases = HashMap::new();
    pkg_aliases.insert("helper".to_string(), "pkg.__private__.helper".to_string());
    let mut aliases = HashMap::new();
    aliases.insert("pkg".to_string(), pkg_aliases);
    let result = lower(
        &ast,
        &LoweringContext::empty().with_private_functions(&owners, &aliases),
    )
    .unwrap();

    let private_function_id = result
        .assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "pkg.__private__.helper")
        .map(|function| function.id)
        .expect("private helper function");
    let entry = result
        .assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "pkg.entry")
        .expect("entry function");

    assert!(
        result.hir_index.calls.iter().any(|call| {
            call.name.0[0].0 == "helper"
                && matches!(call.kind, CallKind::Dynamic)
                && matches!(call.callee, HirCallableRef::Unresolved(_))
        }),
        "root scope should not resolve the package private helper"
    );
    assert!(
        result.hir_index.calls.iter().any(|call| {
            matches!(call.kind, CallKind::DirectFunction(id) if id == private_function_id)
        }),
        "owner scope should resolve direct helper calls to the package private helper"
    );

    let HirStmtKind::Assign(_, handle_expr, _) = &entry.body.statements[1].kind else {
        panic!("expected handle assignment");
    };
    assert!(
        matches!(
            &handle_expr.kind,
            HirExprKind::FunctionHandle(runmat_hir::FunctionHandleTarget::Function(id))
                if *id == private_function_id
        ),
        "owner scope should resolve @helper to the package private helper"
    );
}

#[test]
fn scoped_private_function_alias_resolves_for_class_folder_owner_scope() {
    let mut ast = runmat_parser::parse(
        r#"
        leak = helper(1);
        function y = entry(x)
            a = helper(x);
            h = @helper;
            b = h(x);
            y = a + b;
        end
        function y = helper(x)
            y = x + 1;
        end
        "#,
    )
    .unwrap();
    for stmt in &mut ast.body {
        if let runmat_parser::Stmt::Function { name, .. } = stmt {
            if name == "entry" {
                *name = "C.entry".to_string();
            } else if name == "helper" {
                *name = "C.__private__.helper".to_string();
            }
        }
    }
    let mut owners = HashMap::new();
    owners.insert("C.__private__.helper".to_string(), "C".to_string());
    let mut class_aliases = HashMap::new();
    class_aliases.insert("helper".to_string(), "C.__private__.helper".to_string());
    let mut aliases = HashMap::new();
    aliases.insert("C".to_string(), class_aliases);
    let result = lower(
        &ast,
        &LoweringContext::empty().with_private_functions(&owners, &aliases),
    )
    .unwrap();

    let private_function_id = result
        .assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "C.__private__.helper")
        .map(|function| function.id)
        .expect("private helper function");
    let entry = result
        .assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "C.entry")
        .expect("entry function");

    assert!(
        result.hir_index.calls.iter().any(|call| {
            call.name.0[0].0 == "helper"
                && matches!(call.kind, CallKind::Dynamic)
                && matches!(call.callee, HirCallableRef::Unresolved(_))
        }),
        "root scope should not resolve the class-folder private helper"
    );
    assert!(
        result.hir_index.calls.iter().any(|call| {
            matches!(call.kind, CallKind::DirectFunction(id) if id == private_function_id)
        }),
        "class owner scope should resolve direct helper calls to the class-folder private helper"
    );

    let HirStmtKind::Assign(_, handle_expr, _) = &entry.body.statements[1].kind else {
        panic!("expected handle assignment");
    };
    assert!(
        matches!(
            &handle_expr.kind,
            HirExprKind::FunctionHandle(runmat_hir::FunctionHandleTarget::Function(id))
                if *id == private_function_id
        ),
        "class owner scope should resolve @helper to the class-folder private helper"
    );
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
fn nested_same_name_function_does_not_replace_top_level_output_arity() {
    let assembly = lower_semantic(
        r#"
        function y = foo(x)
            y = x + 1;
        end

        function y = outer(x)
            function foo()
            end
            foo()
            y = x;
        end

        foo(2)
        "#,
    );
    let top_level_foo = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "foo" && function.parent.is_none())
        .unwrap();
    let outer = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "outer")
        .unwrap();
    let nested_foo = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "foo" && function.parent == Some(outer.id))
        .unwrap();

    let HirStmtKind::ExprStmt(nested_expr, false) = &outer.body.statements[0].kind else {
        panic!("expected unsuppressed nested call expression");
    };
    let HirExprKind::Call(nested_call) = &nested_expr.kind else {
        panic!("expected nested function call");
    };
    assert!(matches!(
        nested_call.callee,
        HirCallableRef::Function(id) if id == nested_foo.id
    ));
    assert_eq!(nested_call.requested_outputs, RequestedOutputCount::Zero);

    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let entry_function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let HirStmtKind::ExprStmt(entry_expr, false) = &entry_function.body.statements[0].kind else {
        panic!("expected unsuppressed top-level call expression");
    };
    let HirExprKind::Call(entry_call) = &entry_expr.kind else {
        panic!("expected top-level function call");
    };
    assert!(matches!(
        entry_call.callee,
        HirCallableRef::Function(id) if id == top_level_foo.id
    ));
    assert_eq!(entry_call.requested_outputs, RequestedOutputCount::One);
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
fn class_constructor_call_is_tagged_as_constructor_call_kind() {
    let result = lower_result(
        "classdef C\n methods\n  function obj = C(x)\n   obj.x = x;\n  end\n end\nend\nv = C(3);",
    );
    let kinds = result
        .hir_index
        .calls
        .iter()
        .map(|call| {
            format!(
                "{:?}:{}",
                call.kind,
                call.name.display_name().unwrap_or_default()
            )
        })
        .collect::<Vec<_>>();
    assert!(
        result
            .hir_index
            .calls
            .iter()
            .any(|call| matches!(call.kind, CallKind::Constructor(_))),
        "expected constructor call kind, got {:?}",
        kinds
    );
}

#[test]
fn class_inheritance_links_super_class_id() {
    let assembly = lower_semantic("classdef A\nend\nclassdef B < A\nend\nx = 1;");
    let class_a = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "A")
        .expect("class A");
    let class_b = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "B")
        .expect("class B");
    assert_eq!(class_b.super_class, Some(class_a.id));
}

#[test]
fn transitive_handle_inheritance_marks_handle_kind() {
    let assembly = lower_semantic(
        "classdef A < handle\nend\nclassdef B < A\nend\nclassdef C < B\nend\nx = 1;",
    );
    let class_b = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "B")
        .expect("class B");
    let class_c = assembly
        .classes
        .iter()
        .find(|class| class.name.0[0].0 == "C")
        .expect("class C");
    assert!(matches!(class_b.kind, runmat_hir::ClassKind::Handle));
    assert!(matches!(class_c.kind, runmat_hir::ClassKind::Handle));
}

#[test]
fn struct_aggregate_literal_lowers_with_field_order_and_duplicates() {
    let assembly = lower_semantic("s = struct{a = 1, a = 2, b = 3};");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .expect("entry function");

    let HirStmtKind::Assign(_, expr, _) = &function.body.statements[0].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::StructLiteral(fields) = &expr.kind else {
        panic!("expected struct literal");
    };
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].0 .0, "a");
    assert_eq!(fields[1].0 .0, "a");
    assert_eq!(fields[2].0 .0, "b");
}

#[test]
fn object_aggregate_literal_lowers_to_typed_object_literal() {
    let assembly = lower_semantic("p = ?Point{x = 1, y = 2};");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .expect("entry function");

    let HirStmtKind::Assign(_, expr, _) = &function.body.statements[0].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::ObjectLiteral { class_name, fields } = &expr.kind else {
        panic!("expected object literal");
    };
    assert_eq!(
        class_name
            .0
            .iter()
            .map(|segment| segment.0.as_str())
            .collect::<Vec<_>>(),
        vec!["Point"]
    );
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0 .0, "x");
    assert_eq!(fields[1].0 .0, "y");
}

#[test]
fn class_attributes_lower_to_semantic_metadata() {
    let assembly = lower_semantic(
        "classdef C\n properties(Constant, Hidden, Access=private)\n p\n end\n methods(Static, Access=private, Sealed)\n function y = f(x); y = x; end\n end\n end",
    );
    let class = &assembly.classes[0];

    let prop_attrs = &class.properties[0].attributes;
    assert!(prop_attrs.is_constant);
    assert!(prop_attrs.is_hidden);
    assert!(matches!(prop_attrs.access, MemberAccess::Private));
    assert!(matches!(prop_attrs.get_access, MemberAccess::Private));
    assert!(matches!(prop_attrs.set_access, MemberAccess::Private));

    let method = &class.methods[0];
    assert!(method.is_static);
    assert!(matches!(method.attributes.access, MemberAccess::Private));
    assert!(method.attributes.is_sealed);
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
    assert!(matches!(
        targets.requested_outputs,
        RequestedOutputCount::Exactly(2)
    ));
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
fn ordinary_builtin_args_do_not_infer_outputs_from_last_call_argument() {
    let result = lower_result(
        r#"
function [a,b] = pair(x)
  a = x;
  b = x + 1;
end
r = max(pair(3));
"#,
    );

    assert!(
        result.hir_index.calls.iter().any(|call| {
            call.name.0[0].0 == "pair"
                && matches!(call.requested_outputs, RequestedOutputCount::One)
        }),
        "calls: {:?}",
        result
            .hir_index
            .calls
            .iter()
            .map(|call| (call.name.display_name(), call.requested_outputs.clone()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn variadic_builtin_cat_does_not_infer_outputs_from_last_call_argument() {
    let result = lower_result(
        r#"
function varargout = g()
  varargout = {1, [2,3]};
end
r = cat(g());
"#,
    );

    assert!(
        result.hir_index.calls.iter().any(|call| {
            call.name.0[0].0 == "g" && matches!(call.requested_outputs, RequestedOutputCount::One)
        }),
        "calls: {:?}",
        result
            .hir_index
            .calls
            .iter()
            .map(|call| (call.name.display_name(), call.requested_outputs.clone()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn lowering_emits_only_fixed_requested_output_counts() {
    fn is_fixed(outputs: &RequestedOutputCount) -> bool {
        matches!(
            outputs,
            RequestedOutputCount::Zero
                | RequestedOutputCount::One
                | RequestedOutputCount::Exactly(_)
        )
    }

    fn walk_expr(expr: &runmat_hir::HirExpr) {
        match &expr.kind {
            HirExprKind::Call(call) => {
                assert!(
                    is_fixed(&call.requested_outputs),
                    "unexpected non-fixed requested outputs in lowered call: {:?}",
                    call.requested_outputs
                );
                for arg in &call.args {
                    walk_expr(arg);
                }
                if let HirCallableRef::DynamicExpr(callee) = &call.callee {
                    walk_expr(callee);
                }
            }
            HirExprKind::CommandCall(_) => {}
            HirExprKind::Unary(_, inner)
            | HirExprKind::Spawn(inner)
            | HirExprKind::Await(inner) => walk_expr(inner),
            HirExprKind::Binary(left, _, right) | HirExprKind::MemberDynamic(left, right) => {
                walk_expr(left);
                walk_expr(right);
            }
            HirExprKind::Range(start, step, end) => {
                walk_expr(start);
                if let Some(step) = step {
                    walk_expr(step);
                }
                walk_expr(end);
            }
            HirExprKind::Tensor(rows) | HirExprKind::Cell(rows) => {
                for row in rows {
                    for value in row {
                        walk_expr(value);
                    }
                }
            }
            HirExprKind::StructLiteral(fields) => {
                for (_, value) in fields {
                    walk_expr(value);
                }
            }
            HirExprKind::ObjectLiteral { fields, .. } => {
                for (_, value) in fields {
                    walk_expr(value);
                }
            }
            HirExprKind::Index(base, indexing) => {
                walk_expr(base);
                for component in &indexing.components {
                    if let IndexComponent::Expr(value) | IndexComponent::Logical(value) = component
                    {
                        walk_expr(value);
                    }
                }
            }
            HirExprKind::Member(base, _) => walk_expr(base),
            HirExprKind::FunctionHandle(_)
            | HirExprKind::AnonymousFunction(_)
            | HirExprKind::WorkspaceFirstStaticProperty { .. }
            | HirExprKind::MetaClass(_)
            | HirExprKind::Number(_)
            | HirExprKind::String(_)
            | HirExprKind::Constant(_)
            | HirExprKind::Binding(_)
            | HirExprKind::Colon
            | HirExprKind::End => {}
        }
    }

    fn walk_place(place: &HirPlace) {
        match place {
            HirPlace::Binding(_) => {}
            HirPlace::Member(base, _) => walk_expr(base),
            HirPlace::MemberDynamic(base, member) => {
                walk_expr(base);
                walk_expr(member);
            }
            HirPlace::Index(base, indexing) | HirPlace::IndexCell(base, indexing) => {
                walk_expr(base);
                for component in &indexing.components {
                    if let IndexComponent::Expr(value) | IndexComponent::Logical(value) = component
                    {
                        walk_expr(value);
                    }
                }
            }
        }
    }

    fn walk_stmt(stmt: &runmat_hir::HirStmt) {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, _) => walk_expr(expr),
            HirStmtKind::Assign(place, value, _) => {
                walk_place(place);
                walk_expr(value);
            }
            HirStmtKind::MultiAssign(targets, value, _) => {
                for target in &targets.targets {
                    if let OutputTarget::Place(place) = target {
                        walk_place(place);
                    }
                }
                walk_expr(value);
            }
            HirStmtKind::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
            } => {
                walk_expr(cond);
                for stmt in &then_body.statements {
                    walk_stmt(stmt);
                }
                for (cond, block) in elseif_blocks {
                    walk_expr(cond);
                    for stmt in &block.statements {
                        walk_stmt(stmt);
                    }
                }
                if let Some(block) = else_body {
                    for stmt in &block.statements {
                        walk_stmt(stmt);
                    }
                }
            }
            HirStmtKind::While { cond, body } => {
                walk_expr(cond);
                for stmt in &body.statements {
                    walk_stmt(stmt);
                }
            }
            HirStmtKind::For { range, body, .. } => {
                walk_expr(range);
                for stmt in &body.statements {
                    walk_stmt(stmt);
                }
            }
            HirStmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => {
                walk_expr(expr);
                for (case_expr, block) in cases {
                    walk_expr(case_expr);
                    for stmt in &block.statements {
                        walk_stmt(stmt);
                    }
                }
                if let Some(block) = otherwise {
                    for stmt in &block.statements {
                        walk_stmt(stmt);
                    }
                }
            }
            HirStmtKind::TryCatch {
                try_body,
                catch_body,
                ..
            } => {
                for stmt in &try_body.statements {
                    walk_stmt(stmt);
                }
                for stmt in &catch_body.statements {
                    walk_stmt(stmt);
                }
            }
            HirStmtKind::Global(_)
            | HirStmtKind::Persistent(_)
            | HirStmtKind::Break
            | HirStmtKind::Continue
            | HirStmtKind::Return
            | HirStmtKind::Import(_) => {}
        }
    }

    let assembly = lower_semantic(
        r#"
x = 1;
[~, idx] = max(x);
y = feval(@sin, x);
obj = 1;
z = obj.method(x);
function [a,b] = f(v)
  a = v;
  b = v + 1;
end
function y = g(h)
  y = h(1);
end
"#,
    );
    for function in &assembly.functions {
        for stmt in &function.body.statements {
            walk_stmt(stmt);
        }
    }
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
fn brace_index_in_call_argument_records_expansion_context() {
    let assembly = lower_semantic("function y = f(varargin); y = g(varargin{:}); end");
    let function = &assembly.functions[0];
    let HirStmtKind::Assign(_, expr, _) = &function.body.statements[0].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected call");
    };
    let HirExprKind::Index(_, indexing) = &call.args[0].kind else {
        panic!("expected brace index argument");
    };

    assert!(matches!(indexing.kind, IndexKind::Brace));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::FunctionArgumentExpansion
    ));
}

#[test]
fn brace_index_expression_records_comma_list_read_context() {
    let assembly = lower_semantic("C = {1}; x = C{:};");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let HirStmtKind::Assign(_, expr, _) = &function.body.statements[1].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Index(_, indexing) = &expr.kind else {
        panic!("expected brace index expression");
    };

    assert!(matches!(indexing.kind, IndexKind::Brace));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::ReadCommaList
    ));
}

#[test]
fn binding_call_with_expansion_lowers_to_dynamic_call_dispatch() {
    let assembly = lower_semantic("h = @sin; C = {0}; y = h(C{:});");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let HirStmtKind::Assign(_, expr, _) = &function.body.statements[2].kind else {
        panic!("expected assignment");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected dynamic call expression");
    };
    assert!(matches!(call.callee, HirCallableRef::DynamicExpr(_)));
    assert_eq!(call.args.len(), 1);
    let HirExprKind::Index(_, indexing) = &call.args[0].kind else {
        panic!("expected brace-index expansion argument");
    };
    assert!(matches!(indexing.kind, IndexKind::Brace));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::FunctionArgumentExpansion
    ));
}

#[test]
fn binding_call_with_multi_output_lowers_to_dynamic_call_dispatch() {
    let assembly = lower_semantic("h = @sin; [a, b] = h(1);");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let HirStmtKind::MultiAssign(targets, expr, _) = &function.body.statements[1].kind else {
        panic!("expected multi-assign");
    };
    let HirExprKind::Call(call) = &expr.kind else {
        panic!("expected dynamic call expression");
    };
    assert!(matches!(call.callee, HirCallableRef::DynamicExpr(_)));
    assert_eq!(targets.targets.len(), 2);
    assert!(matches!(
        call.requested_outputs,
        RequestedOutputCount::Exactly(2)
    ));
}

#[test]
fn expr_stmt_output_policy_controls_binding_call_lowering_shape() {
    let assembly = lower_semantic("h = @sin; h(1)\nh(1);");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();

    let HirStmtKind::ExprStmt(expr_unsuppressed, false) = &function.body.statements[1].kind else {
        panic!("expected unsuppressed expression statement");
    };
    let HirExprKind::Index(_, indexing) = &expr_unsuppressed.kind else {
        panic!("unsuppressed expression statement should lower as paren indexing");
    };
    assert!(matches!(indexing.kind, IndexKind::Paren));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::ReadSingle
    ));

    let HirStmtKind::ExprStmt(expr_suppressed, true) = &function.body.statements[2].kind else {
        panic!("expected semicolon-suppressed expression statement");
    };
    let HirExprKind::Call(call) = &expr_suppressed.kind else {
        panic!("suppressed expression statement should lower as dynamic call dispatch");
    };
    assert!(matches!(call.callee, HirCallableRef::DynamicExpr(_)));
    assert!(matches!(call.requested_outputs, RequestedOutputCount::Zero));
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

#[test]
fn await_requires_async_function_or_top_level_script() {
    let ast = runmat_parser::parse("function y = f(t); y = await(t); end").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:AwaitContextInvalid")
    );

    let assembly = lower_semantic("async function y = f(t); y = await(t); end");
    let function = &assembly.functions[0];
    let HirStmtKind::Assign(_, await_expr, _) = &function.body.statements[0].kind else {
        panic!("expected await assignment");
    };
    assert!(matches!(await_expr.kind, HirExprKind::Await(_)));
}

#[test]
fn lowering_policy_can_disable_top_level_await() {
    let ast = runmat_parser::parse("y = await(1);").unwrap();
    let err = lower(
        &ast,
        &LoweringContext::empty().with_top_level_await_enabled(false),
    )
    .unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:AwaitContextInvalid")
    );

    let ast = runmat_parser::parse("x = 1;").unwrap();
    let result = lower(
        &ast,
        &LoweringContext::empty().with_top_level_await_enabled(false),
    )
    .expect("lowering should still succeed");
    assert!(
        !result.assembly.entrypoints[0].policy.top_level_await,
        "entrypoint policy should reflect top-level await host policy"
    );
}

#[test]
fn strict_mode_disables_runmat_extension_calls() {
    let ast = runmat_parser::parse("t = spawn(f());").unwrap();
    let err = lower(
        &ast,
        &LoweringContext::empty().with_runmat_extensions_enabled(false),
    )
    .unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:SpawnExtensionDisabled")
    );

    let ast = runmat_parser::parse("y = await(1);").unwrap();
    let err = lower(
        &ast,
        &LoweringContext::empty().with_runmat_extensions_enabled(false),
    )
    .unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:AwaitExtensionDisabled")
    );
}

#[test]
fn undefined_variable_errors_have_stable_identifier() {
    let ast = runmat_parser::parse("y = x + 1;").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn ragged_tensor_literal_rejected_with_identifier() {
    let ast = runmat_parser::parse("x = [1 2; 3];").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:AggregateShapeMismatch")
    );
}

#[test]
fn ragged_cell_literal_rejected_with_identifier() {
    let ast = runmat_parser::parse("x = {1, 2; 3};").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:AggregateShapeMismatch")
    );
}

#[test]
fn spawn_rejects_anonymous_function_with_lexical_capture() {
    let ast = runmat_parser::parse("function y = f(x); y = spawn(@() x); end").unwrap();
    let err = lower(&ast, &LoweringContext::empty()).unwrap_err();
    assert_eq!(
        err.identifier.as_deref(),
        Some("RunMat:SpawnLexicalCaptureUnsupported")
    );
}

#[test]
fn index_records_bindings_functions_imports_and_calls() {
    let result = lower_result("import pkg.*; x = f(1); function y = f(a); y = a; end");
    let function_id = result.assembly.modules[0].top_level_functions[0];

    assert!(result
        .hir_index
        .imports
        .iter()
        .any(|import| import.import.path.0[0].0 == "pkg" && import.import.wildcard));
    assert!(result
        .hir_index
        .bindings
        .iter()
        .any(|binding| binding.name.0 == "x"));
    assert!(result
        .hir_index
        .functions
        .iter()
        .any(|function| function.name.0 == "f" && function.function == function_id));
    assert!(result.hir_index.calls.iter().any(|call| {
        matches!(call.kind, CallKind::DirectFunction(id) if id == function_id)
            && matches!(call.requested_outputs, RequestedOutputCount::One)
    }));
}

#[test]
fn index_records_unresolved_calls_explicitly() {
    let result = lower_result("x = definitely_missing(1);");

    assert!(result.hir_index.calls.iter().any(|call| {
        call.name.0[0].0 == "definitely_missing"
            && matches!(call.callee, HirCallableRef::Unresolved(_))
            && matches!(call.kind, CallKind::Dynamic)
    }));
}

#[test]
fn indexed_assignment_to_undefined_root_records_creation_policy() {
    let result = lower_result("x(3) = 10;");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::Index(_, _)))
        .unwrap();

    assert!(matches!(mutation.kind, PlaceMutationKind::IndexedAssign));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::CreateArrayByIndex
    ));
    let HirPlace::Index(base, indexing) = &mutation.place else {
        panic!("expected indexed place");
    };
    assert!(matches!(base.kind, HirExprKind::Binding(_)));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget
    ));
}

#[test]
fn assignment_indexing_preserves_colon_and_end_components() {
    let result = lower_result("A(:, end) = 0;");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::Index(_, _)))
        .unwrap();

    let HirPlace::Index(_, indexing) = &mutation.place else {
        panic!("expected indexed place");
    };
    assert!(matches!(indexing.components[0], IndexComponent::Colon));
    assert!(matches!(indexing.components[1], IndexComponent::End { .. }));
}

#[test]
fn empty_array_index_assignment_records_deletion_target() {
    let result = lower_result("A(2) = [];");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::Index(_, _)))
        .unwrap();

    assert!(matches!(mutation.kind, PlaceMutationKind::Delete));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::ExistingOnly
    ));
    let HirPlace::Index(_, indexing) = &mutation.place else {
        panic!("expected indexed place");
    };
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::DeletionTarget
    ));
}

#[test]
fn empty_array_indexed_member_assignment_records_deletion_target() {
    let result = lower_result("s.a = {1, 2, 3}; s.a(2) = [];");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| {
            matches!(mutation.kind, PlaceMutationKind::Delete)
                && matches!(mutation.place, HirPlace::Index(_, _))
        })
        .expect("expected indexed delete mutation");

    assert!(matches!(mutation.kind, PlaceMutationKind::Delete));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::ExistingOnly
    ));
    let HirPlace::Index(base, indexing) = &mutation.place else {
        panic!("expected indexed place");
    };
    assert!(matches!(&base.kind, HirExprKind::Member(_, _)));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::DeletionTarget
    ));
}

#[test]
fn empty_array_indexed_cell_content_assignment_records_deletion_target() {
    let result = lower_result("c = {{1, 2, 3}}; c{1}(2) = [];");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| {
            matches!(mutation.kind, PlaceMutationKind::Delete)
                && matches!(mutation.place, HirPlace::Index(_, _))
        })
        .expect("expected indexed delete mutation");

    assert!(matches!(mutation.kind, PlaceMutationKind::Delete));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::ExistingOnly
    ));
    let HirPlace::Index(base, indexing) = &mutation.place else {
        panic!("expected indexed place");
    };
    assert!(matches!(
        &base.kind,
        HirExprKind::Index(
            _,
            IndexingSemantics {
                kind: IndexKind::Brace,
                ..
            }
        )
    ));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::DeletionTarget
    ));
}

#[test]
fn empty_array_indexed_dynamic_member_assignment_records_deletion_target() {
    let result = lower_result("s.a = {1, 2, 3}; f = 'a'; s.(f)(2) = [];");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| {
            matches!(mutation.kind, PlaceMutationKind::Delete)
                && matches!(mutation.place, HirPlace::Index(_, _))
        })
        .expect("expected indexed delete mutation");

    assert!(matches!(mutation.kind, PlaceMutationKind::Delete));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::ExistingOnly
    ));
    let HirPlace::Index(base, indexing) = &mutation.place else {
        panic!("expected indexed place");
    };
    assert!(matches!(&base.kind, HirExprKind::MemberDynamic(_, _)));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::DeletionTarget
    ));
}

#[test]
fn empty_array_member_assignment_is_not_marked_as_delete() {
    let result = lower_result("s = struct('x', {1, 2}); s(2).x = [];");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::Member(_, _)))
        .expect("expected member mutation");

    assert!(matches!(mutation.kind, PlaceMutationKind::MemberAssign));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::CreateStructFieldPath
    ));
    let HirPlace::Member(base, _) = &mutation.place else {
        panic!("expected member place");
    };
    let HirExprKind::Index(_, indexing) = &base.kind else {
        panic!("expected indexed member base");
    };
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget
    ));
}

#[test]
fn empty_array_cell_content_assignment_is_not_marked_as_delete() {
    let result = lower_result("c = {1}; c{1} = [];");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::IndexCell(_, _)))
        .expect("expected cell content mutation");

    assert!(matches!(mutation.kind, PlaceMutationKind::CellAssign));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::CreateArrayByIndex
    ));
    let HirPlace::IndexCell(_, indexing) = &mutation.place else {
        panic!("expected cell content place");
    };
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget
    ));
}

#[test]
fn struct_field_path_assignment_records_member_creation_policy() {
    let result = lower_result("s.a.b = 1;");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::Member(_, _)))
        .unwrap();

    assert!(matches!(mutation.kind, PlaceMutationKind::MemberAssign));
    assert!(matches!(
        mutation.creation_policy,
        AssignmentCreationPolicy::CreateStructFieldPath
    ));
}

#[test]
fn member_assignment_indexed_base_uses_assignment_target_context() {
    let result = lower_result("s = struct('x', {1, 2}); s(2).x = 9;");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::Member(_, _)))
        .expect("expected member mutation");

    let HirPlace::Member(base, _) = &mutation.place else {
        panic!("expected member place");
    };
    let HirExprKind::Index(_, indexing) = &base.kind else {
        panic!("expected indexed assignment base");
    };
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget
    ));
}

#[test]
fn dynamic_member_assignment_indexed_base_uses_assignment_target_context() {
    let result = lower_result("s = struct('x', {1, 2}); f = 'x'; s(2).(f) = 9;");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::MemberDynamic(_, _)))
        .expect("expected dynamic member mutation");

    let HirPlace::MemberDynamic(base, _) = &mutation.place else {
        panic!("expected dynamic member place");
    };
    let HirExprKind::Index(_, indexing) = &base.kind else {
        panic!("expected indexed assignment base");
    };
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget
    ));
}

#[test]
fn dynamic_member_assignment_cell_indexed_base_uses_assignment_target_context() {
    let result = lower_result("c = {struct('x', 1)}; f = 'x'; c{1}.(f) = 9;");
    let mutation = result
        .hir_index
        .mutations
        .iter()
        .find(|mutation| matches!(mutation.place, HirPlace::MemberDynamic(_, _)))
        .expect("expected dynamic member mutation");

    let HirPlace::MemberDynamic(base, _) = &mutation.place else {
        panic!("expected dynamic member place");
    };
    let HirExprKind::Index(_, indexing) = &base.kind else {
        panic!("expected indexed assignment base");
    };
    assert!(matches!(indexing.kind, IndexKind::Brace));
    assert!(matches!(
        indexing.result_context,
        IndexResultContext::AssignmentTarget
    ));
}

#[test]
fn dynamic_member_expression_lowers_to_member_dynamic_expr() {
    let assembly = lower_semantic("s = struct('x', 1); f = 'x'; y = s.(f);");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let stmt = function
        .body
        .statements
        .iter()
        .rev()
        .find(|stmt| matches!(stmt.kind, HirStmtKind::Assign(_, _, _)))
        .expect("expected assignment statement");

    let HirStmtKind::Assign(_, value, _) = &stmt.kind else {
        panic!("expected assignment statement");
    };
    assert!(matches!(value.kind, HirExprKind::MemberDynamic(_, _)));
}

#[test]
fn indexed_dynamic_member_expression_lowers_to_indexcell_over_member_dynamic_expr() {
    let assembly = lower_semantic("s = struct('x', {1, 2, 3}); f = 'x'; y = s.(f){2};");
    let entry = assembly.modules[0].synthetic_entry_function.unwrap();
    let function = assembly
        .functions
        .iter()
        .find(|function| function.id == entry)
        .unwrap();
    let stmt = function
        .body
        .statements
        .iter()
        .rev()
        .find(|stmt| matches!(stmt.kind, HirStmtKind::Assign(_, _, _)))
        .expect("expected assignment statement");

    let HirStmtKind::Assign(_, value, _) = &stmt.kind else {
        panic!("expected assignment statement");
    };
    assert!(matches!(
        value.kind,
        HirExprKind::Index(ref base, ref indexing)
            if matches!(base.kind, HirExprKind::MemberDynamic(_, _))
                && matches!(indexing.kind, IndexKind::Brace)
    ));
}
