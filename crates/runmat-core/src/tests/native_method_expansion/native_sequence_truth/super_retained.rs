use super::*;
use crate::InvocationControl;

#[test]
fn compiled_derived_constructor_keeps_base_body_for_interpreted_and_native_entries() {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = crate::tests::cwd_lock();
        let directory = tempfile::TempDir::new().expect("compiled super source directory");
        let base = "classdef CompiledBase < handle\n methods\n function obj = CompiledBase(v)\n global compiled_super_witness;\n compiled_super_witness = obj;\n obj.baseValue = v;\n end\n end\n properties\n baseValue\n end\nend";
        let derived = "classdef CompiledDerived < CompiledBase\n methods\n function obj = CompiledDerived(v)\n obj@CompiledBase(v);\n obj.derivedValue = v + 1;\n end\n end\n properties\n derivedValue\n end\nend";
        std::fs::write(directory.path().join("CompiledBase.m"), base).expect("base source");
        std::fs::write(directory.path().join("CompiledDerived.m"), derived)
            .expect("derived source");
        let driver = "function obj = makeCompiled(v)\n obj = CompiledDerived(v);\nend";
        std::fs::write(directory.path().join("makeCompiled.m"), driver).expect("driver source");
        let _cwd = crate::tests::push_cwd(directory.path());
        let mut session = RunMatSession::with_options(false, false).expect("compiled session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::frozen(
                "compiled-super@1",
                "makeCompiled.m",
                directory.path().join("makeCompiled.m").to_string_lossy(),
                driver,
            ),
            None,
        ))
        .expect("compile derived product");
        let primary = unit.bytecode().source_id.unwrap().program().unwrap();
        assert_eq!(unit.mir().classes.len(), 2);
        assert!(unit
            .mir()
            .classes
            .iter()
            .all(|class| class.source != primary));
        assert!(
            unit.primary_declaration_only_class(primary)
                .unwrap()
                .is_none(),
            "companion registration prologues cannot publish a primary declaration"
        );
        let envelope = unit
            .portable_envelope()
            .expect("source-owned portable function");
        let cold = crate::ExecutableUnit::from_portable_envelope(&envelope)
            .expect("cold class-source round trip");
        assert_eq!(cold.mir(), unit.mir());
        source_bound_constructor_owners(&unit, session.runtime_context());
        let mut rebound = unit.mir().clone();
        let remap = rebound
            .functions
            .keys()
            .map(|function| {
                (
                    *function,
                    runmat_hir::FunctionId(function.0.checked_add(1000).unwrap()),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        runmat_mir::remap_closed_function_ids(&mut rebound, &remap).expect("closed function remap");
        assert!(rebound
            .classes
            .iter()
            .zip(&unit.mir().classes)
            .all(|(rebound, original)| rebound.source == original.source));
        rebound
            .validate_class_sources(|source| {
                unit.source_map()
                    .entries()
                    .iter()
                    .any(|entry| entry.source_id == source)
            })
            .unwrap();
        unit.analysis()
            .rebind_closed_functions(&remap)
            .unwrap()
            .validate_class_sources(&rebound)
            .unwrap();
        let registry = unit.bytecode().function_registry();
        let caller = registry
            .functions
            .values()
            .find(|body| {
                body.class_method_owner.as_ref().is_some_and(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor
                        && owner.declaring_class.display_name() == "CompiledDerived"
                })
            })
            .expect("typed derived constructor");
        let targets = registry
            .retained_super_constructor_targets_for(caller.function)
            .expect("complete selected superclass graph");
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].owner.declaring_class.display_name(),
            "CompiledBase"
        );
        for (value, control) in [
            (4.0, InvocationControl::default()),
            (8.0, InvocationControl::default().force_generic_native()),
        ] {
            let outputs = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Function("makeCompiled".into()),
                    arguments: crate::ProcedureArguments::Source(vec![Value::Num(value)]),
                    requested_outputs: 1,
                },
                &control,
            ))
            .expect("compiled derived invocation");
            let [Value::HandleObject(returned)] = outputs.as_slice() else {
                panic!("constructor must return one managed derived instance")
            };
            let Some(Value::HandleObject(observed)) = session
                .runtime_context()
                .session_global_value("compiled_super_witness")
                .expect("session global authority")
            else {
                panic!("compiled base body did not execute")
            };
            assert_eq!(returned.target, observed.target);
            let Some(Value::Object(properties)) = runmat_gc::gc_clone_value(&returned.target).ok()
            else {
                panic!("derived instance was not live after superclass call")
            };
            assert_eq!(
                properties.properties().get("baseValue"),
                Some(&Value::Num(value))
            );
            assert_eq!(
                properties.properties().get("derivedValue"),
                Some(&Value::Num(value + 1.0))
            );
        }
    });
}

fn source_bound_constructor_owners(
    unit: &crate::ExecutableUnit,
    runtime: &runmat_runtime::context::RuntimeContext,
) {
    use std::sync::Arc;
    let mut issued = 0;
    for owner in std::iter::once(unit).chain(
        unit.original_dependencies()
            .values()
            .map(|unit| unit.as_ref()),
    ) {
        let program = owner.revision().program_revision.identity_digest().unwrap();
        let selected_runtime = owner.bind_runtime_sources(runtime.clone()).unwrap();
        let mut first = owner.runtime_bytecode(&selected_runtime).unwrap();
        let _first_admission = owner
            .materialize_runtime_constructor_definitions(
                &mut first.function_registry,
                Some(&mut first.bound_functions),
            )
            .unwrap();
        let mut second = owner.runtime_bytecode(&selected_runtime).unwrap();
        let _second_admission = owner
            .materialize_runtime_constructor_definitions(
                &mut second.function_registry,
                Some(&mut second.bound_functions),
            )
            .unwrap();
        for (id, original) in &owner.functions().functions {
            let a = first.function_registry.get_current(*id).unwrap();
            let b = second.function_registry.get_current(*id).unwrap();
            let newly_issued = original
                .binding_origin
                .is_some_and(|origin| origin.program == program)
                && original.class_method_owner.as_ref().is_some_and(|method| {
                    method.role == runmat_types::ClassMethodRole::Constructor && !method.is_static
                })
                && original
                    .source_id
                    .is_some_and(|source| source.program().is_some());
            if newly_issued {
                issued += 1;
                assert!(!Arc::ptr_eq(
                    &original.definition_owner,
                    &a.definition_owner
                ));
                assert!(!Arc::ptr_eq(&a.definition_owner, &b.definition_owner));
                assert!(Arc::ptr_eq(
                    &a.definition_owner,
                    &first.bound_functions[id].definition_owner
                ));
                assert!(original
                    .definition_owner
                    .selected_receiver_declarations()
                    .is_none());
            } else {
                assert!(Arc::ptr_eq(&original.definition_owner, &a.definition_owner));
                assert!(Arc::ptr_eq(&original.definition_owner, &b.definition_owner));
            }
        }
    }
    assert!(
        issued >= 2,
        "both original class constructors must be source-materialized"
    );
}

#[test]
fn retained_base_constructor_executes_and_merges_on_repeated_derived_calls() {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = crate::tests::cwd_lock();
        let directory = tempfile::TempDir::new().expect("super constructor source directory");
        std::fs::write(
            directory.path().join("RetainedBase.m"),
            "classdef RetainedBase < handle\n properties\n baseValue\n end\n methods\n function obj = RetainedBase(v)\n global retained_base_witness;\n retained_base_witness = obj;\n obj.baseValue = v;\n end\n end\nend",
        )
        .expect("write base class");
        std::fs::write(
            directory.path().join("RetainedDerived.m"),
            "classdef RetainedDerived < RetainedBase\n properties\n derivedValue\n end\n methods\n function obj = RetainedDerived(v)\n obj@RetainedBase(v);\n obj.derivedValue = v + 1;\n end\n end\nend",
        )
        .expect("write derived class");
        let _cwd = crate::tests::push_cwd(directory.path());
        let mut session = RunMatSession::with_options(false, false).expect("VM session");
        let base_admission = crate::execute_text_request_for_testing(
            &mut session,
            "admittedBase = RetainedBase(1);",
        )
        .expect("separate Base.m admission");
        assert!(
            base_admission.error.is_none(),
            "Base.m admission failed: {base_admission:?}"
        );
        for (name, input) in [("first", 4.0), ("second", 8.0)] {
            let expression = format!(
                "{name} = RetainedDerived({input}); base = {name}.baseValue; derived = {name}.derivedValue;"
            );
            let outcome = crate::execute_text_request_for_testing(&mut session, &expression)
                .expect("derived constructor execution");
            assert!(outcome.error.is_none(), "constructor failed: {outcome:?}");
            let base = block_on(session.materialize_variable(
                crate::workspace::WorkspaceMaterializeTarget::Name("base".into()),
                crate::workspace::WorkspaceMaterializeOptions::default(),
            ))
            .expect("base property");
            let derived = block_on(session.materialize_variable(
                crate::workspace::WorkspaceMaterializeTarget::Name("derived".into()),
                crate::workspace::WorkspaceMaterializeOptions::default(),
            ))
            .expect("derived property");
            assert_eq!(base.value, Value::Num(input));
            assert_eq!(derived.value, Value::Num(input + 1.0));

            let instance = block_on(session.materialize_variable(
                crate::workspace::WorkspaceMaterializeTarget::Name(name.into()),
                crate::workspace::WorkspaceMaterializeOptions::default(),
            ))
            .expect("derived instance");
            let Value::HandleObject(returned) = instance.value else {
                panic!("derived constructor must return its managed instance")
            };
            let Some(Value::HandleObject(observed)) = session
                .runtime_context()
                .session_global_value("retained_base_witness")
                .expect("session global authority")
            else {
                panic!("base constructor body did not publish its instance")
            };
            assert_eq!(returned.target, observed.target);
        }
    });
}
