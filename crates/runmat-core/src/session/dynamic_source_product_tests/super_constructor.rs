use super::super::*;

#[test]
fn separately_admitted_superclass_product_is_joined_by_exact_origin() {
    runmat_gc::gc_test_context(|| {
        let _serial = runmat_filesystem::provider_override_lock();
        let temporary = tempfile::TempDir::new().expect("separate class source root");
        let base_dir = temporary.path().join("base");
        let derived_dir = temporary.path().join("derived");
        std::fs::create_dir(&base_dir).expect("Base.m source directory");
        std::fs::create_dir(&derived_dir).expect("Derived.m source directory");
        std::fs::write(
            base_dir.join("CacheBase.m"),
            "classdef CacheBase < handle\n properties\n value\n end\n methods\n function obj = CacheBase(v)\n global cache_base_witness;\n cache_base_witness = obj;\n obj.value = v;\n end\n end\nend",
        )
        .expect("base source");
        std::fs::write(
            derived_dir.join("CacheDerived.m"),
            "classdef CacheDerived < CacheBase\n methods\n function obj = CacheDerived(v)\n obj@CacheBase(v);\n end\n end\nend",
        )
        .expect("derived source");
        std::fs::write(
            derived_dir.join("UnusedDerivedCompanion.m"),
            "classdef UnusedDerivedCompanion < AbsentCompanionParent\nend\n",
        )
        .expect("unrelated companion class");
        let _cwd = crate::tests::push_cwd(temporary.path());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let base_path = base_dir.to_string_lossy().replace('\'', "''");
        let derived_path = derived_dir.to_string_lossy().replace('\'', "''");
        let base = crate::execute_text_request_for_testing(
            &mut session,
            &format!("addpath('{base_path}'); addpath('{derived_path}'); base = CacheBase(1);"),
        )
        .expect("Base.m admission");
        assert!(base.error.is_none(), "{base:?}");
        let derived = crate::execute_text_request_for_testing(
            &mut session,
            "derived = CacheDerived(7); observed = derived.value;",
        )
        .expect("Derived.m admission");
        assert!(derived.error.is_none(), "{derived:?}");
        assert_eq!(
            session.get_variables().unwrap().get("observed"),
            Some(&Value::Num(7.0))
        );
        let cache = session.dynamic_function_cache.snapshot();
        let base = cache
            .values()
            .find(|entry| {
                entry
                    .admitted
                    .registry()
                    .get(
                        entry
                            .admitted
                            .entrypoint_function()
                            .expect("base entrypoint"),
                    )
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "CacheBase"
                        })
                    })
            })
            .expect("selected Base.m product");
        let derived = cache
            .values()
            .find(|entry| {
                entry
                    .admitted
                    .registry()
                    .get(
                        entry
                            .admitted
                            .entrypoint_function()
                            .expect("derived entrypoint"),
                    )
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "CacheDerived"
                        })
                    })
            })
            .expect("selected Derived.m product");
        assert_ne!(
            base.entrypoint_origin.program,
            derived.entrypoint_origin.program
        );
        let companion = runmat_types::ClassIdentity::new("UnusedDerivedCompanion").unwrap();
        assert!(derived.admitted.registry().functions.values().any(|body| body
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, runmat_vm::Instr::RegisterClass { name, .. } if name == &companion))));
        assert!(derived.super_dependency_origins.contains(
            &super::super::DynamicDependencyIdentity::Callable(base.entrypoint_origin)
        ));
        assert_eq!(derived.super_constructors.len(), 1);
        let old_base_origin = base.entrypoint_origin;
        drop(cache);

        std::fs::write(
            base_dir.join("CacheBase.m"),
            "classdef CacheBase < handle\n properties\n value\n end\n methods\n function obj = CacheBase(v)\n global cache_base_witness;\n cache_base_witness = obj;\n obj.value = v + 20;\n end\n end\nend",
        )
        .expect("changed Base.m source");
        let changed = crate::execute_text_request_for_testing(
            &mut session,
            "changedBase = CacheBase(2); changedDerived = CacheDerived(3); changedValue = changedDerived.value;",
        )
        .expect("new Base.m and retained Derived.m selection");
        assert!(changed.error.is_none(), "{changed:?}");
        assert_eq!(
            session.get_variables().unwrap().get("changedValue"),
            Some(&Value::Num(23.0))
        );
        let cache = session.dynamic_function_cache.snapshot();
        let changed_base = cache
            .values()
            .find(|entry| {
                entry
                    .admitted
                    .registry()
                    .get(entry.admitted.entrypoint_function().expect("entrypoint"))
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "CacheBase"
                        })
                    })
            })
            .expect("new Base.m admitted product");
        assert_ne!(changed_base.entrypoint_origin, old_base_origin);
        let changed_derived = cache
            .values()
            .find(|entry| {
                entry
                    .admitted
                    .registry()
                    .get(entry.admitted.entrypoint_function().expect("entrypoint"))
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "CacheDerived"
                        })
                    })
            })
            .expect("retained Derived.m admitted product");
        assert!(changed_derived.super_dependency_origins.contains(
            &super::super::DynamicDependencyIdentity::Callable(changed_base.entrypoint_origin)
        ));
    });
}

#[test]
fn cold_superclass_source_is_admitted_without_running_its_constructor() {
    runmat_gc::gc_test_context(|| {
        let _serial = runmat_filesystem::provider_override_lock();
        let temporary = tempfile::TempDir::new().expect("separate class source root");
        let base_dir = temporary.path().join("cold_base");
        let derived_dir = temporary.path().join("cold_derived");
        std::fs::create_dir(&base_dir).expect("Base source directory");
        std::fs::create_dir(&derived_dir).expect("Derived source directory");
        std::fs::write(
            base_dir.join("ColdBase.m"),
            "classdef ColdBase < handle\n properties\n value\n seed = 11\n end\n methods\n function obj = ColdBase(v)\n global cold_base_calls cold_base_receiver;\n cold_base_calls = cold_base_calls + 1;\n cold_base_receiver = obj;\n obj.value = v;\n end\n end\nend",
        )
        .expect("Base source");
        std::fs::write(
            derived_dir.join("ColdDerived.m"),
            "classdef ColdDerived < ColdBase\n methods\n function obj = ColdDerived(v)\n obj@ColdBase(v);\n end\n end\nend",
        )
        .expect("Derived source");
        let _cwd = crate::tests::push_cwd(temporary.path());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let base_path = base_dir.to_string_lossy().replace('\'', "''");
        let derived_path = derived_dir.to_string_lossy().replace('\'', "''");
        let result = crate::execute_text_request_for_testing(
            &mut session,
            &format!(
                "addpath('{base_path}'); addpath('{derived_path}'); global cold_base_calls cold_base_receiver; cold_base_calls = 0; d = ColdDerived(7); observed = d.value; observedSeed = d.seed; calls = cold_base_calls; witness = cold_base_receiver;"
            ),
        )
        .expect("cold superclass source resolution");
        assert!(result.error.is_none(), "{result:?}");
        let variables = session.get_variables().expect("workspace");
        assert_eq!(variables.get("observed"), Some(&Value::Num(7.0)));
        assert_eq!(variables.get("observedSeed"), Some(&Value::Num(11.0)));
        assert_eq!(variables.get("calls"), Some(&Value::Num(1.0)));
        match (variables.get("d"), variables.get("witness")) {
            (Some(Value::HandleObject(derived)), Some(Value::HandleObject(base_seen))) => {
                assert_eq!(derived.target, base_seen.target);
                assert_eq!(derived.class_name, base_seen.class_name);
            }
            other => panic!("superclass body must receive the Derived handle: {other:?}"),
        }
        let cache = session.dynamic_function_cache.snapshot();
        let derived = cache
            .values()
            .find(|entry| {
                entry
                    .admitted
                    .entrypoint_function()
                    .and_then(|id| entry.admitted.registry().get(id))
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "ColdDerived"
                        })
                    })
            })
            .expect("Derived admitted product");
        assert_eq!(derived.super_dependency_origins.len(), 1);
        assert_eq!(derived.super_constructors.len(), 1);
    });
}

#[test]
fn cold_implicit_superclass_declaration_supplies_inherited_default() {
    runmat_gc::gc_test_context(|| {
        let _serial = runmat_filesystem::provider_override_lock();
        let temporary = tempfile::TempDir::new().expect("separate class source root");
        let base_dir = temporary.path().join("implicit_base");
        let derived_dir = temporary.path().join("implicit_derived");
        std::fs::create_dir(&base_dir).expect("Base source directory");
        std::fs::create_dir(&derived_dir).expect("Derived source directory");
        std::fs::write(
            base_dir.join("ColdImplicitBase.m"),
            "classdef ColdImplicitBase < handle\n properties\n seed = 13\n end\nend",
        )
        .expect("Base source");
        std::fs::write(
            derived_dir.join("ColdImplicitDerived.m"),
            "classdef ColdImplicitDerived < ColdImplicitBase\n methods\n function obj = ColdImplicitDerived()\n obj.seed = obj.seed + 1;\n end\n end\nend",
        )
        .expect("Derived source");
        let _cwd = crate::tests::push_cwd(temporary.path());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let base_path = base_dir.to_string_lossy().replace('\'', "''");
        let derived_path = derived_dir.to_string_lossy().replace('\'', "''");
        let result = crate::execute_text_request_for_testing(
            &mut session,
            &format!(
                "addpath('{base_path}'); addpath('{derived_path}'); d = ColdImplicitDerived(); observed = d.seed;"
            ),
        )
        .expect("implicit superclass source resolution");
        assert!(result.error.is_none(), "{result:?}");
        assert_eq!(
            session.get_variables().expect("workspace").get("observed"),
            Some(&Value::Num(14.0))
        );
        let cache = session.dynamic_function_cache.snapshot();
        let derived = cache
            .values()
            .find(|entry| {
                entry
                    .admitted
                    .entrypoint_function()
                    .and_then(|id| entry.admitted.registry().get(id))
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "ColdImplicitDerived"
                        })
                    })
            })
            .expect("Derived admitted product");
        assert_eq!(derived.super_dependency_origins.len(), 1);
        assert!(derived.super_constructors.is_empty());
    });
}

#[test]
fn retained_derived_constructor_keeps_old_constructorless_base_defaults() {
    runmat_gc::gc_test_context(|| {
        let _serial = runmat_filesystem::provider_override_lock();
        let temporary = tempfile::TempDir::new().expect("separate class source root");
        let base_dir = temporary.path().join("base_revision");
        let derived_dir = temporary.path().join("derived_revision");
        std::fs::create_dir(&base_dir).expect("base directory");
        std::fs::create_dir(&derived_dir).expect("derived directory");
        let base_source = base_dir.join("RevisionBase.m");
        std::fs::write(
            &base_source,
            "classdef RevisionBase < handle\n properties\n seed = 11\n end\nend",
        )
        .expect("old constructorless Base source");
        std::fs::write(
            derived_dir.join("RevisionDerived.m"),
            "classdef RevisionDerived < RevisionBase\n methods\n function obj = RevisionDerived()\n obj@RevisionBase();\n obj.seed = obj.seed + 1;\n end\n end\nend",
        )
        .expect("Derived source");
        let _cwd = crate::tests::push_cwd(temporary.path());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let base_path = base_dir.to_string_lossy().replace('\'', "''");
        let derived_path = derived_dir.to_string_lossy().replace('\'', "''");
        let first = crate::execute_text_request_for_testing(
            &mut session,
            &format!(
                "addpath('{base_path}'); addpath('{derived_path}'); builtin_handle = @sin; old_ctor = @RevisionDerived; first = old_ctor(); direct = RevisionDerived(); first_seed = first.seed; direct_seed = direct.seed;"
            ),
        )
        .expect("old product admission");
        assert!(first.error.is_none(), "{first:?}");
        assert_eq!(
            session.get_variables().unwrap().get("first_seed"),
            Some(&Value::Num(12.0))
        );
        assert_eq!(
            session.get_variables().unwrap().get("direct_seed"),
            Some(&Value::Num(12.0))
        );
        assert!(matches!(
            session.get_variables().unwrap().get("old_ctor"),
            Some(Value::BoundFunctionHandle(_))
        ));
        assert!(matches!(
            session.get_variables().unwrap().get("builtin_handle"),
            Some(Value::FunctionHandle(name)) if name.name() == "sin"
        ));
        std::fs::write(
            &base_source,
            "classdef RevisionBase < handle\n properties\n seed = 31\n end\nend",
        )
        .expect("new constructorless Base source");
        let second = crate::execute_text_request_for_testing(
            &mut session,
            "new_value = RevisionDerived(); new_ctor = @RevisionDerived; old_again = old_ctor(); new_again = new_ctor(); new_seed = new_value.seed; old_seed = old_again.seed; newest_seed = new_again.seed;",
        )
        .expect("old and new constructor selection");
        assert!(second.error.is_none(), "{second:?}");
        let values = session.get_variables().expect("workspace values");
        let (old_name, old_file, old_authority) = match values.get("old_ctor") {
            Some(Value::BoundFunctionHandle(handle)) => {
                (&handle.name, &handle.source_file, &handle.authority)
            }
            other => panic!("old constructor lost its bound authority: {other:?}"),
        };
        let (new_name, new_file, new_authority) = match values.get("new_ctor") {
            Some(Value::BoundFunctionHandle(handle)) => {
                (&handle.name, &handle.source_file, &handle.authority)
            }
            other => panic!("new constructor lacks bound authority: {other:?}"),
        };
        assert_eq!(old_name, new_name, "Derived source name is unchanged");
        assert_eq!(old_file, new_file, "Derived source path is unchanged");
        assert_ne!(
            old_authority.local_definition(),
            new_authority.local_definition(),
            "same-named Derived handles retain different exact definitions"
        );
        assert_eq!(values.get("new_seed"), Some(&Value::Num(32.0)));
        assert_eq!(values.get("newest_seed"), Some(&Value::Num(32.0)));
        assert_eq!(values.get("old_seed"), Some(&Value::Num(12.0)));
        assert!(matches!(
            values.get("old_again"),
            Some(Value::HandleObject(_))
        ));
        assert!(matches!(
            values.get("new_value"),
            Some(Value::HandleObject(_))
        ));
    });
}

#[test]
fn cold_constructorless_superclass_with_other_methods_has_exact_no_body_target() {
    runmat_gc::gc_test_context(|| {
        let _serial = runmat_filesystem::provider_override_lock();
        let temporary = tempfile::TempDir::new().expect("separate class source root");
        let base_dir = temporary.path().join("explicit_base");
        let derived_dir = temporary.path().join("explicit_derived");
        std::fs::create_dir(&base_dir).expect("Base source directory");
        std::fs::create_dir(&derived_dir).expect("Derived source directory");
        std::fs::write(
            base_dir.join("ColdNoBodyBase.m"),
            "classdef ColdNoBodyBase < handle\n properties\n seed = 13\n end\n methods\n function v = marker(obj)\n v = obj.seed;\n end\n function v = secondMarker(obj)\n v = obj.seed + 1;\n end\n end\nend",
        )
        .expect("constructorless Base source with an ordinary method");
        std::fs::write(
            derived_dir.join("ColdExplicitDerived.m"),
            "classdef ColdExplicitDerived < ColdNoBodyBase\n methods\n function obj = ColdExplicitDerived()\n obj@ColdNoBodyBase();\n obj.seed = obj.seed + 2;\n end\n end\nend",
        )
        .expect("Derived source");
        let _cwd = crate::tests::push_cwd(temporary.path());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let base_path = base_dir.to_string_lossy().replace('\'', "''");
        let derived_path = derived_dir.to_string_lossy().replace('\'', "''");
        let result = crate::execute_text_request_for_testing(
            &mut session,
            &format!(
                "addpath('{base_path}'); addpath('{derived_path}'); d = ColdExplicitDerived(); observed = d.seed; marker_value = d.marker(); second_value = d.secondMarker();"
            ),
        )
        .expect("explicit no-body superclass source resolution");
        assert!(result.error.is_none(), "{result:?}");
        assert_eq!(
            session.get_variables().expect("workspace").get("observed"),
            Some(&Value::Num(15.0))
        );
        assert_eq!(
            session
                .get_variables()
                .expect("workspace")
                .get("marker_value"),
            Some(&Value::Num(15.0))
        );
        assert_eq!(
            session
                .get_variables()
                .expect("workspace")
                .get("second_value"),
            Some(&Value::Num(16.0))
        );
        let derived = session
            .dynamic_function_cache
            .snapshot()
            .into_values()
            .find(|entry| {
                entry
                    .admitted
                    .entrypoint_function()
                    .and_then(|id| entry.admitted.registry().get(id))
                    .is_some_and(|body| {
                        body.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class.display_name() == "ColdExplicitDerived"
                        })
                    })
            })
            .expect("Derived admitted product");
        assert_eq!(derived.super_constructors.len(), 1);
        assert_eq!(derived.super_dependency_origins.len(), 1);
        // The wire artifact has no live class-product authority. Inspect the
        // exact execution view issued for this cached constructor binding.
        let selected_view = session
            .source_evaluator
            .selected_binding_view(&derived.binding)
            .expect("selected Derived binding view")
            .expect("live Derived binding view");
        let selected_lease = derived
            .binding
            .authority
            .local_lease(derived.binding.function)
            .expect("selected Derived definition lease");
        let derived_body = selected_view
            .get_owned(
                runmat_hir::FunctionId(derived.binding.function),
                selected_lease,
            )
            .expect("selected Derived constructor body");
        let selected_class = derived_body
            .selected_class_product
            .as_ref()
            .expect("selected Derived class ancestry");
        let base = selected_class
            .declaration_products()
            .iter()
            .find(|product| product.class().display_name() == "ColdNoBodyBase")
            .expect("exact constructorless Base declaration product");
        assert!(base.declarations().iter().any(|instruction| {
            matches!(instruction,
                runmat_vm::Instr::RegisterClass { name, methods, .. }
                    if name.display_name() == "ColdNoBodyBase" && methods.len() == 2)
        }));
        let admitted_base = runmat_vm::admit_executable_unit(base.canonical_bytes())
            .expect("admitted constructorless Base product");
        assert!(admitted_base.declaration_class().is_some());
        assert!(admitted_base.registry().functions.values().all(|body| body
            .class_method_owner
            .as_ref()
            .is_none_or(|owner| {
                owner.declaring_class.display_name() != "ColdNoBodyBase"
                    || owner.role != runmat_types::ClassMethodRole::Constructor
            })));
    });
}
