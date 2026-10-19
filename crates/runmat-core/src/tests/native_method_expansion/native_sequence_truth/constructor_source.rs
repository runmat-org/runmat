use super::*;
use runmat_types::{ClassIdentity, MemberAccess, MethodName};

fn register_source_constructor(session: &RunMatSession, class_name: &str) {
    let class = ClassIdentity::new(class_name).expect("class identity");
    let leaf = class
        .qualified_name()
        .0
        .last()
        .expect("class leaf")
        .0
        .clone();
    let method = MethodName::from(leaf.as_str());
    let mut methods = std::collections::HashMap::new();
    methods.insert(
        method.clone(),
        runmat_runtime::class_registry::RuntimeMethod {
            name: method,
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: format!("{class_name}.{leaf}"),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Constructor,
        },
    );
    let _selected = session.runtime_context().enter();
    runmat_runtime::class_registry::register_class(runmat_runtime::class_registry::RuntimeClass {
        name: class,
        parent: None,
        properties: std::collections::HashMap::new(),
        methods,
        events: std::collections::HashMap::new(),
    });
}

#[test]
fn registered_plain_and_package_sources_execute_their_exact_constructor_bodies() {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = crate::tests::cwd_lock();
        let directory = tempfile::TempDir::new().expect("constructor source directory");
        let package = directory.path().join("+pkg");
        std::fs::create_dir(&package).expect("package source directory");
        std::fs::write(
            directory.path().join("SourceCtor.m"),
            "classdef SourceCtor < handle\n methods\n function obj = SourceCtor()\n global source_ctor_witness;\n source_ctor_witness = obj;\n end\n end\nend",
        )
        .expect("plain class source");
        std::fs::write(
            package.join("PackageCtor.m"),
            "classdef PackageCtor < handle\n methods\n function obj = PackageCtor()\n global package_ctor_witness;\n package_ctor_witness = obj;\n end\n end\nend",
        )
        .expect("package class source");
        let _cwd = crate::tests::push_cwd(directory.path());

        for (class_name, global_name, expression) in [
            (
                "SourceCtor",
                "source_ctor_witness",
                "object = SourceCtor();",
            ),
            (
                "pkg.PackageCtor",
                "package_ctor_witness",
                "object = pkg.PackageCtor();",
            ),
        ] {
            let mut session = RunMatSession::with_options(false, false).expect("VM session");
            register_source_constructor(&session, class_name);
            let outcome = crate::execute_text_request_for_testing(&mut session, expression)
                .expect("constructor source execution");
            assert!(!outcome.used_jit, "constructor must run through VM");
            assert!(
                outcome.error.is_none(),
                "constructor body failed: {outcome:?}"
            );
            let object = block_on(session.materialize_variable(
                crate::workspace::WorkspaceMaterializeTarget::Name("object".into()),
                crate::workspace::WorkspaceMaterializeOptions::default(),
            ))
            .expect("returned constructor object");
            let Value::HandleObject(returned) = object.value else {
                panic!("constructor must return its object")
            };
            assert_eq!(
                returned.class_name,
                ClassIdentity::new(class_name).expect("class identity")
            );
            let Some(Value::HandleObject(observed)) = session
                .runtime_context()
                .session_global_value(global_name)
                .expect("session global authority")
            else {
                panic!(
                    "constructor body did not publish its instance; session globals={:?}, witness={:?}",
                    session.runtime_context().session_global_names(),
                    session.runtime_context().session_global_value(global_name)
                )
            };
            assert_eq!(returned.target, observed.target);
        }
    });
}

#[test]
fn registered_source_mismatch_cannot_execute_an_ordinary_same_named_function() {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = crate::tests::cwd_lock();
        let directory = tempfile::TempDir::new().expect("mismatched constructor source directory");
        std::fs::write(
            directory.path().join("MismatchCtor.m"),
            "function obj = MismatchCtor()\n global constructor_mismatch_effect;\n constructor_mismatch_effect = 1;\n obj = 1;\nend",
        )
        .expect("ordinary same-named source");
        let _cwd = crate::tests::push_cwd(directory.path());
        let mut session = RunMatSession::with_options(false, false).expect("VM session");
        register_source_constructor(&session, "MismatchCtor");
        let outcome =
            crate::execute_text_request_for_testing(&mut session, "object = MismatchCtor();")
                .expect("source mismatch execution");
        assert_eq!(
            outcome.error.as_ref().and_then(|error| error.identifier()),
            Some("RunMat:ConstructorSourceMismatch")
        );
        assert!(
            session
                .runtime_context()
                .session_global_value("constructor_mismatch_effect")
                .expect("session global authority")
                .is_none(),
            "ordinary function must not execute before constructor proof"
        );
    });
}
