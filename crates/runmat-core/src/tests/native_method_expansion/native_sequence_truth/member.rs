use super::*;
use crate::InvocationControl;
use futures::channel::oneshot;

#[test]
fn vm_constructor_reads_the_same_implicit_instance_that_it_returns() {
    with_native_unit(|session, unit, _calls| {
        assert!(
            unit.analysis()
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-DEAL-COUNT"),
            "the invalid unused companion remains diagnosed"
        );
        let outcome = crate::execute_text_request_for_testing(session, "object = PendingMember();")
            .expect("VM constructor execution");
        assert!(!outcome.used_jit, "the constructor must run through the VM");
        assert!(
            outcome.error.is_none(),
            "constructor instance must be live: {outcome:?}"
        );
        let object = block_on(session.materialize_variable(
            crate::workspace::WorkspaceMaterializeTarget::Name("object".into()),
            crate::workspace::WorkspaceMaterializeOptions::default(),
        ))
        .expect("returned constructor object");
        let Value::HandleObject(returned) = object.value else {
            panic!("constructor must return its managed instance")
        };
        let Some(Value::HandleObject(observed)) = session
            .runtime_context()
            .session_global_value("pending_member_witness")
            .expect("session global authority")
        else {
            panic!(
                "constructor body must observe its implicit instance; session globals={:?}, witness={:?}",
                session.runtime_context().session_global_names(),
                session.runtime_context().session_global_value("pending_member_witness")
            )
        };
        assert_eq!(
            returned.target, observed.target,
            "constructor changed instance identity"
        );
    });
}

#[test]
fn vm_super_constructor_retains_the_derived_instance() {
    runmat_gc::gc_test_context(|| {
        let _cwd_lock = crate::tests::cwd_lock();
        let directory = tempfile::TempDir::new().expect("super constructor source directory");
        std::fs::write(
            directory.path().join("SeedBase.m"),
            "classdef SeedBase < handle\n methods\n function obj = SeedBase()\n global constructor_super_witness;\n constructor_super_witness = obj;\n end\n end\nend",
        )
        .expect("write base class");
        std::fs::write(
            directory.path().join("SeedDerived.m"),
            "classdef SeedDerived < SeedBase\n methods\n function obj = SeedDerived()\n obj@SeedBase();\n end\n end\nend",
        )
        .expect("write derived class");
        let _cwd = crate::tests::push_cwd(directory.path());
        let mut session = RunMatSession::with_options(false, false).expect("VM session");
        let outcome =
            crate::execute_text_request_for_testing(&mut session, "instance = SeedDerived();")
                .expect("VM superclass constructor execution");
        assert!(!outcome.used_jit, "the constructor must run through the VM");
        assert!(
            outcome.error.is_none(),
            "super constructor failed: {outcome:?}"
        );
        let instance = block_on(session.materialize_variable(
            crate::workspace::WorkspaceMaterializeTarget::Name("instance".into()),
            crate::workspace::WorkspaceMaterializeOptions::default(),
        ))
        .expect("derived constructor instance");
        let Value::HandleObject(returned) = instance.value else {
            panic!("derived constructor must return its managed instance")
        };
        let Some(Value::HandleObject(observed)) = session
            .runtime_context()
            .session_global_value("constructor_super_witness")
            .expect("session global authority")
        else {
            panic!(
                "base constructor must observe the derived instance; session globals={:?}, witness={:?}",
                session.runtime_context().session_global_names(),
                session.runtime_context().session_global_value("constructor_super_witness")
            )
        };
        assert_eq!(
            returned.target, observed.target,
            "base constructor allocated a new instance"
        );
    });
}

#[test]
fn generated_member_reads_resume_without_replay_and_retire_managed_receivers() {
    with_native_unit(|session, unit, calls| {
        let control = InvocationControl::default().force_generic_native();
        for function in ["readMember", "readDynamicMember"] {
            let (release, gate) = oneshot::channel();
            install_input(
                session,
                calls.clone(),
                vec![(
                    Some(gate),
                    Ok(crate::execution::InputResponse::Line("1".into())),
                )],
            );
            calls.store(0, Ordering::SeqCst);
            let mut pending =
                Box::pin(session.invoke_executable(unit, invoke(function, 1), &control));
            pending_once(pending.as_mut(), &calls);
            let target = input::observed_member_target();
            assert_eq!(
                input::observed_member_reads(),
                Some(1.0),
                "overload must run only once before suspension"
            );
            runmat_gc::gc_collect_major().expect("collect during pending member read");
            assert!(runmat_gc::gc_clone_value(&target).is_ok());
            release.send(()).expect("release member read");
            let values = block_on(pending).expect("resume generated member read");
            assert_eq!(values.as_slice(), &[Value::Num(7.0)]);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "member read did not replay"
            );
            drop(values);
            runmat_gc::gc_collect_major().expect("collect retired member receiver");
            assert!(runmat_gc::gc_clone_value(&target).is_err());
        }

        let (release, gate) = oneshot::channel();
        install_input(
            session,
            calls.clone(),
            vec![(
                Some(gate),
                Ok(crate::execution::InputResponse::Line("1".into())),
            )],
        );
        calls.store(0, Ordering::SeqCst);
        let mut abandoned =
            Box::pin(session.invoke_executable(unit, invoke("readMember", 1), &control));
        pending_once(abandoned.as_mut(), &calls);
        let target = input::observed_member_target();
        runmat_gc::gc_collect_major().expect("collect before abandoned native read");
        assert!(runmat_gc::gc_clone_value(&target).is_ok());
        drop(abandoned);
        assert!(
            release.send(()).is_err(),
            "dropped read retained its input future"
        );
        runmat_gc::gc_collect_major().expect("collect after abandoned native read");
        assert!(runmat_gc::gc_clone_value(&target).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1, "dropped read replayed");
    });
}
