use runmat_value::Value;

use crate::{execute_text_request_for_testing, RunMatSession};
#[cfg(not(target_arch = "wasm32"))]
use crate::{ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget};
#[cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;

#[test]
fn current_constructor_handle_keeps_its_class_defaults_after_republication() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let original = execute_text_request_for_testing(
            &mut session,
            "classdef LocalRevision < handle\n properties\n seed = 11\n bodyTag\n end\n methods\n function obj = LocalRevision()\n obj.bodyTag = 101;\n end\n end\nend",
        )
        .expect("first class publication");
        assert!(original.error.is_none(), "{original:?}");
        let retained = execute_text_request_for_testing(&mut session, "old_ctor = @LocalRevision;")
            .expect("retain the first constructor");
        assert!(retained.error.is_none(), "{retained:?}");

        let replacement = execute_text_request_for_testing(
            &mut session,
            "classdef LocalRevision < handle\n properties\n seed = 31\n bodyTag\n end\n methods\n function obj = LocalRevision()\n obj.bodyTag = 303;\n end\n end\nend",
        )
        .expect("replacement class publication");
        assert!(replacement.error.is_none(), "{replacement:?}");
        #[cfg(not(target_arch = "wasm32"))]
        {
            for source in [
                "function y = retainedOrdinary(x)\n y = x + 1;\nend",
                "old_fn = @retainedOrdinary;",
                "function y = retainedOrdinary(x)\n y = x + 100;\nend",
            ] {
                let published = execute_text_request_for_testing(&mut session, source)
                    .expect("ordinary function publication");
                assert!(published.error.is_none(), "{source}: {published:?}");
            }
            let unrelated = "function [out, obj, oldResult] = unrelatedNative(v, ctor, callback)\n out = v + 1;\n obj = ctor();\n oldResult = callback(v);\nend";
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new("unrelated-native@1", "unrelatedNative.m", unrelated),
                None,
            ))
            .expect("compile unrelated native source after class replacement");
            let workspace = session.get_variables().expect("workspace");
            let old_ctor = workspace
                .get("old_ctor")
                .cloned()
                .expect("retained old constructor handle");
            let Value::BoundFunctionHandle(old_ctor_handle) = &old_ctor else {
                panic!("admitted constructor handle must retain an exact bound definition: {old_ctor:?}")
            };
            let old_ctor_lease = old_ctor_handle
                .authority
                .local_lease(old_ctor_handle.function)
                .expect("retained constructor must carry its local definition lease");
            let native_registry = unit.bytecode().function_registry();
            let native_selected_body = native_registry.get_owned(
                runmat_hir::FunctionId(old_ctor_handle.function),
                old_ctor_lease,
            );
            let native_has_old_product = native_selected_body
                .and_then(|body| body.selected_class_product.as_ref())
                .is_some();
            let old_fn = workspace
                .get("old_fn")
                .cloned()
                .expect("retained old ordinary handle");
            let outputs = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Function("unrelatedNative".into()),
                    arguments: vec![Value::Num(4.0), old_ctor, old_fn],
                    requested_outputs: 3,
                },
                &InvocationControl::default().force_generic_native(),
            ))
            .expect("execute retained constructor inside unrelated native source");
            let [Value::Num(result), Value::HandleObject(old_native), Value::Num(old_result)] =
                outputs.as_slice()
            else {
                panic!("native source must return numeric result, old object, and old callback")
            };
            assert_eq!(*result, 5.0);
            assert_eq!(*old_result, 5.0);
            let Some(Value::Object(properties)) =
                runmat_gc::gc_clone_value(&old_native.target).ok()
            else {
                panic!("native-retained constructor must return a live object")
            };
            assert_eq!(
                properties.properties.get("seed"),
                Some(&Value::Num(11.0)),
                "native registry owns old constructor: {}; selected product: {}; returned bodyTag: {:?}",
                native_selected_body.is_some(),
                native_has_old_product,
                properties.properties.get("bodyTag")
            );
            assert_eq!(
                properties.properties.get("bodyTag"),
                Some(&Value::Num(101.0))
            );
        }
        let observed = execute_text_request_for_testing(
            &mut session,
            "new_ctor = @LocalRevision; old_obj = old_ctor(); new_obj = new_ctor(); old_seed = old_obj.seed; new_seed = new_obj.seed; old_tag = old_obj.bodyTag; new_tag = new_obj.bodyTag;",
        )
        .expect("invoke both selected constructor products");
        assert!(observed.error.is_none(), "{observed:?}");
        let workspace = session.get_variables().expect("workspace");
        assert_eq!(workspace.get("old_seed"), Some(&Value::Num(11.0)));
        assert_eq!(workspace.get("new_seed"), Some(&Value::Num(31.0)));
        assert_eq!(workspace.get("old_tag"), Some(&Value::Num(101.0)));
        assert_eq!(workspace.get("new_tag"), Some(&Value::Num(303.0)));
        assert!(matches!(
            workspace.get("old_obj"),
            Some(Value::HandleObject(_))
        ));
        assert!(matches!(
            workspace.get("new_obj"),
            Some(Value::HandleObject(_))
        ));
    });
}
