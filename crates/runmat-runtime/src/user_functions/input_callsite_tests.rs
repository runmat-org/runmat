use super::*;
use crate::callsite::{current_function_input_callsite, CallsiteInfo};
use runmat_types::{Span, SymbolName};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn runtime_named_resolution_installs_caller_only_for_selected_user_function() {
    let _resolver = install_semantic_function_resolver(Some(Arc::new(|name| {
        (name == "late_user").then_some(17)
    })));
    let _invoker = install_semantic_function_invoker(Some(Arc::new(|function, _, _| {
        assert_eq!(function, 17);
        Box::pin(async {
            let callsite = current_function_input_callsite().expect("selected user caller facts");
            assert_eq!(callsite.source_id, Some(SourceId(13)));
            assert_eq!(callsite.arg_spans, vec![Span { start: 4, end: 9 }]);
            crate::sequence::single_value_sequence(Value::Num(17.0))
        })
    })));
    let request = CallableRequest::resolved(
        CallableIdentity::DynamicName(SymbolName("late_user".into())),
        CallableFallbackPolicy::RuntimeNameResolution,
        vec![Value::Num(1.0)],
        1,
    )
    .with_function_input_callsite(Some(CallsiteInfo {
        source_id: Some(SourceId(13)),
        arg_spans: vec![Span { start: 4, end: 9 }],
    }));
    let result = futures::executor::block_on(try_call_semantic_descriptor(request))
        .expect("user function selected")
        .expect("user function completed");
    assert_eq!(result.into_values(), vec![Value::Num(17.0)]);
    assert!(current_function_input_callsite().is_none());
}
