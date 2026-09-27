use super::{instr_fusion_signal_kind, mir_call_has_fusion_signal};
use crate::bytecode::Instr;
use runmat_builtins::{BuiltinEffects, BuiltinPurity, BuiltinSemanticKind};
use runmat_hir::{CallSyntax, CallableFallbackPolicy, CallableIdentity, RequestedOutputCount};
use runmat_mir::{AsyncBehaviorFact, MirCall, MirCallArg, MirCallee, MirConstant, MirOperand};

fn single_call() -> MirCall {
    MirCall {
        callee: MirCallee::Static(CallableIdentity::Builtin(runmat_types::BuiltinId(
            "single".into(),
        ))),
        args: vec![MirCallArg::Single(MirOperand::Constant(
            MirConstant::Number("1".into()),
        ))],
        arg_spans: vec![runmat_hir::Span::default()],
        syntax: CallSyntax::Plain,
        requested_outputs: RequestedOutputCount::One,
        fallback_policy: CallableFallbackPolicy::None,
        workspace_first_name: None,
        bare_identifier: false,
        async_behavior: AsyncBehaviorFact::MaySuspend,
        effects: BuiltinEffects::none(),
        workspace_effect: None,
        environment_effect: None,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
    }
}

#[test]
fn canonical_numeric_suspension_is_not_an_inference_category() {
    let call = single_call();
    assert!(mir_call_has_fusion_signal(&call));
    assert!(instr_fusion_signal_kind(&Instr::CallBuiltinMulti("single".into(), 1, 1)).is_some());
    for outputs in [0, 2] {
        assert!(
            instr_fusion_signal_kind(&Instr::CallBuiltinMulti("single".into(), 1, outputs))
                .is_none()
        );
    }
    assert!(instr_fusion_signal_kind(&Instr::CallBuiltinMulti("single".into(), 3, 1)).is_none());
}

#[test]
fn effects_async_services_dynamic_calls_and_sequences_do_not_become_candidates() {
    let call = single_call();
    let mut changed = call.clone();
    changed.effects = BuiltinEffects::unknown();
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.async_behavior = AsyncBehaviorFact::RequiresAsyncRuntime;
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.purity = BuiltinPurity::Impure;
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.callee = MirCallee::Static(CallableIdentity::DynamicName(runmat_types::SymbolName(
        "single".into(),
    )));
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.args[0] = MirCallArg::CapturedSequence(runmat_mir::MirSequenceLocalId(0));
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call;
    changed.requested_outputs = RequestedOutputCount::CurrentFunctionNargout;
    assert!(!mir_call_has_fusion_signal(&changed));
}

#[test]
fn unsupported_canonical_binding_and_effectful_declarations_fail_closed() {
    use crate::accel::fusion_declaration::{call_kind, canonical_kind};
    use runmat_builtins::{
        builtin_catalog_entry_by_name, BuiltinBindingAvailability, BuiltinBindingDeclaration,
    };
    let mut entry = *builtin_catalog_entry_by_name("single").unwrap();
    const WRONG: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
        variant: "not-the-registered-binding",
        availability: BuiltinBindingAvailability::Required,
    }];
    entry.bindings = &WRONG;
    assert!(canonical_kind(&entry, Some(1)).is_none());
    entry = *builtin_catalog_entry_by_name("single").unwrap();
    entry.contract.effects = &[runmat_types::EffectKind::HostCallback];
    assert!(canonical_kind(&entry, Some(1)).is_none());
    assert!(call_kind("syms", 1, 1).is_none());
    assert!(call_kind("__not_a_registered_builtin__", 1, 1).is_none());
}
