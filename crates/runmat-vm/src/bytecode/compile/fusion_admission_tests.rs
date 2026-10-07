use super::{instr_fusion_signal_kind, mir_call_has_fusion_signal};
use crate::bytecode::Instr;
use runmat_builtins::{
    builtin_catalog_entry_by_name, BuiltinAsyncBehavior, BuiltinEffects, BuiltinEnvironmentEffect,
    BuiltinPurity, BuiltinWorkspaceEffect,
};
use runmat_hir::{CallSyntax, CallableFallbackPolicy, CallableIdentity, RequestedOutputCount};
use runmat_mir::{AsyncBehaviorFact, MirCall, MirCallArg, MirCallee, MirConstant, MirOperand};

fn catalog_call(name: &str) -> MirCall {
    let semantics = builtin_catalog_entry_by_name(name)
        .expect("canonical numerical catalog entry")
        .legacy_semantics();
    MirCall {
        callee: MirCallee::Static(CallableIdentity::Builtin(runmat_types::BuiltinId(
            name.into(),
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
        async_behavior: match semantics.async_behavior {
            BuiltinAsyncBehavior::NeverSuspends => AsyncBehaviorFact::NeverSuspends,
            BuiltinAsyncBehavior::MaySuspend => AsyncBehaviorFact::MaySuspend,
            BuiltinAsyncBehavior::RequiresAsyncRuntime => AsyncBehaviorFact::RequiresAsyncRuntime,
        },
        effects: semantics.effects,
        workspace_effect: semantics.workspace_effect,
        environment_effect: semantics.environment_effect,
        purity: semantics.purity,
        semantic_kind: semantics.semantic_kind,
    }
}

#[test]
fn canonical_numeric_suspension_is_not_an_inference_category() {
    let call = catalog_call("single");
    assert_eq!(call.effects, BuiltinEffects::none());
    assert_eq!(call.async_behavior, AsyncBehaviorFact::MaySuspend);
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
fn catalog_numerical_effects_agree_with_mir_and_instruction_admission() {
    use runmat_builtins::BuiltinFusionPolicy;

    for name in ["single", "asinh"] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical numerical entry");
        let call = catalog_call(name);
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Candidate);
        assert_eq!(call.effects, BuiltinEffects::none());
        assert_eq!(call.async_behavior, AsyncBehaviorFact::MaySuspend);
        assert!(mir_call_has_fusion_signal(&call), "{name} MIR");
        assert!(
            instr_fusion_signal_kind(&Instr::CallBuiltinMulti(name.into(), 1, 1)).is_some(),
            "{name} instruction"
        );
    }
    for name in ["acosh", "atanh"] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical numerical entry");
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Never);
        assert!(
            !mir_call_has_fusion_signal(&catalog_call(name)),
            "{name} MIR"
        );
        assert!(
            instr_fusion_signal_kind(&Instr::CallBuiltinMulti(name.into(), 1, 1)).is_none(),
            "{name} instruction"
        );
    }
}

#[test]
fn effects_async_services_dynamic_calls_and_sequences_do_not_become_candidates() {
    let call = catalog_call("asinh");
    let mut changed = call.clone();
    changed.effects = BuiltinEffects::unknown();
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.effects = BuiltinEffects::none().with_host_callback();
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.effects = BuiltinEffects::none().with_network();
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.workspace_effect = Some(BuiltinWorkspaceEffect::ReadsWorkspace);
    assert!(!mir_call_has_fusion_signal(&changed));
    changed = call.clone();
    changed.environment_effect = Some(BuiltinEnvironmentEffect::PathMutation);
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

#[test]
fn canonical_matrix_product_has_exact_structural_call_contract() {
    use crate::accel::fusion_declaration::{call_kind, canonical_kind};
    use crate::bytecode::FusionInstructionKind;
    assert!(matches!(
        call_kind("mtimes", 2, 1),
        Some(FusionInstructionKind::Matmul)
    ));
    for inputs in [0, 1, 3] {
        assert!(call_kind("mtimes", inputs, 1).is_none());
    }
    for outputs in [0, 2, 3] {
        assert!(call_kind("mtimes", 2, outputs).is_none());
    }
    let mut entry = *runmat_builtins::builtin_catalog_entry_by_name("mtimes").unwrap();
    entry.contract.effects = &[runmat_types::EffectKind::HostCallback];
    assert!(canonical_kind(&entry, Some(2)).is_none());
}
