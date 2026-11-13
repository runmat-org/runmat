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
            BuiltinAsyncBehavior::MaySuspend | BuiltinAsyncBehavior::MaySuspendLocally => {
                AsyncBehaviorFact::MaySuspend
            }
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
fn empty_private_axis_placeholder_keeps_authentic_reduction_candidate() {
    let ast = runmat_parser::parse("x = 1; y = min(x + x, [], 2);").expect("parse");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mut mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    let entrypoint = hir.assembly.entrypoints[0].id;
    let target = hir.assembly.entrypoints[0].target;
    let (_, groups) = super::derive_semantic_fusion_candidate_groups(&mir, Some(target));
    assert!(
        groups.iter().any(|group| group.signal_count == 2),
        "the Add and selected minimum must remain one MIR candidate across []"
    );
    let bytecode = super::compile(&hir.assembly, &mir, entrypoint).expect("compile");
    assert!(
        bytecode
            .runtime_accel_graph_for_fusion(&bytecode.runtime_fusion_groups())
            .is_some(),
        "the admitted candidate must materialize an authoritative runtime graph"
    );

    let body = mir.bodies.get_mut(&target).expect("entrypoint body");
    let empty_local = body
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .find_map(|statement| match &statement.kind {
            runmat_mir::MirStmtKind::Assign {
                place: runmat_mir::MirPlace::Local(local),
                value:
                    runmat_mir::MirRvalue::Aggregate {
                        kind: runmat_mir::MirAggregateKind::Tensor,
                        elements,
                        ..
                    },
            } if elements.is_empty() => Some(*local),
            _ => None,
        })
        .expect("compiler-owned [] control");
    body.locals[empty_local.0].kind = runmat_mir::MirLocalKind::Binding;
    let (_, forged) = super::derive_semantic_fusion_candidate_groups(&mir, Some(target));
    assert!(
        forged.is_empty(),
        "a bound/user-visible empty tensor must break this candidate"
    );
}

#[test]
fn nonempty_literal_control_is_not_a_transparent_candidate_bridge() {
    let ast = runmat_parser::parse("x = 1; y = min(x + x, [1], 2);").expect("parse");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    let target = hir.assembly.entrypoints[0].target;
    let (_, groups) = super::derive_semantic_fusion_candidate_groups(&mir, Some(target));
    assert!(
        groups.is_empty(),
        "nonempty control materialization cannot bridge producer and reduction"
    );
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
    for (name, predicate) in [
        ("acosh", runmat_types::RealUnaryPredicate::AtLeastOne),
        (
            "atanh",
            runmat_types::RealUnaryPredicate::ClosedUnitInterval,
        ),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("canonical numerical entry");
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Candidate);
        let [binding] = entry.bindings else {
            panic!("{name}: expected one selected binding");
        };
        let declaration = runmat_runtime::builtins::common::spec::canonical_fusion_declaration(
            entry.binding_identity(binding),
        )
        .expect("selected fusion declaration");
        let semantics = declaration.semantics.expect("selected numerical admission");
        assert_eq!(
            semantics.required_real_unary.map(|(actual, _)| actual),
            Some(predicate)
        );
        assert!(semantics.admit.is_some(), "{name}: runtime domain proof");
        assert!(
            semantics.admit_static.is_none(),
            "{name}: no speculative static proof"
        );
        assert!(
            mir_call_has_fusion_signal(&catalog_call(name)),
            "{name} MIR"
        );
        assert!(
            instr_fusion_signal_kind(&Instr::CallBuiltinMulti(name.into(), 1, 1)).is_some(),
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
        owned_worker: runmat_builtins::BuiltinOwnedWorkerPolicy::Forbidden,
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
