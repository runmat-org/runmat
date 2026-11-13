//! Executable lifecycle operations use the same canonical binding owner as
//! construction; neither entry introduces a cleanup registry or executor.
use crate::*;
use runmat_types::*;

pub const CANCEL_ERROR_RUNTIME_CONTEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CANCEL.RUNTIME_CONTEXT",
    identifier: Some("RunMat:parallel:RuntimeContextUnavailable"),
    when: "An execution target is cancelled without the selected runtime context.",
    message: "cancel: no selected runtime context",
};
pub const CANCEL_ERROR_EXECUTION_SERVICE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CANCEL.EXECUTION_SERVICE",
    identifier: Some("RunMat:cancel:ExecutionService"),
    when: "The selected execution service rejects cancellation of a future, task, or job.",
    message: "cancel: execution service failed",
};
pub const CANCEL_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CANCEL.OUTPUT_COUNT",
    identifier: Some("RunMat:cancel:TooManyOutputs"),
    when: "More than one output is requested, before cancellation has any effect.",
    message: "cancel: at most one output is supported",
};
pub const ON_CLEANUP_DELETE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.DELETE.OUTPUT_COUNT",
    identifier: Some("RunMat:onCleanup:TooManyOutputs"),
    when: "More than one output is requested, before callback queueing or handle invalidation.",
    message: "onCleanup delete: at most one output is supported",
};
const CALLBACK_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.DELETE.CALLBACK",
    identifier: None,
    when: "A queued cleanup callback fails or its lifecycle dispatch is rejected or abandoned; the original diagnostic and identifier are retained.",
    message: "The cleanup callback or lifecycle host reported a failure.",
};
pub const CANCEL_ERRORS: [BuiltinErrorDescriptor; 6] = [
    super::ON_CLEANUP_ERROR_INVALID_OBJECT,
    super::ON_CLEANUP_ERROR_GC,
    super::ON_CLEANUP_ERROR_HOST,
    CANCEL_ERROR_RUNTIME_CONTEXT,
    CANCEL_ERROR_EXECUTION_SERVICE,
    CANCEL_ERROR_OUTPUTS,
];
pub const ON_CLEANUP_DELETE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    super::ON_CLEANUP_ERROR_INVALID_OBJECT,
    super::ON_CLEANUP_ERROR_GC,
    super::ON_CLEANUP_ERROR_HOST,
    ON_CLEANUP_DELETE_ERROR_OUTPUTS,
    CALLBACK_ERROR,
];
const STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double scalar zero after the operation succeeds.",
}];
const CANCEL_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "target", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required,
    default: None, description: "onCleanup handle, typed Future/Task/Job, parallel.FevalFuture scalar/array, or parallel.FevalOnAllFuture task aggregate.",
}];
const DELETE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Session-owned onCleanup handle to execute once and invalidate.",
}];
const CANCEL_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "cancel(target)",
        inputs: &CANCEL_INPUT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = cancel(target)",
        inputs: &CANCEL_INPUT,
        outputs: &STATUS,
    },
];
const DELETE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = __runmat_oncleanup_delete(cleanupObj)",
    inputs: &DELETE_INPUT,
    outputs: &STATUS,
}];
pub const CANCEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CANCEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CANCEL_ERRORS,
};
pub const ON_CLEANUP_DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DELETE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &ON_CLEANUP_DELETE_ERRORS,
};
pub const CANCEL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Targets are actual lifecycle or typed execution handles, never numeric IDs. All numeric classes, including integers and resident numeric values, are rejected without conversion or gather; success returns double zero.",
};
pub const ON_CLEANUP_DELETE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Internal deletion accepts only an onCleanup handle. Captured numeric values are owned by the existing callback; no numeric input conversion or integer output form exists.",
};

fn valid_target(value: &ValueFact, cancel: bool) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Execution(
            ExecutionFact::Future { .. } | ExecutionFact::Task { .. } | ExecutionFact::Job { .. },
        ) if cancel => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Object(object) => {
            let Some(class) = &object.runtime_class else {
                return true;
            };
            if class.is(StaticClassIdentity::new("onCleanup")) {
                object.handle_semantics != Some(false)
                    && value.shape.element_count().is_none_or(|count| count == 1)
            } else {
                cancel
                    && (class.is(StaticClassIdentity::new("parallel.FevalFuture"))
                        || (class.is(StaticClassIdentity::new("parallel.FevalOnAllFuture"))
                            && value.shape.element_count().is_none_or(|count| count == 1)))
            }
        }
        _ => false,
    }
}
fn infer_operation(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    cancel: bool,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CLEANUP-OPERATION-ARITY",
            "exactly one lifecycle or execution target is required",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !valid_target(value, cancel))
    {
        diagnostics.push(crate::catalog::inference::argument_error(
            super::ON_CLEANUP_ERROR_INVALID_OBJECT.code,
            if cancel {
                "expected an onCleanup or supported execution handle"
            } else {
                "expected an onCleanup handle"
            },
            0,
        ));
    }
    crate::catalog::inference::finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}
fn infer_cancel(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_operation(request, entry, true)
}
fn infer_delete(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_operation(request, entry, false)
}
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Process,
    artifact_dependencies: &[],
};
pub const CANCEL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "cancel" },
    category: "introspection",
    documentation: super::documentation_cancel::DOCUMENTATION,
    descriptor: &CANCEL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer_cancel)),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::HostCallback, EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&CANCEL_INTEGER_AUDIT),
    suppress_auto_output: true,
};
const DELETE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("__runmat_oncleanup_delete"), slug: Some("__runmat_oncleanup_delete"),
    summary: "Execute and invalidate a session-owned onCleanup handle.",
    description: "Internal executable target of the onCleanup delete method. The existing lifecycle registry queues the callback at most once, its dispatch owner starts or joins the queued task, and completion preserves callback errors. After validating session ownership, the handle is invalidated even when dispatch reports failure. A missing or foreign selected lifecycle owner cannot invalidate another session's target. The selected runtime supplies ownership and command profiling; there is no inline fallback executor.",
    keywords: &["onCleanup", "delete", "internal"],
    related: &["onCleanup", "delete", "cancel"],
    example_exemption: Some("HiddenInternal executable implementation of the onCleanup delete method; public onCleanup examples and native/browser lifecycle fixtures cover this route."),
    ..BuiltinDocumentation::EMPTY
};
pub const ON_CLEANUP_DELETE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "__runmat_oncleanup_delete",
    },
    category: "introspection",
    documentation: DELETE_DOCUMENTATION,
    descriptor: &ON_CLEANUP_DELETE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer_delete)),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ],
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&ON_CLEANUP_DELETE_INTEGER_AUDIT),
    suppress_auto_output: true,
};
