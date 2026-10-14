//! Hidden MemoizedFunction indexing and invocation binding.

use crate::*;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason};

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MemoizedFunction receiver.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct indexing path.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "varargout",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Result of function invocation or property access.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "varargout = subsref(obj, S)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.MEMOIZE.INVALID_OBJECT",
        identifier: Some("RunMat:memoize:InvalidObject"),
        when: "A memoized-function method receives a non-MemoizedFunction object.",
        message: "memoize: expected MemoizedFunction object",
    },
    BuiltinErrorDescriptor {
        code: "RM.MEMOIZE.INVALID_CACHE_SIZE",
        identifier: Some("RunMat:memoize:InvalidCacheSize"),
        when: "CacheSize is not a positive finite integer scalar.",
        message: "memoize: CacheSize must be a positive integer scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.MEMOIZE.INVALID_INDEX",
        identifier: Some("RunMat:memoize:InvalidIndex"),
        when: "The MemoizedFunction object is indexed with an unsupported indexing form.",
        message: "memoize: unsupported MemoizedFunction indexing",
    },
    BuiltinErrorDescriptor {
        code: "RM.MEMOIZE.GC",
        identifier: Some("RunMat:memoize:GcFailure"),
        when: "The memoized object target cannot be allocated or accessed.",
        message: "memoize: internal object storage failed",
    },
];
pub const MEMOIZED_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &ERRORS,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    runmat_types::infer_call(&contract, request)
}

const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("MemoizedFunction.subsref"), slug: Some("memoizedfunction.subsref"),
    summary: "Dispatch MemoizedFunction indexing and invocation.",
    description: "Internal class indexing hook for MemoizedFunction. Parenthesis indexing invokes the memoized function and manages the cache; member indexing reads supported properties. The invoked callable determines result class, residency, effects, and output count.",
    keywords: &["memoize", "MemoizedFunction", "subsref"], related: &["memoize", "clearCache", "stats"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The standard substruct path preserves chained indexing. Cache hits retain the cached output and cache misses invoke the wrapped callable before inserting the result.", "This hidden binding does not impose a numeric output class or gather resident values. Callable effects and provider capabilities remain dynamic."] }],
    examples: &[], example_exemption: Some("Hidden class method; the public memoize examples exercise MemoizedFunction invocation."),
    faqs: &[], links: &[BuiltinDocumentationLink { label: "memoize", target: BuiltinDocumentationLinkTarget::Documentation("./memoize") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/memoize.rs") }], verification: &[], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub const MEMOIZED_SUBSREF_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "MemoizedFunction.subsref",
    },
    category: "introspection",
    documentation: DOCUMENTATION,
    descriptor: &MEMOIZED_SUBSREF_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::DynamicByDesign,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::HostCallback,
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
            runmat_types::EffectKind::Unknown,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Dynamic,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MEMOIZED_SUBSREF_CATALOG_ENTRY];
