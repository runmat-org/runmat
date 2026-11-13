use super::documentation::{GCP_DOCUMENTATION, PARPOOL_DOCUMENTATION};
use super::*;

mod inference;

const POOL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pool",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Execution pool owned by the current RunMat session.",
}];
const GCP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pool",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current session pool, or an empty double matrix for 'nocreate' without a pool.",
}];
const PARPOOL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional pool kind and worker count.",
}];
const GCP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Use 'nocreate' to inspect the current pool without creating one.",
}];
const PARPOOL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pool = parpool(___)",
    inputs: &PARPOOL_INPUTS,
    outputs: &POOL_OUTPUT,
}];
const GCP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pool = gcp(option)",
    inputs: &GCP_INPUTS,
    outputs: &GCP_OUTPUT,
}];
const PARPOOL_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.POOL_INVALID_INPUT",
    identifier: Some("RunMat:parpool:InvalidInput"),
    when: "The requested pool kind, worker count, or option is invalid for the active host.",
    message: "parpool: invalid pool request",
}];
const GCP_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.GCP_INVALID_INPUT",
    identifier: Some("RunMat:gcp:InvalidInput"),
    when: "An argument other than 'nocreate' is supplied, or more than one argument is supplied.",
    message: "gcp accepts no arguments or the option 'nocreate'",
}];
pub const PARPOOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARPOOL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PARPOOL_ERRORS,
};
pub const GCP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GCP_ERRORS,
};
parallel_entry!(
    PARPOOL_CATALOG_ENTRY,
    "parpool",
    BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(inference::infer_parpool)),
    PARPOOL_DOCUMENTATION,
    PARPOOL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &POOL_WRITE_EFFECTS,
    BuiltinSemanticKind::ExecutorIntrinsic(crate::BuiltinExecutorIntrinsic::Parallel(
        crate::ParallelExecutorIntrinsic::EnsurePool
    ))
);
parallel_entry!(
    GCP_CATALOG_ENTRY,
    "gcp",
    BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(inference::infer_gcp)),
    GCP_DOCUMENTATION,
    GCP_DESCRIPTOR,
    BuiltinContractMaturity::DynamicByDesign,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &POOL_WRITE_EFFECTS,
    BuiltinSemanticKind::ExecutorIntrinsic(crate::BuiltinExecutorIntrinsic::Parallel(
        crate::ParallelExecutorIntrinsic::CurrentPool
    ))
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GCP_CATALOG_ENTRY, &PARPOOL_CATALOG_ENTRY];
