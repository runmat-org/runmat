use super::*;

const POOL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pool",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Execution pool owned by the current RunMat session.",
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
    outputs: &POOL_OUTPUT,
}];
const POOL_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.PARALLEL.POOL_INVALID_INPUT",
    identifier: Some("RunMat:parpool:InvalidInput"),
    when: "The requested pool kind, worker count, or option is invalid for the active host.",
    message: "parpool: invalid pool request",
}];
pub const PARPOOL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARPOOL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POOL_ERRORS,
};
pub const GCP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POOL_ERRORS,
};
parallel_entry!(
    PARPOOL_CATALOG_ENTRY,
    "parpool",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Parpool),
    "Create or return the execution pool for the current session.",
    &["parallel", "pool", "workers", "parpool"],
    PARPOOL_DESCRIPTOR,
    BuiltinContractMaturity::Complete,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::Impure,
    &POOL_WRITE_EFFECTS
);
parallel_entry!(
    GCP_CATALOG_ENTRY,
    "gcp",
    BuiltinInferenceRule::Parallel(ParallelInferenceRule::Gcp),
    "Return the current execution pool.",
    &["parallel", "pool", "current", "gcp", "nocreate"],
    GCP_DESCRIPTOR,
    BuiltinContractMaturity::DynamicByDesign,
    BuiltinAsyncBehavior::MaySuspend,
    BuiltinPurity::DeterministicReadOnly,
    &POOL_READ_EFFECTS
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GCP_CATALOG_ENTRY, &PARPOOL_CATALOG_ENTRY];
