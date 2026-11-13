use super::documentation::{
    GET_CURRENT_JOB_DOCUMENTATION, GET_CURRENT_TASK_DOCUMENTATION, GET_CURRENT_WORKER_DOCUMENTATION,
};
use super::*;

mod inference;

const CURRENT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "context",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current parallel execution object, or an empty matrix when none is active.",
}];
const GET_CURRENT_TASK_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "task = getCurrentTask()",
    inputs: &[],
    outputs: &CURRENT_OUTPUT,
}];
const GET_CURRENT_WORKER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "worker = getCurrentWorker()",
        inputs: &[],
        outputs: &CURRENT_OUTPUT,
    }];
const GET_CURRENT_JOB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "job = getCurrentJob()",
    inputs: &[],
    outputs: &CURRENT_OUTPUT,
}];

pub const GET_CURRENT_TASK_DESCRIPTOR: BuiltinDescriptor =
    current_descriptor(&GET_CURRENT_TASK_SIGNATURES);
pub const GET_CURRENT_WORKER_DESCRIPTOR: BuiltinDescriptor =
    current_descriptor(&GET_CURRENT_WORKER_SIGNATURES);
pub const GET_CURRENT_JOB_DESCRIPTOR: BuiltinDescriptor =
    current_descriptor(&GET_CURRENT_JOB_SIGNATURES);

const CURRENT_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.PARALLEL.CURRENT_TOO_MANY_INPUTS",
        identifier: Some("RunMat:parallel:TooManyInputs"),
        when: "An input argument is supplied.",
        message: "current-execution introspection does not accept input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.PARALLEL.RUNTIME_CONTEXT_UNAVAILABLE",
        identifier: Some("RunMat:parallel:RuntimeContextUnavailable"),
        when: "No active runtime context is installed for the call.",
        message: "no active runtime context",
    },
];

const fn current_descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &CURRENT_ERRORS,
    }
}
macro_rules! current_execution_entry {
    ($constant:ident, $name:literal, $rule:expr, $documentation:expr, $descriptor:ident) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "parallel",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                // The result depends on the scoped execution identity: it is
                // empty outside that scope and an object inside it.
                maturity: BuiltinContractMaturity::DynamicByDesign,
                inference_rule: $rule,
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::DeterministicReadOnly,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &CURRENT_EFFECTS,
                capabilities: &PARALLEL_RUNTIME,
            },
            placement: CURRENT_PLACEMENT,
            link: CURRENT_LINK,
            bindings: &crate::REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: &[],
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}

current_execution_entry!(
    GET_CURRENT_TASK_CATALOG_ENTRY,
    "getCurrentTask",
    BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(inference::infer)),
    GET_CURRENT_TASK_DOCUMENTATION,
    GET_CURRENT_TASK_DESCRIPTOR
);
current_execution_entry!(
    GET_CURRENT_WORKER_CATALOG_ENTRY,
    "getCurrentWorker",
    BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(inference::infer)),
    GET_CURRENT_WORKER_DOCUMENTATION,
    GET_CURRENT_WORKER_DESCRIPTOR
);
current_execution_entry!(
    GET_CURRENT_JOB_CATALOG_ENTRY,
    "getCurrentJob",
    BuiltinInferenceRule::Identity(crate::IdentityInferenceRule::new(inference::infer)),
    GET_CURRENT_JOB_DOCUMENTATION,
    GET_CURRENT_JOB_DESCRIPTOR
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &GET_CURRENT_JOB_CATALOG_ENTRY,
    &GET_CURRENT_TASK_CATALOG_ENTRY,
    &GET_CURRENT_WORKER_CATALOG_ENTRY,
];
