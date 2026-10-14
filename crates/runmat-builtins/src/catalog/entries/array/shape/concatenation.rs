//! Canonical contracts for the two dimension-specific `cat` delegates.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod inference;
#[cfg(test)]
mod tests;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Concatenated array, retaining its class and provider placement where supported.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
];
const HORZ_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = horzcat()",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = horzcat(A1, An...)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
];
const VERT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = vertcat()",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = vertcat(A1, An...)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    },
];
const HORZ_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.HORZCAT.INVALID_INPUT",
        identifier: Some("RunMat:horzcat:InvalidInput"),
        when: "Inputs are malformed or incompatible for horizontal concatenation.",
        message: "horzcat: invalid input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.HORZCAT.TYPE_MISMATCH",
        identifier: Some("RunMat:horzcat:TypeMismatch"),
        when: "Input classes cannot be concatenated together.",
        message: "horzcat: incompatible input classes for concatenation",
    },
];
const VERT_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.VERTCAT.INVALID_INPUT",
        identifier: Some("RunMat:vertcat:InvalidInput"),
        when: "Inputs are malformed or incompatible for vertical concatenation.",
        message: "vertcat: invalid input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.VERTCAT.TYPE_MISMATCH",
        identifier: Some("RunMat:vertcat:TypeMismatch"),
        when: "Input classes cannot be concatenated together.",
        message: "vertcat: incompatible input classes for concatenation",
    },
];
pub const HORZCAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HORZ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HORZ_ERRORS,
};
pub const VERTCAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VERT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VERT_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A1,...,An",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight real integer classes participate; the leftmost integer input selects the output class when integers mix with unlike integer, floating, or logical operands.",
}];
pub const HORZCAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "B = horzcat(A1,...,An) with real integer data",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Saturate,
    backend: BuiltinIntegerBackendRule::HostAndGpu,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "horzcat delegates to cat(dim=2), including exact wide-integer conversion, saturation, empty handling, owner validation, and typed gather/assemble/restore fallback.",
}];
pub const VERTCAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "B = vertcat(A1,...,An) with real integer data",
    inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Saturate,
    backend: BuiltinIntegerBackendRule::HostAndGpu,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "vertcat delegates to cat(dim=1), including exact wide-integer conversion, saturation, empty handling, owner validation, and typed gather/assemble/restore fallback.",
}];

mod documentation;
use documentation::{HORZ_DOC, VERT_DOC};

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];
const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    kind: ConcatKind,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    integers: &'static [BuiltinIntegerCapabilityDescriptor],
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "array/shape",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::ShapeTransform(ShapeTransformKind::Concatenate(
                kind,
            )),
            workspace_effect: None,
            environment_effect: None,
            effects: EFFECTS,
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
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: integers,
        integer_audit: None,
        suppress_auto_output: false,
    }
}

pub const HORZCAT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "horzcat",
    &HORZCAT_DESCRIPTOR,
    HORZ_DOC,
    ConcatKind::Horizontal,
    inference::horizontal,
    &HORZCAT_INTEGER_CAPABILITIES,
);
pub const VERTCAT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "vertcat",
    &VERTCAT_DESCRIPTOR,
    VERT_DOC,
    ConcatKind::Vertical,
    inference::vertical,
    &VERTCAT_INTEGER_CAPABILITIES,
);
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&HORZCAT_CATALOG_ENTRY, &VERTCAT_CATALOG_ENTRY];
