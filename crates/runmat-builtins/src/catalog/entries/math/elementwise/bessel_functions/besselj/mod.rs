mod documentation;

use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use documentation::DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "J",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First-kind Bessel values, preserving output shape and single/double precision.",
}];
const ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nu",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real single or double order, scalar or equal size to Z.",
};
const ARGUMENT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense real or complex single/double argument, scalar or equal size to nu.",
};
const SCALE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "0 (unscaled) or 1 (multiply by exp(-abs(imag(Z)))).",
};
const INPUTS_2: [BuiltinParamDescriptor; 2] = [ORDER, ARGUMENT];
const INPUTS_3: [BuiltinParamDescriptor; 3] = [ORDER, ARGUMENT, SCALE];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "J = besselj(nu,Z)",
        inputs: &INPUTS_2,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "J = besselj(nu,Z,scale)",
        inputs: &INPUTS_3,
        outputs: &OUTPUTS,
    },
];

pub const BESSELJ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BESSELJ.INVALID_ARGUMENT",
    identifier: Some("RunMat:besselj:InvalidArgument"),
    when: "Argument count or scale selection is invalid.",
    message: "besselj: invalid argument",
};
pub const BESSELJ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BESSELJ.INVALID_INPUT",
    identifier: Some("RunMat:besselj:InvalidInput"),
    when: "Order or argument class is unsupported by the dense host contract.",
    message: "besselj: invalid input",
};
pub const BESSELJ_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BESSELJ.SIZE_MISMATCH",
    identifier: Some("RunMat:besselj:SizeMismatch"),
    when: "nu and Z are neither equal-size nor scalar-expanded.",
    message: "besselj: size mismatch",
};
pub const BESSELJ_ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BESSELJ.NUMERICAL",
    identifier: Some("RunMat:besselj:NumericalFailure"),
    when: "The AMOS computation reports overflow, precision loss, or convergence failure.",
    message: "besselj: numerical failure",
};
pub const BESSELJ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BESSELJ.INTERNAL",
    identifier: Some("RunMat:besselj:Internal"),
    when: "Output construction fails after validated admission.",
    message: "besselj: internal error",
};
pub const BESSELJ_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BESSELJ.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:besselj:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "besselj: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    BESSELJ_ERROR_INVALID_ARGUMENT,
    BESSELJ_ERROR_INVALID_INPUT,
    BESSELJ_ERROR_SIZE_MISMATCH,
    BESSELJ_ERROR_NUMERICAL,
    BESSELJ_ERROR_INTERNAL,
    BESSELJ_ERROR_TOO_MANY_OUTPUTS,
];
pub const BESSELJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "nu",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Fixed-width integer order classes are not silently converted.",
    },
    BuiltinIntegerInputCapability {
        name: "Z",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Fixed-width integer argument classes are not silently converted.",
    },
];
pub const BESSELJ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "besselj with fixed-width integer input",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Use an explicit single/double conversion; ordinary numeric literals are double.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const BESSELJ_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "besselj" },
    category: "math/elementwise",
    documentation: DOCUMENTATION,
    descriptor: &BESSELJ_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::BesselJ),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &[],
    integer_capabilities: &BESSELJ_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
