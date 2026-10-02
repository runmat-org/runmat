use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matching graphics handles in search order.",
}];
const FILTERS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filters",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Property predicates, grouping, regular expressions, and logical operators.",
}];
const ROOTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "objhandles",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Search root handle or handle array.",
    },
    BuiltinParamDescriptor {
        name: "filters",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional flat/depth controls and property predicates.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "h = findobj()",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj(prop, value, ...)",
        inputs: &FILTERS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj(objhandles, '-depth', d, filters...)",
        inputs: &ROOTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj(objhandles, 'flat', filters...)",
        inputs: &ROOTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj('-property', prop, filters...)",
        inputs: &FILTERS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj('-regexp', prop, pattern, filters...)",
        inputs: &FILTERS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = findobj({grouped predicates}, filters...)",
        inputs: &FILTERS,
        outputs: &OUTPUT,
    },
];
pub const FINDOBJ_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FINDOBJ.INVALID_ARGUMENT",
    identifier: Some("RunMat:findobj:InvalidArgument"),
    when: "Search roots, depth, operators, property names, or regular expressions are malformed.",
    message: "findobj: invalid argument",
};
pub const FINDOBJ_ERROR_PROPERTY_ACCESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FINDOBJ.PROPERTY_ACCESS",
    identifier: Some("RunMat:findobj:PropertyAccess"),
    when: "A graphics property read or its plotting-owned source readback fails during search.",
    message: "findobj: graphics property access failed",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    FINDOBJ_ERROR_INVALID_ARGUMENT,
    FINDOBJ_ERROR_PROPERTY_ACCESS,
];
pub const FINDOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const FINDOBJ_INTEGER_ROOT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "findobj-integer-root-aliases",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "findobj with typed-integer numeric graphics-handle aliases is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FindobjIntegerRootExtension"),
};
pub const FINDOBJ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [FINDOBJ_INTEGER_ROOT_EXTENSION];

const INTEGER_ROOT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "objhandles", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat mode accepts exactly representable nonnegative host integer aliases for its numeric handle registry; MATLAB accepts graphics objects.",
}];
const INTEGER_DEPTH: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "d", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Depth is decoded exactly as a nonnegative host integer; positive double infinity means unlimited depth.",
}];
const INTEGER_PROPERTY_VALUE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "value", classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Property integers compare class-sensitively from authoritative storage without rounding through double.",
}];
const RESIDENT_INTEGER: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "resident root, depth, or property value",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Resident numeric arrays are not host graphics metadata and are never gathered by findobj.",
}];
pub const FINDOBJ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = findobj(integer_objhandles, ...)", inputs: &INTEGER_ROOT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Exactly representable aliases are gated by RunMat mode; results retain double graphics handles." },
    BuiltinIntegerCapabilityDescriptor { form: "h = findobj(objhandles, '-depth', integer_d, ...)", inputs: &INTEGER_DEPTH, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Depth is range-checked without a floating computation boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "h = findobj(..., prop, integer_value, ...)", inputs: &INTEGER_PROPERTY_VALUE, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Signed and unsigned property values compare exactly, including values above flintmax." },
    BuiltinIntegerCapabilityDescriptor { form: "findobj(resident_integer, ...)", inputs: &RESIDENT_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Resident numeric controls reject before provider lookup or transfer." },
];
