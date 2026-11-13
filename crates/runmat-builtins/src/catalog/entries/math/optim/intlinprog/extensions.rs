use crate::*;
pub const INTLINPROG_INTEGER_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"intlinprog-native-integer-numeric",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Native-integer numeric data and controls are RunMat extensions; floating data cross an exact binary64 boundary, structural counts retain their exact addressable value.",
    error_identifier:Some("RunMat:compatibility:IntlinprogNativeIntegerExtension"),
};
pub const INTLINPROG_SINGLE_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"intlinprog-single-data",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Single f, intcon, Aeq, lb, ub and x0 normalized to double are RunMat extensions; A, b and beq permit ordinary Single data.",
    error_identifier:Some("RunMat:compatibility:IntlinprogSingleDataExtension"),
};
pub const INTLINPROG_EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[INTLINPROG_INTEGER_EXTENSION, INTLINPROG_SINGLE_EXTENSION];
pub const INTLINPROG_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{
        form:"intlinprog(native-integer numeric operands)",
        inputs:&[BuiltinIntegerInputCapability{name:"f, intcon, A, b, Aeq, beq, lb, ub and x0",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Native integer numeric data must be exactly representable as binary64 before conversion; intcon still selects variable indices, not extended integer-domain objects."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::Multiple,notes:"Exact rational certificates verify the original binary64 model; this is not an exact-integer data optimization overload.",
    },
    BuiltinIntegerCapabilityDescriptor{
        form:"intlinprog(...,native-integer options)",
        inputs:&[BuiltinIntegerInputCapability{name:"MaxNodes and MaxFeasiblePoints",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Positive structural counts decode exactly and must fit usize; MaxFeasiblePoints also permits floating Inf."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Counts are node relaxations/certified feasible candidate discoveries, not interior-point iterations.",
    },
    BuiltinIntegerCapabilityDescriptor{
        form:"intlinprog(...,native-integer scalar tolerances and limits)",
        inputs:&[BuiltinIntegerInputCapability{name:"AbsoluteGapTolerance, RelativeGapTolerance, ConstraintTolerance, MaxTime and ObjectiveCutOff",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Non-count numeric controls must be exactly representable as binary64 and satisfy their own scalar range."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Native integers are not silently accepted as ordinary MATLAB numeric-option forms.",
    },
];
