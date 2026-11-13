use crate::*;
const fn input(
    name: &'static str,
    ty: BuiltinParamType,
    optional: bool,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: if optional {
            BuiltinParamArity::Optional
        } else {
            BuiltinParamArity::Required
        },
        default: None,
        description,
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[
    input(
        "C",
        BuiltinParamType::NumericArray,
        false,
        "Nonempty real observation coefficient matrix, dense or sparse.",
    ),
    input(
        "d",
        BuiltinParamType::NumericArray,
        false,
        "Real observation vector with one element per C row.",
    ),
    input(
        "options",
        BuiltinParamType::Any,
        true,
        "Consumed TolX, MaxIter and Display controls or [].",
    ),
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[
    input("x",BuiltinParamType::NumericArray,false,"Nonnegative decision column, one element per C column."),
    input("resnorm",BuiltinParamType::NumericScalar,false,"Full squared residual norm."),
    input("residual",BuiltinParamType::NumericArray,false,"Original residual d-C*x, opposite to lsqlin."),
    input("exitflag",BuiltinParamType::NumericScalar,false,"1 verified solution, 0 iteration limit."),
    input("output",BuiltinParamType::Any,false,"Actual accepted-feasibility iterations, active-set algorithm and message."),
    input("lambda",BuiltinParamType::NumericArray,false,"Original dual column C'*(d-C*x), without factor two; nonpositive up to tolerance at convergence."),
];
const PROBLEM: &[BuiltinParamDescriptor] = &[input(
    "problem",
    BuiltinParamType::Any,
    false,
    "Scalar struct with C,d,solver='lsqnonneg' and options.",
)];
pub const LSQNONNEG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONNEG.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsqnonneg:InvalidArgument"),
    when: "Invalid requested output count.",
    message: "lsqnonneg: invalid argument",
};
pub const LSQNONNEG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQNONNEG.INVALID_INPUT",
    identifier: Some("RunMat:lsqnonneg:InvalidInput"),
    when: "Invalid arity, geometry, class, control or problem fields.",
    message: "lsqnonneg: invalid input",
};
pub const LSQNONNEG_ERROR_NUMERICAL:BuiltinErrorDescriptor=BuiltinErrorDescriptor{
    code:"RM.LSQNONNEG.NUMERICAL",identifier:Some("RunMat:lsqnonneg:NumericalFailure"),
    when:"Numerical rank, original residual/dual validation, or representable coordinates cannot be established.",
    message:"lsqnonneg: numerical failure",
};
pub const LSQNONNEG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[x,resnorm,residual,exitflag,output,lambda] = lsqnonneg(C,d,options)",
            inputs: INPUTS,
            outputs: OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[x,resnorm,residual,exitflag,output,lambda] = lsqnonneg(problem)",
            inputs: PROBLEM,
            outputs: OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        LSQNONNEG_ERROR_INVALID_ARGUMENT,
        LSQNONNEG_ERROR_INVALID_INPUT,
        LSQNONNEG_ERROR_NUMERICAL,
    ],
};
macro_rules! extension {
    ($name:ident,$id:literal,$description:literal,$error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}
extension!(
    LSQNONNEG_INTEGER_C_EXTENSION,
    "lsqnonneg-integer-c",
    "Native-integer C is a gated RunMat exact-binary64 extension.",
    "RunMat:compatibility:LsqnonnegIntegerCExtension"
);
extension!(
    LSQNONNEG_INTEGER_D_EXTENSION,
    "lsqnonneg-integer-d",
    "Native-integer d is a gated RunMat exact-binary64 extension.",
    "RunMat:compatibility:LsqnonnegIntegerDExtension"
);
extension!(
    LSQNONNEG_INTEGER_OPTION_EXTENSION,
    "lsqnonneg-integer-option",
    "Native-integer numeric controls are a gated RunMat extension.",
    "RunMat:compatibility:LsqnonnegIntegerOptionExtension"
);
extension!(
    LSQNONNEG_SINGLE_EXTENSION,
    "lsqnonneg-single-data",
    "Single data normalized to double is a gated RunMat host extension.",
    "RunMat:compatibility:LsqnonnegSingleDataExtension"
);
pub const LSQNONNEG_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    LSQNONNEG_INTEGER_C_EXTENSION,
    LSQNONNEG_INTEGER_D_EXTENSION,
    LSQNONNEG_INTEGER_OPTION_EXTENSION,
    LSQNONNEG_SINGLE_EXTENSION,
];
pub const LSQNONNEG_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{
        form:"lsqnonneg(integer_C,integer_d,options)",
        inputs:&[BuiltinIntegerInputCapability{name:"C and d",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Each data role is separately gated, and all integer values must be exactly representable as binary64."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::Multiple,notes:"No exact-integer solver claim; all public numerical outputs are double.",
    },
    BuiltinIntegerCapabilityDescriptor{
        form:"lsqnonneg(...,integer_options)",
        inputs:&[BuiltinIntegerInputCapability{name:"MaxIter",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Exact nonnegative addressable structural count, including zero."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"The total accepted-solution/feasibility iteration limit never resets after passive-set changes.",
    },
    BuiltinIntegerCapabilityDescriptor{
        form:"lsqnonneg(...,integer_TolX)",
        inputs:&[BuiltinIntegerInputCapability{name:"TolX",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Positive numeric tolerance crosses an exact binary64 boundary."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Tolerance is not an iteration count and is not relaxed by a floor.",
    },
];
