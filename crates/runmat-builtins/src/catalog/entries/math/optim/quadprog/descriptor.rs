use crate::*;
pub const QUADPROG_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor {
        form:"quadprog(integer-valued numeric solver data)",
        inputs:&[BuiltinIntegerInputCapability{name:"H, f, constraint, initial-point and bound data",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Native integer data must be exactly representable as binary64; lossy values reject before numerical conversion."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::Multiple,notes:"RunMat numeric boundary; not MATLAB integer-overload parity or exact-integer optimization.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form:"quadprog(...,options)",
        inputs:&[BuiltinIntegerInputCapability{name:"MaxIter",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Nonnegative structural counts decode exactly and must fit u32; zero is honored."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Original total invocation iteration budget, including any auxiliary feasibility phase.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form:"quadprog(...,integer_tolerances)",
        inputs:&[BuiltinIntegerInputCapability{name:"TolFun and TolCon",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Positive finite tolerances cross an exact-binary64 numeric boundary under the numeric-option extension."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Tolerances are not iteration counts and are never relaxed by a lower floor.",
    },
];
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
const INPUTS: [BuiltinParamDescriptor; 10] = [
    input(
        "H",
        BuiltinParamType::NumericArray,
        false,
        "Real symmetric positive semidefinite quadratic coefficient matrix, dense or sparse.",
    ),
    input(
        "f",
        BuiltinParamType::NumericArray,
        false,
        "Linear objective coefficients; [] is the zero vector.",
    ),
    input(
        "A",
        BuiltinParamType::NumericArray,
        true,
        "Linear inequality coefficients.",
    ),
    input(
        "b",
        BuiltinParamType::NumericArray,
        true,
        "Linear inequality right-hand sides.",
    ),
    input(
        "Aeq",
        BuiltinParamType::NumericArray,
        true,
        "Linear equality coefficients.",
    ),
    input(
        "beq",
        BuiltinParamType::NumericArray,
        true,
        "Linear equality right-hand sides.",
    ),
    input(
        "lb",
        BuiltinParamType::NumericArray,
        true,
        "Lower bounds; omitted trailing elements are unbounded.",
    ),
    input(
        "ub",
        BuiltinParamType::NumericArray,
        true,
        "Upper bounds; omitted trailing elements are unbounded.",
    ),
    input(
        "x0",
        BuiltinParamType::NumericArray,
        true,
        "Validated initial point; ignored by interior-point-convex.",
    ),
    input(
        "options",
        BuiltinParamType::Any,
        true,
        "Consumed quadprog options or [].",
    ),
];
const OUTPUTS:[BuiltinParamDescriptor;5]=[
    input("x",BuiltinParamType::Any,false,"Decision column, or a new immutable QuadprogWarmStart in the ws form; evaluated current iterate on an active-set stop, empty on terminal failure."),
    input("fval",BuiltinParamType::NumericArray,false,"0.5*x'*H*x + f'*x, or [] when unavailable."),
    input("exitflag",BuiltinParamType::NumericScalar,false,"Verified optimization status."),
    input("output",BuiltinParamType::Any,false,"Actual iteration, algorithm and numerical diagnostic information."),
    input("lambda",BuiltinParamType::Any,false,"Recovered original-row eqlin/ineqlin/lower/upper multipliers at a verified optimum; [] otherwise."),
];
const PROBLEM: [BuiltinParamDescriptor; 1] = [input(
    "problem",
    BuiltinParamType::Any,
    false,
    "Scalar problem struct with H, f, solver, options and optional constraint fields.",
)];
const WARM_INPUTS: [BuiltinParamDescriptor; 9] = [
    input(
        "H",
        BuiltinParamType::NumericArray,
        false,
        "Original quadratic matrix.",
    ),
    input(
        "f",
        BuiltinParamType::NumericArray,
        false,
        "Original linear coefficients, or [].",
    ),
    input(
        "A",
        BuiltinParamType::NumericArray,
        false,
        "Inequality matrix, or [].",
    ),
    input(
        "b",
        BuiltinParamType::NumericArray,
        false,
        "Inequality right-hand sides, or [].",
    ),
    input(
        "Aeq",
        BuiltinParamType::NumericArray,
        false,
        "Equality matrix, or [].",
    ),
    input(
        "beq",
        BuiltinParamType::NumericArray,
        false,
        "Equality right-hand sides, or [].",
    ),
    input(
        "lb",
        BuiltinParamType::NumericArray,
        false,
        "Lower bounds, or [].",
    ),
    input(
        "ub",
        BuiltinParamType::NumericArray,
        false,
        "Upper bounds, or [].",
    ),
    input(
        "ws",
        BuiltinParamType::Any,
        false,
        "Immutable QuadprogWarmStart supplying X and active-set Options.",
    ),
];
pub const QUADPROG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUADPROG.INVALID_ARGUMENT",
    identifier: Some("RunMat:quadprog:InvalidArgument"),
    when: "Invalid arity or requested output count.",
    message: "quadprog: invalid argument",
};
pub const QUADPROG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUADPROG.INVALID_INPUT",
    identifier: Some("RunMat:quadprog:InvalidInput"),
    when: "Invalid numeric geometry, problem fields, or unsupported option.",
    message: "quadprog: invalid input",
};
pub const QUADPROG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)",
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[x,fval,exitflag,output,lambda] = quadprog(problem)",
            inputs: &PROBLEM,
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[wsout,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,lb,ub,ws)",
            inputs: &WARM_INPUTS,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        QUADPROG_ERROR_INVALID_ARGUMENT,
        QUADPROG_ERROR_INVALID_INPUT,
    ],
};
