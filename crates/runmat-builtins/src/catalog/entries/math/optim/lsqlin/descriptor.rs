use crate::*;
pub const LSQLIN_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor {
        form:"lsqlin(integer-valued numeric solver data)",
        inputs:&[BuiltinIntegerInputCapability{name:"C, d, constraint, initial-point and bound data",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Native integer data must be exactly representable as binary64; lossy values reject before numerical conversion."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::Multiple,notes:"RunMat numeric boundary; not MATLAB integer-overload parity or exact-integer optimization.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form:"lsqlin(...,options)",
        inputs:&[BuiltinIntegerInputCapability{name:"MaxIter",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,
            notes:"Nonnegative structural counts decode exactly and must fit u32; zero is honored."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::Double,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,
        overload:BuiltinIntegerOverloadKind::StructuralParameter,notes:"Original total invocation iteration budget, including any auxiliary feasibility phase.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form:"lsqlin(...,integer_tolerances)",
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
        "C",
        BuiltinParamType::NumericArray,
        false,
        "Real nonempty residual coefficient matrix, dense or sparse.",
    ),
    input(
        "d",
        BuiltinParamType::NumericArray,
        false,
        "Finite real observation vector with one element per C row.",
    ),
    input(
        "A",
        BuiltinParamType::NumericArray,
        false,
        "Linear inequality coefficients.",
    ),
    input(
        "b",
        BuiltinParamType::NumericArray,
        false,
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
        "Initial array in original column order; reflective execution uses it and interior-point ignores it.",
    ),
    input(
        "options",
        BuiltinParamType::Any,
        true,
        "Consumed lsqlin options or [].",
    ),
];
const OUTPUTS:[BuiltinParamDescriptor;6]=[
    input("x",BuiltinParamType::Any,false,"Decision column, or a new immutable LsqlinWarmStart for the warm-start calling form; empty numeric decision on terminal failure."),
    input("resnorm",BuiltinParamType::NumericArray,false,"Full sum of squared original residuals, or [] when unavailable."),
    input("residual",BuiltinParamType::NumericArray,false,"Original residual column C*x-d, or [] when unavailable."),
    input("exitflag",BuiltinParamType::NumericScalar,false,"Verified optimization status."),
    input("output",BuiltinParamType::Any,false,"Actual iteration, algorithm and numerical diagnostic information."),
    input("lambda",BuiltinParamType::Any,false,"Recovered original-row eqlin/ineqlin/lower/upper multipliers at a verified optimum; [] otherwise."),
];
const PROBLEM: [BuiltinParamDescriptor; 1] = [input(
    "problem",
    BuiltinParamType::Any,
    false,
    "Scalar problem struct with C, d, solver, options and optional constraint fields.",
)];
const WARM: [BuiltinParamDescriptor; 9] = [
    input("C", BuiltinParamType::NumericArray, false, "Residual coefficient matrix."),
    input("d", BuiltinParamType::NumericArray, false, "Observation column."),
    input("A", BuiltinParamType::NumericArray, false, "Inequality coefficients or []."),
    input("b", BuiltinParamType::NumericArray, false, "Inequality right-hand sides or []."),
    input("Aeq", BuiltinParamType::NumericArray, false, "Equality coefficients or []."),
    input("beq", BuiltinParamType::NumericArray, false, "Equality right-hand sides or []."),
    input("lb", BuiltinParamType::NumericArray, false, "Lower bounds or []."),
    input("ub", BuiltinParamType::NumericArray, false, "Upper bounds or []."),
    input("ws", BuiltinParamType::Any, false, "Immutable LsqlinWarmStart containing active-set options, initial point and saved working set."),
];
pub const LSQLIN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQLIN.INVALID_ARGUMENT",
    identifier: Some("RunMat:lsqlin:InvalidArgument"),
    when: "Invalid arity or requested output count.",
    message: "lsqlin: invalid argument",
};
pub const LSQLIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSQLIN.INVALID_INPUT",
    identifier: Some("RunMat:lsqlin:InvalidInput"),
    when: "Invalid numeric geometry, problem fields, or unsupported option.",
    message: "lsqlin: invalid input",
};
pub const LSQLIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[wsout,resnorm,residual,exitflag,output,lambda] = lsqlin(C,d,A,b,Aeq,beq,lb,ub,ws)",
            inputs: &WARM,
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[x,resnorm,residual,exitflag,output,lambda] = lsqlin(C,d,A,b,Aeq,beq,lb,ub,x0,options)",
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[x,resnorm,residual,exitflag,output,lambda] = lsqlin(problem)",
            inputs: &PROBLEM,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        LSQLIN_ERROR_INVALID_ARGUMENT,
        LSQLIN_ERROR_INVALID_INPUT,
    ],
};
