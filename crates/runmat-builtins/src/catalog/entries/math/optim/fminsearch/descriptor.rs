use crate::*;
const fn param(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const INPUTS: [BuiltinParamDescriptor; 2] = [
    param("fun", "Real scalar objective callback."),
    param("x0", "Nonempty finite real double initial array."),
];
const OPTIONS: [BuiltinParamDescriptor; 3] = [
    param("fun", "Real scalar objective callback."),
    param("x0", "Nonempty finite real double initial array."),
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Legacy optimset options structure or [].",
    },
];
const PROBLEM: [BuiltinParamDescriptor; 1] = [param(
    "problem",
    "Scalar structure with objective, x0, solver and optional options.",
)];
const OUTPUTS: [BuiltinParamDescriptor; 4] = [
    param("x", "Best evaluated point, preserving x0 shape."),
    param("fval", "Objective at x."),
    param(
        "exitflag",
        "1: simplex converged; 0: budget; -1: observer stop.",
    ),
    param("output", "iterations, funcCount, algorithm and message."),
];
const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "x = fminsearch(fun, x0)",
        inputs: &INPUTS,
        outputs: &[param("x", "Best evaluated point, preserving x0 shape.")],
    },
    BuiltinSignatureDescriptor {
        label: "x = fminsearch(fun, x0, options)",
        inputs: &OPTIONS,
        outputs: &[param("x", "Best evaluated point, preserving x0 shape.")],
    },
    BuiltinSignatureDescriptor {
        label: "x = fminsearch(problem)",
        inputs: &PROBLEM,
        outputs: &[param("x", "Best evaluated point, preserving x0 shape.")],
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = fminsearch(...)",
        inputs: &OPTIONS,
        outputs: &[
            param("x", "Best evaluated point, preserving x0 shape."),
            param("fval", "Objective at x."),
        ],
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = fminsearch(...)",
        inputs: &OPTIONS,
        outputs: &[
            param("x", "Best evaluated point, preserving x0 shape."),
            param("fval", "Objective at x."),
            param(
                "exitflag",
                "1: simplex converged; 0: budget; -1: observer stop.",
            ),
        ],
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fminsearch(...)",
        inputs: &OPTIONS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = fminsearch(problem)",
        inputs: &PROBLEM,
        outputs: &OUTPUTS,
    },
];
pub const FMINSEARCH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FMINSEARCH.INVALID_ARGUMENT",
    identifier: Some("RunMat:fminsearch:InvalidArgument"),
    when: "Invalid argument, unsupported option, class or objective contract.",
    message: "fminsearch: invalid argument",
};
pub const FMINSEARCH_ERROR_INVALID_FUNCTION_VALUE: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.FMINSEARCH.INVALID_FUNCTION_VALUE",
        identifier: Some("RunMat:fminsearch:InvalidFunctionValue"),
        when: "FunValCheck is on and the objective returns NaN.",
        message: "fminsearch: invalid objective value",
    };
pub const FMINSEARCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        FMINSEARCH_ERROR_INVALID_ARGUMENT,
        FMINSEARCH_ERROR_INVALID_FUNCTION_VALUE,
        BuiltinErrorDescriptor{code:"RM.FMINSEARCH.ALLOCATION_FAILED",identifier:Some("RunMat:fminsearch:AllocationFailed"),when:"Fallible host simplex/callback backing allocation fails.",message:"fminsearch: workspace allocation failed"},
        BuiltinErrorDescriptor{code:"RM.FMINSEARCH.INVALID_OBSERVER",identifier:None,when:"OutputFcn/PlotFcns is not a supported callback form or does not return a logical scalar stop flag.",message:"fminsearch: invalid observer contract"},
    ],
};
