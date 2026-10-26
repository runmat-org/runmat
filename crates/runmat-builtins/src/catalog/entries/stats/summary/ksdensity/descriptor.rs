use crate::*;
pub const KSDENSITY_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor {
    code:"RM.KSDENSITY.INVALID_ARGUMENT",identifier:Some("RunMat:ksdensity:InvalidArgument"),
    when:"Sample/point class or geometry, bandwidth identification, numeric controls, supported option surface, device intent or output count is invalid.",message:"ksdensity: invalid argument",
};
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single/double univariate observations; NaNs are omitted.",
};
const POINTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "pts",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Evaluation vector, or numeric [] for the automatic grid.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name_value_pairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Bandwidth, NumPoints, Function, Kernel and Support controls.",
};
const AXES:BuiltinParamDescriptor=BuiltinParamDescriptor{name:"ax",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Typed axes target for the zero-output plotting form; numeric handle IDs are not used to reinterpret scalar samples."};
const OUTPUTS:[BuiltinParamDescriptor;3]=[
    BuiltinParamDescriptor{name:"f",ty:BuiltinParamType::NumericArray,arity:BuiltinParamArity::Required,default:None,description:"Double PDF, CDF or survivor estimates, with the evaluation-point shape."},
    BuiltinParamDescriptor{name:"xi",ty:BuiltinParamType::NumericArray,arity:BuiltinParamArity::Optional,default:None,description:"Explicit points with their original floating class/shape, or a generated double row vector."},
    BuiltinParamDescriptor{name:"bw",ty:BuiltinParamType::NumericArray,arity:BuiltinParamArity::Optional,default:None,description:"Positive reusable double bandwidth."},
];
pub const KSDENSITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[f,xi,bw] = ksdensity(x,Name,Value,...)",
            inputs: &[X, OPTIONS],
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[f,xi,bw] = ksdensity(x,pts,Name,Value,...)",
            inputs: &[X, POINTS, OPTIONS],
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "ksdensity(ax,x,Name,Value,...)",
            inputs: &[AXES, X, OPTIONS],
            outputs: &[],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[KSDENSITY_ERROR_INVALID_ARGUMENT],
};
