use crate::*;
const INPUTS:&[BuiltinParamDescriptor]=&[
    BuiltinParamDescriptor{name:"x",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Current solver point. optimplotx reads finite real floating coordinates in column-major order; scalar-history helpers do not read x."},
    BuiltinParamDescriptor{name:"optimValues",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Scalar struct with iteration, and funccount or fval for the corresponding history chart."},
    BuiltinParamDescriptor{name:"state",ty:BuiltinParamType::StringScalar,arity:BuiltinParamArity::Required,default:None,description:"init resets this chart, iter updates it, done leaves the final chart without duplicating its last iteration."},
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[BuiltinParamDescriptor{name:"stop",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Optional,default:None,description:"Logical false after a successful real chart update; execution cancellation remains a typed error."}];
pub const OPTIMPLOTX_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{code:"RM.OPTIMPLOTX.INVALID_ARGUMENT",identifier:Some("RunMat:optimplotx:InvalidArgument"),when:"Callback arguments, x coordinates, iteration, state or output count are invalid, or a needed resident operand has explicit or unproven device intent.",message:"optimplotx: invalid callback argument"};
pub const OPTIMPLOTFUNCCOUNT_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{code:"RM.OPTIMPLOTFUNCCOUNT.INVALID_ARGUMENT",identifier:Some("RunMat:optimplotfunccount:InvalidArgument"),when:"Callback arguments, iteration or funccount, state or output count are invalid, or a needed resident operand has explicit or unproven device intent.",message:"optimplotfunccount: invalid callback argument"};
pub const OPTIMPLOTFVAL_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{code:"RM.OPTIMPLOTFVAL.INVALID_ARGUMENT",identifier:Some("RunMat:optimplotfval:InvalidArgument"),when:"Callback arguments, iteration or scalar fval, state or output count are invalid, or a needed resident operand has explicit or unproven device intent.",message:"optimplotfval: invalid callback argument"};
pub const OPTIMPLOTX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotx(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTX_ERROR_INVALID_ARGUMENT],
};
pub const OPTIMPLOTFUNCCOUNT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotfunccount(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTFUNCCOUNT_ERROR_INVALID_ARGUMENT],
};
pub const OPTIMPLOTFVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotfval(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTFVAL_ERROR_INVALID_ARGUMENT],
};

pub const OPTIMPLOTCONSTRVIOLATION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OPTIMPLOTCONSTRVIOLATION.INVALID_ARGUMENT",
        identifier: Some("RunMat:optimplotconstrviolation:InvalidArgument"),
        when: "Invalid callback arguments, selected metric, state or output count.",
        message: "optimplotconstrviolation: invalid callback argument",
    };
pub const OPTIMPLOTCONSTRVIOLATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotconstrviolation(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTCONSTRVIOLATION_ERROR_INVALID_ARGUMENT],
};

pub const OPTIMPLOTSTEPSIZE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OPTIMPLOTSTEPSIZE.INVALID_ARGUMENT",
        identifier: Some("RunMat:optimplotstepsize:InvalidArgument"),
        when: "Invalid callback arguments, selected metric, state or output count.",
        message: "optimplotstepsize: invalid callback argument",
    };
pub const OPTIMPLOTSTEPSIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotstepsize(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTSTEPSIZE_ERROR_INVALID_ARGUMENT],
};

pub const OPTIMPLOTFIRSTORDEROPT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OPTIMPLOTFIRSTORDEROPT.INVALID_ARGUMENT",
        identifier: Some("RunMat:optimplotfirstorderopt:InvalidArgument"),
        when: "Invalid callback arguments, selected metric, state or output count.",
        message: "optimplotfirstorderopt: invalid callback argument",
    };
pub const OPTIMPLOTFIRSTORDEROPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotfirstorderopt(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTFIRSTORDEROPT_ERROR_INVALID_ARGUMENT],
};

pub const OPTIMPLOTFVALCONSTR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.OPTIMPLOTFVALCONSTR.INVALID_ARGUMENT",
        identifier: Some("RunMat:optimplotfvalconstr:InvalidArgument"),
        when: "Invalid callback arguments, selected metric, state or output count.",
        message: "optimplotfvalconstr: invalid callback argument",
    };
pub const OPTIMPLOTFVALCONSTR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotfvalconstr(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTFVALCONSTR_ERROR_INVALID_ARGUMENT],
};
