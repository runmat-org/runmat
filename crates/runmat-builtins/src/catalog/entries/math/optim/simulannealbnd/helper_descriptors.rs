use super::descriptor::parameter;
use super::SIMULANNEALBND_ERROR_INVALID_ARGUMENT;
use crate::*;
const STOP: &[BuiltinParamDescriptor] =
    &[parameter("stop", "Logical acceptance or observer result.")];
const POINT: &[BuiltinParamDescriptor] = &[parameter(
    "newx",
    "Actual generated double point preserving x shape.",
)];
const TEMP: &[BuiltinParamDescriptor] = &[parameter(
    "temperature",
    "Actual nonnegative column temperature vector.",
)];
pub const ACCEPTANCESA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "acceptpoint = acceptancesa(optimValues,newx,newfval)",
        inputs: &[
            parameter("optimValues", "Original current solver metadata."),
            parameter("newx", "Actual selected input."),
            parameter("newfval", "Actual selected input."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const ANNEALINGFAST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newx = annealingfast(optimValues,problem)",
        inputs: &[
            parameter("optimValues", "Original current solver metadata."),
            parameter("problem", "objective,x0,nvar,lb,ub."),
        ],
        outputs: POINT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const ANNEALINGBOLTZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newx = annealingboltz(optimValues,problem)",
        inputs: &[
            parameter("optimValues", "Original current solver metadata."),
            parameter("problem", "objective,x0,nvar,lb,ub."),
        ],
        outputs: POINT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const TEMPERATUREEXP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "temperature = temperatureexp(optimValues,options)",
        inputs: &[
            parameter("optimValues", "Original current solver metadata."),
            parameter("options", "Actual selected input."),
        ],
        outputs: TEMP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const TEMPERATUREFAST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "temperature = temperaturefast(optimValues,options)",
        inputs: &[
            parameter("optimValues", "Original current solver metadata."),
            parameter("options", "Actual selected input."),
        ],
        outputs: TEMP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const TEMPERATUREBOLTZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "temperature = temperatureboltz(optimValues,options)",
        inputs: &[
            parameter("optimValues", "Original current solver metadata."),
            parameter("options", "Actual selected input."),
        ],
        outputs: TEMP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SAPLOTBESTF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = saplotbestf(options,optimValues,flag)",
        inputs: &[
            parameter("options", "Actual selected input."),
            parameter("optimValues", "Original current solver metadata."),
            parameter("flag", "init,iter,done."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SAPLOTBESTX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = saplotbestx(options,optimValues,flag)",
        inputs: &[
            parameter("options", "Actual selected input."),
            parameter("optimValues", "Original current solver metadata."),
            parameter("flag", "init,iter,done."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SAPLOTF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = saplotf(options,optimValues,flag)",
        inputs: &[
            parameter("options", "Actual selected input."),
            parameter("optimValues", "Original current solver metadata."),
            parameter("flag", "init,iter,done."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SAPLOTX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = saplotx(options,optimValues,flag)",
        inputs: &[
            parameter("options", "Actual selected input."),
            parameter("optimValues", "Original current solver metadata."),
            parameter("flag", "init,iter,done."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SAPLOTSTOPPING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = saplotstopping(options,optimValues,flag)",
        inputs: &[
            parameter("options", "Actual selected input."),
            parameter("optimValues", "Original current solver metadata."),
            parameter("flag", "init,iter,done."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
pub const SAPLOTTEMPERATURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = saplottemperature(options,optimValues,flag)",
        inputs: &[
            parameter("options", "Actual selected input."),
            parameter("optimValues", "Original current solver metadata."),
            parameter("flag", "init,iter,done."),
        ],
        outputs: STOP,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SIMULANNEALBND_ERROR_INVALID_ARGUMENT],
};
