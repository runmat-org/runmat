macro_rules! moving_descriptor {
    ($name:literal, $nan_default:literal) => {
        const NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
            default: Some($nan_default),
            ..crate::catalog::entries::math::reduction::moving_window_contract::PARAM_NANFLAG
        };
        const INPUTS_WITH_NAN: [BuiltinParamDescriptor; 3] = [
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_A,
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_K,
            NANFLAG,
        ];
        const INPUTS_WITH_NAN_OPTIONS: [BuiltinParamDescriptor; 4] = [
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_A,
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_K,
            NANFLAG,
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_OPTIONS,
        ];
        const INPUTS_WITH_DIM_NAN_OPTIONS: [BuiltinParamDescriptor; 5] = [
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_A,
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_K,
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_DIM,
            NANFLAG,
            crate::catalog::entries::math::reduction::moving_window_contract::PARAM_OPTIONS,
        ];
        const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, dim)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_DIM,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, nanflag)"),
                inputs: &INPUTS_WITH_NAN,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_NV,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, nanflag, Name, Value)"),
                inputs: &INPUTS_WITH_NAN_OPTIONS,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, dim, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_OPTIONS,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, dim, nanflag, Name, Value)"),
                inputs: &INPUTS_WITH_DIM_NAN_OPTIONS,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
        ];

        const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
            when: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_ARGUMENT.when,
            message: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_ARGUMENT.message,
        };

        const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_INPUT.when,
            message: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_INPUT.message,
        };

        const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INTERNAL.when,
            message: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INTERNAL.message,
        };

        const ERRORS: [BuiltinErrorDescriptor; 5] =
            [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL, ERROR_TOO_MANY_OUTPUTS, crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_HOST_WORKSPACE];
        pub const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: concat!("RM.", $name, ".TOO_MANY_OUTPUTS"), identifier: Some(concat!("RunMat:", $name, ":TooManyOutputs")), when: "More than one output is requested.", message: concat!($name, ": too many output arguments") };

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}
pub(in crate::catalog::entries::math::reduction) use moving_descriptor;

macro_rules! moving_stdvar_descriptor {
    ($name:literal) => {
        const SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_DIM,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, nanflag)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_NANFLAG,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, nanflag)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_NANFLAG,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim, nanflag)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_DIM_NANFLAG,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_NV,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_NV,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, nanflag, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_NANFLAG_NV,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_DIM_NV,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim, nanflag, Name, Value)"),
                inputs: &crate::catalog::entries::math::reduction::moving_window_contract::INPUTS_A_K_W_DIM_NANFLAG_NV,
                outputs: &crate::catalog::entries::math::reduction::moving_window_contract::OUTPUT_Y,
            },
        ];

        const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
            when: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_ARGUMENT.when,
            message: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_ARGUMENT.message,
        };

        const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_INPUT.when,
            message: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INVALID_INPUT.message,
        };

        const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INTERNAL.when,
            message: crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_INTERNAL.message,
        };

        const ERRORS: [BuiltinErrorDescriptor; 5] =
            [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL, ERROR_TOO_MANY_OUTPUTS, crate::catalog::entries::math::reduction::moving_window_contract::MOVING_WINDOW_ERROR_HOST_WORKSPACE];
        pub const ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: concat!("RM.", $name, ".TOO_MANY_OUTPUTS"), identifier: Some(concat!("RunMat:", $name, ":TooManyOutputs")), when: "More than one output is requested.", message: concat!($name, ": too many output arguments") };

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}
pub(in crate::catalog::entries::math::reduction) use moving_stdvar_descriptor;
