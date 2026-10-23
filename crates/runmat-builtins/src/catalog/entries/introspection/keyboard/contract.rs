use crate::*;
pub const SIMPLE_NO_OUTPUT_SIGNATURE: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "keyboard",
        inputs: &[],
        outputs: &[],
    }];

pub const KEYBOARD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIMPLE_NO_OUTPUT_SIGNATURE,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &KEYBOARD_PUBLIC_ERRORS,
};

pub const KEYBOARD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KEYBOARD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:keyboard:TooManyOutputs"),
    when: "More outputs are requested than this form supports.",
    message: "keyboard: too many output arguments",
};
const KEYBOARD_PUBLIC_ERRORS: [BuiltinErrorDescriptor; 1] = [KEYBOARD_ERROR_TOO_MANY_OUTPUTS];
