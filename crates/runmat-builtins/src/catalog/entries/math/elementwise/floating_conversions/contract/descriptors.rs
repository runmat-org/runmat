//! Error, public descriptor, and RunMat-only extension declarations shared by
//! the single and double conversion contracts.

macro_rules! define_floating_conversion_descriptors {
    (
        $name:literal,
        $upper:literal,
        $gpu_storage_description:literal,
        $extension_id:literal,
        $extension_description:literal,
        $extension_error:literal
    ) => {
        pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_ARGUMENT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
            when: "Optional arguments are malformed or unsupported, or more than one output is requested.",
            message: concat!($name, ": invalid argument"),
        };
        pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: concat!(
                "Input value or prototype cannot be converted to ",
                $name,
                "."
            ),
            message: concat!($name, ": invalid input"),
        };
        pub const ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".GPU_UNSUPPORTED"),
            identifier: Some(concat!("RunMat:", $name, ":GpuUnsupported")),
            when: concat!(
                "GPU output via \"like\" is requested but no compatible ",
                $gpu_storage_description,
                " provider is active."
            ),
            message: concat!($name, ": gpu output not supported"),
        };
        pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $upper, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: "Internal conversion, gather, or provider upload failed.",
            message: concat!($name, ": internal error"),
        };
        const ERRORS: [BuiltinErrorDescriptor; 4] = [
            ERROR_INVALID_ARGUMENT,
            ERROR_INVALID_INPUT,
            ERROR_GPU_UNSUPPORTED,
            ERROR_INTERNAL,
        ];
        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        pub const LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $extension_id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $extension_description,
            error_identifier: Some($extension_error),
        };
        pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LIKE_EXTENSION];
    };
}
