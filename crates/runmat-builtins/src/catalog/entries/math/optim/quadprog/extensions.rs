use crate::*;
macro_rules! integer_extension {
    ($name:ident,$id:literal,$role:literal,$error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!(
                "quadprog native-integer ",
                $role,
                " data is a RunMat extension"
            ),
            error_identifier: Some($error),
        };
    };
}
integer_extension!(
    QUADPROG_INTEGER_H_EXTENSION,
    "quadprog-integer-h",
    "H",
    "RunMat:compatibility:QuadprogIntegerHExtension"
);
integer_extension!(
    QUADPROG_INTEGER_F_EXTENSION,
    "quadprog-integer-f",
    "f",
    "RunMat:compatibility:QuadprogIntegerFExtension"
);
integer_extension!(
    QUADPROG_INTEGER_A_EXTENSION,
    "quadprog-integer-a",
    "A",
    "RunMat:compatibility:QuadprogIntegerAExtension"
);
integer_extension!(
    QUADPROG_INTEGER_B_EXTENSION,
    "quadprog-integer-b",
    "B",
    "RunMat:compatibility:QuadprogIntegerBExtension"
);
integer_extension!(
    QUADPROG_INTEGER_AEQ_EXTENSION,
    "quadprog-integer-aeq",
    "AEQ",
    "RunMat:compatibility:QuadprogIntegerAEQExtension"
);
integer_extension!(
    QUADPROG_INTEGER_BEQ_EXTENSION,
    "quadprog-integer-beq",
    "BEQ",
    "RunMat:compatibility:QuadprogIntegerBEQExtension"
);
integer_extension!(
    QUADPROG_INTEGER_LB_EXTENSION,
    "quadprog-integer-lb",
    "LB",
    "RunMat:compatibility:QuadprogIntegerLBExtension"
);
integer_extension!(
    QUADPROG_INTEGER_UB_EXTENSION,
    "quadprog-integer-ub",
    "UB",
    "RunMat:compatibility:QuadprogIntegerUBExtension"
);
integer_extension!(
    QUADPROG_INTEGER_X0_EXTENSION,
    "quadprog-integer-x0",
    "X0",
    "RunMat:compatibility:QuadprogIntegerX0Extension"
);
integer_extension!(
    QUADPROG_INTEGER_OPTION_EXTENSION,
    "quadprog-integer-option",
    "OPTION",
    "RunMat:compatibility:QuadprogIntegerOPTIONExtension"
);
pub const QUADPROG_SINGLE_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"quadprog-single-host",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Single data on the ordinary host interior-point path is a RunMat extension; MATLAB restricts Single to code generation/UseCodegenSolver",
    error_identifier:Some("RunMat:compatibility:QuadprogSingleHostExtension"),
};
pub const QUADPROG_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    QUADPROG_INTEGER_H_EXTENSION,
    QUADPROG_INTEGER_F_EXTENSION,
    QUADPROG_INTEGER_A_EXTENSION,
    QUADPROG_INTEGER_B_EXTENSION,
    QUADPROG_INTEGER_AEQ_EXTENSION,
    QUADPROG_INTEGER_BEQ_EXTENSION,
    QUADPROG_INTEGER_LB_EXTENSION,
    QUADPROG_INTEGER_UB_EXTENSION,
    QUADPROG_INTEGER_X0_EXTENSION,
    QUADPROG_INTEGER_OPTION_EXTENSION,
    QUADPROG_SINGLE_EXTENSION,
    QUADPROG_GPU_INPUT_EXTENSION,
];

pub const QUADPROG_GPU_INPUT_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"quadprog-explicit-gpu-input",mode:BuiltinExtensionMode::RunMatOnly,
    description:"quadprog explicit gpuArray input uses a RunMat host-gather extension, not a device optimizer",
    error_identifier:Some("RunMat:compatibility:QuadprogExplicitGpuInputExtension"),
};
