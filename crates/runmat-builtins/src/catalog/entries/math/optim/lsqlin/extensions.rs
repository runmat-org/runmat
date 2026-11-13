use crate::*;
macro_rules! integer_extension {
    ($name:ident,$id:literal,$role:literal,$error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: concat!(
                "lsqlin native-integer ",
                $role,
                " data is a RunMat extension"
            ),
            error_identifier: Some($error),
        };
    };
}
integer_extension!(
    LSQLIN_INTEGER_C_EXTENSION,
    "lsqlin-integer-c",
    "C",
    "RunMat:compatibility:LsqlinIntegerCExtension"
);
integer_extension!(
    LSQLIN_INTEGER_D_EXTENSION,
    "lsqlin-integer-d",
    "D",
    "RunMat:compatibility:LsqlinIntegerDExtension"
);
integer_extension!(
    LSQLIN_INTEGER_A_EXTENSION,
    "lsqlin-integer-a",
    "A",
    "RunMat:compatibility:LsqlinIntegerAExtension"
);
integer_extension!(
    LSQLIN_INTEGER_B_EXTENSION,
    "lsqlin-integer-b",
    "B",
    "RunMat:compatibility:LsqlinIntegerBExtension"
);
integer_extension!(
    LSQLIN_INTEGER_AEQ_EXTENSION,
    "lsqlin-integer-aeq",
    "AEQ",
    "RunMat:compatibility:LsqlinIntegerAEQExtension"
);
integer_extension!(
    LSQLIN_INTEGER_BEQ_EXTENSION,
    "lsqlin-integer-beq",
    "BEQ",
    "RunMat:compatibility:LsqlinIntegerBEQExtension"
);
integer_extension!(
    LSQLIN_INTEGER_LB_EXTENSION,
    "lsqlin-integer-lb",
    "LB",
    "RunMat:compatibility:LsqlinIntegerLBExtension"
);
integer_extension!(
    LSQLIN_INTEGER_UB_EXTENSION,
    "lsqlin-integer-ub",
    "UB",
    "RunMat:compatibility:LsqlinIntegerUBExtension"
);
integer_extension!(
    LSQLIN_INTEGER_X0_EXTENSION,
    "lsqlin-integer-x0",
    "X0",
    "RunMat:compatibility:LsqlinIntegerX0Extension"
);
integer_extension!(
    LSQLIN_INTEGER_OPTION_EXTENSION,
    "lsqlin-integer-option",
    "OPTION",
    "RunMat:compatibility:LsqlinIntegerOPTIONExtension"
);
pub const LSQLIN_SINGLE_EXTENSION:BuiltinExtensionDescriptor=BuiltinExtensionDescriptor{
    id:"lsqlin-single-host",mode:BuiltinExtensionMode::RunMatOnly,
    description:"Single data on the ordinary host interior-point path is a RunMat extension; MATLAB restricts Single to code generation/UseCodegenSolver",
    error_identifier:Some("RunMat:compatibility:LsqlinSingleHostExtension"),
};
pub const LSQLIN_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    LSQLIN_INTEGER_C_EXTENSION,
    LSQLIN_INTEGER_D_EXTENSION,
    LSQLIN_INTEGER_A_EXTENSION,
    LSQLIN_INTEGER_B_EXTENSION,
    LSQLIN_INTEGER_AEQ_EXTENSION,
    LSQLIN_INTEGER_BEQ_EXTENSION,
    LSQLIN_INTEGER_LB_EXTENSION,
    LSQLIN_INTEGER_UB_EXTENSION,
    LSQLIN_INTEGER_X0_EXTENSION,
    LSQLIN_INTEGER_OPTION_EXTENSION,
    LSQLIN_SINGLE_EXTENSION,
];
