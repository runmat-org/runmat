use crate::*;
macro_rules! extension {
    ($name:ident,$id:literal,$text:literal,$error:literal) => {
        pub const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $text,
            error_identifier: Some($error),
        };
    };
}
extension!(FMINIMAX_NUMERIC_EXTENSION,"fminimax-exact-native-numeric","Exact native-integer/logical data and numeric callback results use a RunMat floating-boundary extension.","RunMat:compatibility:FminimaxNumericExtension");
extension!(FMINIMAX_SINGLE_EXTENSION,"fminimax-single-host","Single data on this ordinary host SQP path is a RunMat extension, not code-generated MATLAB Single support.","RunMat:compatibility:FminimaxSingleExtension");
extension!(FMINIMAX_RESIDENT_EXTENSION,"fminimax-explicit-gpu-input","Explicit gpuArray input uses a RunMat original-context host gather; the solver is not a GPU optimizer.","RunMat:compatibility:FminimaxResidentExtension");
extension!(
    FMINIMAX_INTEGER_OPTION_EXTENSION,
    "fminimax-integer-option",
    "Native-integer scalar solver controls use an exact structural RunMat extension.",
    "RunMat:compatibility:FminimaxIntegerOptionExtension"
);
extension!(FMINIMAX_SPARSE_EXTENSION,"fminimax-sparse-sqp","Sparse matrices/Jacobians are a RunMat extension; MATLAB fminimax does not support sparse data.","RunMat:compatibility:FminimaxSparseSqpExtension");
pub const FMINIMAX_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    FMINIMAX_NUMERIC_EXTENSION,
    FMINIMAX_SINGLE_EXTENSION,
    FMINIMAX_RESIDENT_EXTENSION,
    FMINIMAX_INTEGER_OPTION_EXTENSION,
    FMINIMAX_SPARSE_EXTENSION,
];
pub const FMINIMAX_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{form:"fminimax integer data or numeric callback result",inputs:&[
        BuiltinIntegerInputCapability{name:"numeric data/callback result",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Compatibility admission precedes transfer; only exact binary64 values are converted."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::Multiple,
        notes:"Solver numeric results are double; no native-integer optimization arithmetic is claimed."},
    BuiltinIntegerCapabilityDescriptor{form:"fminimax native-integer options",inputs:&[
        BuiltinIntegerInputCapability{name:"selected scalar control",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Original typed policy admits exact scalar controls before any provider readback."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::StructuralParameter,
        notes:"Count conversion must fit usize; tolerances must be finite and exactly convertible."},
];
