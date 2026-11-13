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
extension!(FMINCON_NUMERIC_EXTENSION,"fmincon-exact-native-numeric","Exact native-integer/logical data and numeric callback results use a RunMat floating-boundary extension.","RunMat:compatibility:FminconNumericExtension");
extension!(FMINCON_SINGLE_EXTENSION,"fmincon-single-host","Single data on this ordinary host SQP path is a RunMat extension, not code-generated MATLAB Single support.","RunMat:compatibility:FminconSingleExtension");
extension!(FMINCON_RESIDENT_EXTENSION,"fmincon-explicit-gpu-input","Explicit gpuArray input uses a RunMat original-context host gather; the solver is not a GPU optimizer.","RunMat:compatibility:FminconResidentExtension");
extension!(
    FMINCON_INTEGER_OPTION_EXTENSION,
    "fmincon-integer-option",
    "Native-integer scalar solver controls use an exact structural RunMat extension.",
    "RunMat:compatibility:FminconIntegerOptionExtension"
);
extension!(FMINCON_SPARSE_EXTENSION,"fmincon-sparse-sqp","Sparse inputs and analytic Jacobians on the SQP and active-set paths are a RunMat extension; MATLAB SQP and active-set do not support sparse data.","RunMat:compatibility:FminconSparseSqpExtension");
pub const FMINCON_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    FMINCON_NUMERIC_EXTENSION,
    FMINCON_SINGLE_EXTENSION,
    FMINCON_RESIDENT_EXTENSION,
    FMINCON_INTEGER_OPTION_EXTENSION,
    FMINCON_SPARSE_EXTENSION,
];
pub const FMINCON_INTEGER_CAPABILITIES:&[BuiltinIntegerCapabilityDescriptor]=&[
    BuiltinIntegerCapabilityDescriptor{form:"fmincon integer data or numeric callback result",inputs:&[
        BuiltinIntegerInputCapability{name:"numeric data/callback result",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Compatibility admission precedes transfer; only exact binary64 values are converted."}],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint,output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::Multiple,
        notes:"Solver numeric results are double; no native-integer optimization arithmetic is claimed."},
    BuiltinIntegerCapabilityDescriptor{form:"fmincon native-integer options",inputs:&[
        BuiltinIntegerInputCapability{name:"selected scalar control",classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly,scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Original typed policy admits exact scalar controls before any provider readback."}],
        computation_domain:BuiltinIntegerComputationDomain::Structural,output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error,backend:BuiltinIntegerBackendRule::GatherFallback,overload:BuiltinIntegerOverloadKind::StructuralParameter,
        notes:"Count conversion must fit usize; tolerances must be finite and exactly convertible."},
];
