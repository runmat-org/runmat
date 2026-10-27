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
extension!(FSEMINF_NUMERIC_EXTENSION,"fseminf-exact-native-numeric","Numeric policy identity; the public fseminf ingress has no admitted native-integer/logical problem-data overload.","RunMat:compatibility:FseminfNumericExtension");
extension!(FSEMINF_SINGLE_EXTENSION,"fseminf-single-host","Numeric policy identity; public fseminf problem data and callback outputs require host double, not Single.","RunMat:compatibility:FseminfSingleExtension");
extension!(FSEMINF_RESIDENT_EXTENSION,"fseminf-explicit-gpu-input","Explicit gpuArray scalar/vector options use an original-context host gather; problem data and callbacks remain host double.","RunMat:compatibility:FseminfResidentExtension");
extension!(
    FSEMINF_INTEGER_OPTION_EXTENSION,
    "fseminf-integer-option",
    "Native-integer scalar solver controls use an exact structural RunMat extension.",
    "RunMat:compatibility:FseminfIntegerOptionExtension"
);
pub const FSEMINF_EXTENSIONS: &[BuiltinExtensionDescriptor] =
    &[FSEMINF_RESIDENT_EXTENSION, FSEMINF_INTEGER_OPTION_EXTENSION];
pub const FSEMINF_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "fseminf native-integer options",
        inputs: &[BuiltinIntegerInputCapability {
            name: "selected scalar control",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:
                "Original typed policy admits exact scalar controls before any provider readback.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "Count conversion must fit usize; tolerances must be finite and exactly convertible.",
    }];
