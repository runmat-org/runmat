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
extension!(PATTERNSEARCH_NUMERIC_EXTENSION,"patternsearch-exact-native-numeric","Native integer/logical data and numeric callback results cross an exactly representable binary64 boundary.","RunMat:compatibility:PatternsearchNumericExtension");
extension!(
    PATTERNSEARCH_SINGLE_EXTENSION,
    "patternsearch-single-host",
    "Single problem data and numeric callback results are promoted to host double.",
    "RunMat:compatibility:PatternsearchSingleExtension"
);
extension!(PATTERNSEARCH_RESIDENT_EXTENSION,"patternsearch-explicit-gpu-input","Explicit gpuArray numeric data/controls use a host gather. Automatic numeric carriers remain transparent; unproven intent is refused. This is not a GPU solver.","RunMat:compatibility:PatternsearchResidentExtension");
extension!(PATTERNSEARCH_INTEGER_OPTION_EXTENSION,"patternsearch-integer-option","Native integer scalar controls preserve structural counts exactly and use exact double conversion for continuous values.","RunMat:compatibility:PatternsearchIntegerOptionExtension");
pub const PATTERNSEARCH_EXTENSIONS: &[BuiltinExtensionDescriptor] = &[
    PATTERNSEARCH_NUMERIC_EXTENSION,
    PATTERNSEARCH_SINGLE_EXTENSION,
    PATTERNSEARCH_RESIDENT_EXTENSION,
    PATTERNSEARCH_INTEGER_OPTION_EXTENSION,
];
pub const PATTERNSEARCH_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor { form:"selected patternsearch data and callback results", inputs:&[
        BuiltinIntegerInputCapability { name:"numeric data/callback result", classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly, scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Compatibility admission precedes provider transfer. Every converted integer must be exactly representable as binary64." }],
        computation_domain:BuiltinIntegerComputationDomain::FloatingPoint, output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error, backend:BuiltinIntegerBackendRule::GatherFallback, overload:BuiltinIntegerOverloadKind::Multiple,
        notes:"Solver numeric outputs are double, not native integer optimization arithmetic." },
    BuiltinIntegerCapabilityDescriptor { form:"selected patternsearch controls", inputs:&[
        BuiltinIntegerInputCapability { name:"active scalar count/control", classes:&ALL_INTEGER_CLASSES,
            availability:BuiltinIntegerInputAvailability::RunMatOnly, scalar_double:BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes:"Counts fit usize without an intermediate double conversion; continuous controls require exact conversion." }],
        computation_domain:BuiltinIntegerComputationDomain::Structural, output_class:BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow:BuiltinIntegerOverflowRule::Error, backend:BuiltinIntegerBackendRule::GatherFallback, overload:BuiltinIntegerOverloadKind::StructuralParameter,
        notes:"No callback capture traversal is used to admit scalar controls." },
];
