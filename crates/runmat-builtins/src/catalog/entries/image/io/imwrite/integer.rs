use crate::*;

const ALL_INTEGER_CLASSES: [BuiltinIntegerClass; 8] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint8,
    BuiltinIntegerClass::Uint16,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];

const IMWRITE_DOCUMENTED_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];
const IMWRITE_REJECTED_INTEGER_CLASSES: [BuiltinIntegerClass; 6] = [
    BuiltinIntegerClass::Int8,
    BuiltinIntegerClass::Int16,
    BuiltinIntegerClass::Int32,
    BuiltinIntegerClass::Int64,
    BuiltinIntegerClass::Uint32,
    BuiltinIntegerClass::Uint64,
];
const IMWRITE_DOCUMENTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "A_or_X", classes: &IMWRITE_DOCUMENTED_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Uint8 and uint16 direct or indexed images retain their native sample/index interpretation through the host encoder sink." }];
const IMWRITE_REJECTED_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "A_or_X", classes: &IMWRITE_REJECTED_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Signed integer, uint32, and uint64 image arrays are outside the documented imwrite image-data surface and reject before file effects." }];
const IMWRITE_ALPHA_INTEGER_CLASSES: [BuiltinIntegerClass; 2] =
    [BuiltinIntegerClass::Uint8, BuiltinIntegerClass::Uint16];
const IMWRITE_ALPHA_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "Alpha", classes: &IMWRITE_ALPHA_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Documented uint8 and uint16 Alpha matrices are read from authoritative storage and scaled by their own full class range into the encoded channel." }];
const IMWRITE_CONTROL_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability { name: "integer-valued format control", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "RowsPerStrip is documented for every integer class. Other integer-valued controls are format-specific; implemented controls read exact scalar storage before validation." }];
pub const IMWRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(integer_A_or_X, ...)", inputs: &IMWRITE_DOCUMENTED_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The documented host/file sink retains native uint8 or uint16 samples and indexed values through encoding. Resident image input is gathered non-destructively through its exact owner, and all validation/encoding completes before the write begins. Encoder limitations such as 16-bit JPEG and TIFF CMYK reject explicitly without changing the class contract or creating a file." },
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(unsupported_integer_A_or_X, ...)", inputs: &IMWRITE_REJECTED_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Unsupported image classes reject from authoritative host or resident metadata without a floating compatibility conversion or file effect." },
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(A, ..., Alpha=integer_alpha)", inputs: &IMWRITE_ALPHA_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Documented uint8 and uint16 Alpha arrays preserve exact native samples until deterministic class-range scaling into the image bit depth; shape and format validation complete before file effects." },
    BuiltinIntegerCapabilityDescriptor { form: "imwrite(A, ..., integer_control)", inputs: &IMWRITE_CONTROL_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Typed scalar controls are read exactly and range-checked before encoding. Documented but not-yet-supported format controls, including RowsPerStrip and explicit BitDepth, reject with the stable invalid-option error before any file effect rather than silently coercing or ignoring the value." },
];
