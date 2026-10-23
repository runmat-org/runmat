use crate::*;

pub(super) const GRAYSCALE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "I",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Grayscale image data accepts every integer class and derives the default display range from its exact class.",
    },
    BuiltinIntegerInputCapability {
        name: "DisplayRange",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Explicit two-element display limits accept all integer classes and are parsed from exact storage.",
    },
];

pub(super) const TRUECOLOR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &UNSIGNED_8_16_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer truecolor input accepts uint8 and uint16; other integer classes reject.",
    }];

pub(super) const FOUR_CHANNEL_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "I",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric M-by-N-by-4 image data is gated by imshow-four-channel-image.",
    }];

pub const IMSHOW_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = imshow(I, DisplayRange) [grayscale]",
        inputs: &GRAYSCALE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact integer class selects normalization limits before values enter the rendering domain; grayscale GPU inputs may remain resident with shared WGPU context.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = imshow(I) [truecolor]",
        inputs: &TRUECOLOR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Three-channel integer truecolor rendering is documented only for uint8 and uint16.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = imshow(I) [four-channel image]",
        inputs: &FOUR_CHANNEL_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Four-channel numeric rendering is a separately gated RunMat-only form.",
    },
];
