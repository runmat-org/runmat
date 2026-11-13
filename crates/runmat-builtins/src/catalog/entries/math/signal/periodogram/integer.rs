use crate::*;

const PERIODOGRAM_INTEGER_SIGNAL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double signals. RunMat gates typed integer signals before provider dispatch and requires exact binary64 conversion.",
    }];
const PERIODOGRAM_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "window",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented window storage is single or double; typed windows cross a checked floating boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "freqSpec/nfft",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A scalar is decoded as an exact structural DFT length; a vector is a checked floating frequency grid.",
    },
    BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed sample rates enter the floating frequency-scaling boundary only after exactness is proved.",
    },
    BuiltinIntegerInputCapability {
        name: "confidence or numeric option",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Additional supported numeric controls remain independently classified as checked floating inputs.",
    },
];
pub const PERIODOGRAM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[pxx,f] = periodogram(integer_x, ...)",
        inputs: &PERIODOGRAM_INTEGER_SIGNAL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed signals are a RunMat-only extension; authoritative storage is checked before FFT conversion and supported resident floating paths remain unaffected.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "periodogram(x, integer_window_or_frequency_controls)",
        inputs: &PERIODOGRAM_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed numerical controls are independently gated; scalar DFT length is structural and remaining controls cross a checked binary64 boundary. Computed output class follows the signal: Single stays Single; Double and admitted integer/logical signals produce Double. Controls do not determine result class.",
    },
];
