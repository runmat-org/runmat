use crate::*;

const SIGNAL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "x",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Signals cross a checked exact binary64 Welch boundary before floating FFT evaluation.",
}];
const CONTROLS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "window",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar length is structural; vector windows cross a checked floating boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "noverlap",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Overlap is an exact structural nonnegative integer.",
    },
    BuiltinIntegerInputCapability {
        name: "freqSpec/nfft",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar DFT length is structural; vector grid crosses a checked floating boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "Fs",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Sample rate crosses a checked exact binary64 boundary.",
    },
];
pub const PWELCH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "[pxx,f] = pwelch(integer_x, ...)", inputs: &SIGNAL, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only signal extension. Resident integers use an exact owner-aware host fallback; computed outputs are Double." },
    BuiltinIntegerCapabilityDescriptor { form: "pwelch(x, integer_window_or_numeric_controls)", inputs: &CONTROLS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Numeric controls do not determine output class: Single signals publish Single results; Double and admitted integer/logical signals publish Double results." },
];
