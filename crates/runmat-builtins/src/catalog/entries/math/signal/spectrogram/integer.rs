use crate::*;

const SPECTROGRAM_INTEGER_SIGNAL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented signal classes are single and double; admitted typed values must be exact at the binary64 STFT boundary.",
    }];
const SPECTROGRAM_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    spectrogram_integer_control("win"),
    spectrogram_integer_control("nOverlap"),
    spectrogram_integer_control("freqSpec"),
    spectrogram_integer_control("Fs"),
];
const fn spectrogram_integer_control(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public documentation establishes single and double storage for this role; RunMat gates typed input and prevents rounded conversion.",
    }
}
pub const SPECTROGRAM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[s,f,t,ps] = spectrogram(integer_x, ...)",
        inputs: &SPECTROGRAM_INTEGER_SIGNAL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The signal crosses an exact binary64 boundary before host STFT or provider spectral dispatch; GPU support remains provider-dependent.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "spectrogram(x, integer_win/nOverlap/freqSpec/Fs, ...)",
        inputs: &SPECTROGRAM_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed numeric controls are gated RunMat extensions. Structural scalars are range checked, and numerical vectors enter binary64 only after exact representability is proven. Computed output class follows the signal: Single stays Single; Double and admitted integer/logical signals produce Double. Controls do not determine result class.",
    },
];
