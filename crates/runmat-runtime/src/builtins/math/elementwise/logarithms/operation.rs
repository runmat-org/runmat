use runmat_accelerate_api::{AccelProvider, AccelProviderFuture, GpuTensorHandle};
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, LOG10_CHARACTER_INPUT_EXTENSION,
    LOG10_ERROR_INTERNAL, LOG10_ERROR_INVALID_INPUT, LOG10_EXPLICIT_GPU_COMPLEX_EXTENSION,
    LOG10_INTEGER_INPUT_EXTENSION, LOG10_LOGICAL_INPUT_EXTENSION, LOG1P_CHARACTER_INPUT_EXTENSION,
    LOG1P_ERROR_INTERNAL, LOG1P_ERROR_INVALID_INPUT, LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION,
    LOG1P_INTEGER_INPUT_EXTENSION, LOG1P_LOGICAL_INPUT_EXTENSION, LOG2_CHARACTER_EXTENSION,
    LOG2_ERROR_INTERNAL, LOG2_ERROR_INVALID_INPUT, LOG2_INTEGER_EXTENSION, LOG2_LOGICAL_EXTENSION,
    LOG_CHARACTER_INPUT_EXTENSION, LOG_ERROR_INTERNAL, LOG_ERROR_INVALID_INPUT,
    LOG_EXPLICIT_GPU_COMPLEX_EXTENSION, LOG_INTEGER_INPUT_EXTENSION, LOG_LOGICAL_INPUT_EXTENSION,
};
use runmat_value::SymbolicFunction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LogarithmOperation {
    Natural,
    OnePlus,
    Binary,
    Common,
}

impl LogarithmOperation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Natural => "log",
            Self::OnePlus => "log1p",
            Self::Binary => "log2",
            Self::Common => "log10",
        }
    }
    pub(super) const fn real_boundary(self) -> f64 {
        match self {
            Self::OnePlus => -1.0,
            Self::Natural | Self::Binary | Self::Common => 0.0,
        }
    }
    pub(super) fn evaluate_provider<'a>(
        self,
        provider: &'a dyn AccelProvider,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        match self {
            Self::Natural => provider.unary_log(input),
            Self::OnePlus => provider.unary_log1p(input),
            Self::Binary => provider.unary_log2(input),
            Self::Common => provider.unary_log10(input),
        }
    }
    pub(super) const fn accepts_tabular(self) -> bool {
        !matches!(self, Self::OnePlus)
    }
    pub(super) const fn normalizes_near_zero_real(self) -> bool {
        !matches!(self, Self::OnePlus)
    }
    pub(super) const fn restores_automatic_fallback(self) -> bool {
        !matches!(self, Self::Binary)
    }
    pub(super) const fn invalid_input(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Natural => &LOG_ERROR_INVALID_INPUT,
            Self::OnePlus => &LOG1P_ERROR_INVALID_INPUT,
            Self::Binary => &LOG2_ERROR_INVALID_INPUT,
            Self::Common => &LOG10_ERROR_INVALID_INPUT,
        }
    }
    pub(super) const fn internal_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Natural => &LOG_ERROR_INTERNAL,
            Self::OnePlus => &LOG1P_ERROR_INTERNAL,
            Self::Binary => &LOG2_ERROR_INTERNAL,
            Self::Common => &LOG10_ERROR_INTERNAL,
        }
    }
    pub(super) const fn integer_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Natural => &LOG_INTEGER_INPUT_EXTENSION,
            Self::OnePlus => &LOG1P_INTEGER_INPUT_EXTENSION,
            Self::Binary => &LOG2_INTEGER_EXTENSION,
            Self::Common => &LOG10_INTEGER_INPUT_EXTENSION,
        }
    }
    pub(super) const fn logical_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Natural => &LOG_LOGICAL_INPUT_EXTENSION,
            Self::OnePlus => &LOG1P_LOGICAL_INPUT_EXTENSION,
            Self::Binary => &LOG2_LOGICAL_EXTENSION,
            Self::Common => &LOG10_LOGICAL_INPUT_EXTENSION,
        }
    }
    pub(super) const fn character_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Natural => &LOG_CHARACTER_INPUT_EXTENSION,
            Self::OnePlus => &LOG1P_CHARACTER_INPUT_EXTENSION,
            Self::Binary => &LOG2_CHARACTER_EXTENSION,
            Self::Common => &LOG10_CHARACTER_INPUT_EXTENSION,
        }
    }
    pub(super) const fn explicit_complex_extension(
        self,
    ) -> Option<&'static BuiltinExtensionDescriptor> {
        match self {
            Self::Natural => Some(&LOG_EXPLICIT_GPU_COMPLEX_EXTENSION),
            Self::OnePlus => Some(&LOG1P_EXPLICIT_GPU_COMPLEX_EXTENSION),
            Self::Binary => None,
            Self::Common => Some(&LOG10_EXPLICIT_GPU_COMPLEX_EXTENSION),
        }
    }
    pub(super) const fn symbolic_function(self) -> Option<SymbolicFunction> {
        match self {
            Self::Natural => Some(SymbolicFunction::Log),
            Self::OnePlus | Self::Binary | Self::Common => None,
        }
    }
    pub(super) fn complex_f64(self, real: f64, imag: f64) -> (f64, f64) {
        super::complex::evaluate_f64(self, real, imag)
    }
    pub(super) fn complex_f32(self, real: f32, imag: f32) -> (f32, f32) {
        super::complex::evaluate_f32(self, real, imag)
    }
}
