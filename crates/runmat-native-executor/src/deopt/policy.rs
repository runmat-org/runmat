use runmat_native_codegen::NativeSafepointId;
use runmat_types::RegionGuardId;

use crate::specialization::GuardEnvironment;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResumeTarget {
    Interpreter,
    #[default]
    GenericNative,
}

impl ResumeTarget {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn native(self) -> runmat_runtime::native::NativeResumeKind {
        match self {
            Self::Interpreter => runmat_runtime::native::NativeResumeKind::INTERPRETER,
            Self::GenericNative => runmat_runtime::native::NativeResumeKind::GENERIC_NATIVE,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaultInjection {
    Guard(RegionGuardId),
    Safepoint(NativeSafepointId),
}

#[derive(Clone, Debug, Default)]
pub struct DeoptimizationPolicy {
    pub target: ResumeTarget,
    pub guards: GuardEnvironment,
    pub inject: Option<FaultInjection>,
}

impl DeoptimizationPolicy {
    pub fn inject(mut self, injection: FaultInjection) -> Self {
        self.inject = Some(injection);
        self
    }
}
