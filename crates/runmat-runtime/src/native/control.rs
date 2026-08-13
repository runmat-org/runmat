use super::{NativeResumeState, NativeRootSet, NativeSourceLocation};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeExitKind(pub u32);

impl NativeExitKind {
    pub const COMPLETED: Self = Self(0);
    pub const EXCEPTION: Self = Self(1);
    pub const CANCELLED: Self = Self(2);
    pub const SUSPENDED: Self = Self(3);
    pub const DEOPTIMIZED: Self = Self(4);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::DEOPTIMIZED.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCancellationReason(pub u32);

impl NativeCancellationReason {
    pub const REQUESTED: Self = Self(0);
    pub const DEADLINE: Self = Self(1);
    pub const PARENT_SCOPE: Self = Self(2);
    pub const SHUTDOWN: Self = Self(3);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeResumeKind(pub u32);

impl NativeResumeKind {
    pub const INTERPRETER: Self = Self(0);
    pub const GENERIC_NATIVE: Self = Self(1);
    pub const OPTIMIZED_NATIVE: Self = Self(2);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeDeoptReason(pub u32);

impl NativeDeoptReason {
    pub const GUARD: Self = Self(0);
    pub const REPRESENTATION: Self = Self(1);
    pub const DEPENDENCY_INVALIDATED: Self = Self(2);
    pub const RUNTIME_STATE: Self = Self(3);
    pub const EXPLICIT_SLOW_PATH: Self = Self(4);
}

/// Opaque exception identity plus precise throw location.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeException {
    pub handle: u64,
    pub generation: u64,
    pub source: NativeSourceLocation,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeCancellation {
    pub reason: NativeCancellationReason,
    pub reserved: u32,
    pub generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeSuspension {
    pub continuation: u64,
    pub generation: u64,
    pub resume: *const NativeResumeState,
    pub roots: NativeRootSet,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeDeoptimization {
    pub reason: NativeDeoptReason,
    pub target: NativeResumeKind,
    pub guard: u64,
    pub resume: *const NativeResumeState,
}

/// Result of a native call. Only the payload selected by `kind` is meaningful.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeExit {
    pub abi_version: u32,
    pub kind: NativeExitKind,
    pub produced_outputs: u32,
    pub flags: u32,
    pub exception: NativeException,
    pub cancellation: NativeCancellation,
    pub suspension: NativeSuspension,
    pub deoptimization: NativeDeoptimization,
}

/// State exposed at every allocation, cancellation, transfer, or deopt poll.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeSafepoint {
    pub id: u64,
    pub resume: *const NativeResumeState,
    pub roots: NativeRootSet,
}

impl NativeExit {
    pub fn completed(produced_outputs: u32) -> Self {
        Self::base(NativeExitKind::COMPLETED, produced_outputs)
    }

    pub fn exception(exception: NativeException) -> Self {
        Self {
            exception,
            ..Self::base(NativeExitKind::EXCEPTION, 0)
        }
    }

    pub fn cancelled(cancellation: NativeCancellation) -> Self {
        Self {
            cancellation,
            ..Self::base(NativeExitKind::CANCELLED, 0)
        }
    }

    pub fn suspended(suspension: NativeSuspension) -> Self {
        Self {
            suspension,
            ..Self::base(NativeExitKind::SUSPENDED, 0)
        }
    }

    pub fn deoptimized(deoptimization: NativeDeoptimization) -> Self {
        Self {
            deoptimization,
            ..Self::base(NativeExitKind::DEOPTIMIZED, 0)
        }
    }

    fn base(kind: NativeExitKind, produced_outputs: u32) -> Self {
        Self {
            abi_version: super::NATIVE_ABI_VERSION.encoded(),
            kind,
            produced_outputs,
            flags: 0,
            exception: NativeException::default(),
            cancellation: NativeCancellation {
                reason: NativeCancellationReason::REQUESTED,
                reserved: 0,
                generation: 0,
            },
            suspension: NativeSuspension {
                continuation: 0,
                generation: 0,
                resume: std::ptr::null(),
                roots: NativeRootSet::default(),
            },
            deoptimization: NativeDeoptimization {
                reason: NativeDeoptReason::GUARD,
                target: NativeResumeKind::INTERPRETER,
                guard: 0,
                resume: std::ptr::null(),
            },
        }
    }
}
