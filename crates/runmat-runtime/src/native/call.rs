use super::{NativeExit, NativeFrame, NativeHostVTable, NativeValueRef};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCallKind(pub u32);

impl NativeCallKind {
    pub const DIRECT: Self = Self(0);
    pub const DYNAMIC: Self = Self(1);
    pub const BUILTIN: Self = Self(2);
    pub const METHOD: Self = Self(3);
    pub const CALLBACK: Self = Self(4);
    pub const RESUME: Self = Self(5);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::RESUME.0
    }
}

/// One borrowed call transaction. Results are committed only when the returned
/// `NativeExit` is `Completed`; all other exits leave host-visible outputs unset.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeCall {
    pub abi_version: u32,
    pub kind: NativeCallKind,
    pub flags: u32,
    pub requested_outputs: u32,
    pub host: *const NativeHostVTable,
    pub frame: *mut NativeFrame,
    pub arguments: *const NativeValueRef,
    pub argument_count: usize,
    pub results: *mut NativeValueRef,
    pub result_capacity: usize,
}

impl Default for NativeCall {
    fn default() -> Self {
        Self {
            abi_version: super::NATIVE_ABI_VERSION.encoded(),
            kind: NativeCallKind::DIRECT,
            flags: 0,
            requested_outputs: 0,
            host: std::ptr::null(),
            frame: std::ptr::null_mut(),
            arguments: std::ptr::null(),
            argument_count: 0,
            results: std::ptr::null_mut(),
            result_capacity: 0,
        }
    }
}

impl NativeCall {
    /// Validate the transactional result produced for this call.
    ///
    /// Native code may write result slots speculatively, but the host commits
    /// them only for a completed exit that fits the requested result window.
    pub fn validate_exit(&self, exit: &NativeExit) -> Result<(), super::NativeAbiError> {
        self.validate()?;
        exit.validate()?;
        if exit.kind == super::NativeExitKind::COMPLETED {
            if exit.produced_outputs > self.requested_outputs
                || exit.produced_outputs as usize > self.result_capacity
            {
                return Err(super::NativeAbiError::new(
                    "native.exit.produced_outputs",
                    "completed outputs exceed the call result window",
                ));
            }
        } else if exit.produced_outputs != 0 {
            return Err(super::NativeAbiError::new(
                "native.exit.produced_outputs",
                "non-completed exits cannot commit outputs",
            ));
        }
        Ok(())
    }
}
