use super::{
    NativeCall, NativeCancellationReason, NativeDeoptReason, NativeExit, NativeExitKind,
    NativeFrame, NativeResumeKind, NativeRootSet, NativeSafepoint, NativeSourceMapView,
    NATIVE_ABI_VERSION,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAbiError {
    pub field: &'static str,
    pub message: &'static str,
}

impl NativeAbiError {
    pub(super) const fn new(field: &'static str, message: &'static str) -> Self {
        Self { field, message }
    }
}

impl std::fmt::Display for NativeAbiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.message)
    }
}

impl std::error::Error for NativeAbiError {}

impl NativeRootSet {
    pub fn validate(self) -> Result<(), NativeAbiError> {
        validate_slice("native.roots", self.roots, self.count)
    }
}

impl NativeSourceMapView {
    pub fn validate(self) -> Result<(), NativeAbiError> {
        validate_slice("native.source_map", self.entries, self.count)
    }
}

impl NativeFrame {
    pub fn validate(&self) -> Result<(), NativeAbiError> {
        validate_version(self.abi_version)?;
        if !self.kind.is_known() {
            return Err(NativeAbiError::new(
                "native.frame.kind",
                "frame kind is unknown",
            ));
        }
        validate_zero("native.frame.flags", self.flags)?;
        validate_zero("native.frame.reserved", self.reserved)?;
        validate_slice("native.frame.locals", self.locals, self.local_count)?;
        validate_slice(
            "native.frame.operands",
            self.operands,
            self.operand_capacity,
        )?;
        self.roots.validate()?;
        if self.resume.is_null() {
            return Err(NativeAbiError::new(
                "native.frame.resume",
                "exact resume state is required",
            ));
        }
        Ok(())
    }
}

impl NativeCall {
    pub fn validate(&self) -> Result<(), NativeAbiError> {
        validate_version(self.abi_version)?;
        if !self.kind.is_known() {
            return Err(NativeAbiError::new(
                "native.call.kind",
                "call kind is unknown",
            ));
        }
        validate_zero("native.call.flags", self.flags)?;
        if self.host.is_null() {
            return Err(NativeAbiError::new(
                "native.call.host",
                "host service table must not be null",
            ));
        }
        if self.frame.is_null() {
            return Err(NativeAbiError::new(
                "native.call.frame",
                "frame must not be null",
            ));
        }
        validate_slice("native.call.arguments", self.arguments, self.argument_count)?;
        validate_slice("native.call.results", self.results, self.result_capacity)?;
        if self.requested_outputs as usize > self.result_capacity {
            return Err(NativeAbiError::new(
                "native.call.requested_outputs",
                "requested outputs exceed result capacity",
            ));
        }
        Ok(())
    }
}

impl NativeExit {
    pub fn validate(&self) -> Result<(), NativeAbiError> {
        validate_version(self.abi_version)?;
        validate_zero("native.exit.flags", self.flags)?;
        match self.kind {
            NativeExitKind::COMPLETED => Ok(()),
            NativeExitKind::EXCEPTION
                if self.exception.handle != 0 && self.exception.generation != 0 =>
            {
                self.exception.source.validate()
            }
            NativeExitKind::CANCELLED
                if self.cancellation.reason.is_known()
                    && self.cancellation.reserved == 0
                    && self.cancellation.generation != 0 =>
            {
                Ok(())
            }
            NativeExitKind::SUSPENDED => {
                if self.suspension.continuation == 0
                    || self.suspension.generation == 0
                    || self.suspension.resume.is_null()
                {
                    return Err(NativeAbiError::new(
                        "native.exit.suspension",
                        "continuation and exact resume state are required",
                    ));
                }
                self.suspension.roots.validate()
            }
            NativeExitKind::DEOPTIMIZED
                if self.deoptimization.reason.is_known()
                    && self.deoptimization.target.is_known()
                    && !self.deoptimization.resume.is_null() =>
            {
                Ok(())
            }
            NativeExitKind::EXCEPTION => Err(NativeAbiError::new(
                "native.exit.exception",
                "exception handle, generation, and source must be valid",
            )),
            NativeExitKind::CANCELLED => Err(NativeAbiError::new(
                "native.exit.cancellation",
                "cancellation reason, generation, and reserved fields must be valid",
            )),
            NativeExitKind::DEOPTIMIZED => Err(NativeAbiError::new(
                "native.exit.deoptimization",
                "reason, target, and resume state must be valid",
            )),
            _ => Err(NativeAbiError::new(
                "native.exit.kind",
                "exit kind is unknown",
            )),
        }
    }
}

impl NativeSafepoint {
    pub fn validate(&self) -> Result<(), NativeAbiError> {
        if self.resume.is_null() {
            return Err(NativeAbiError::new(
                "native.safepoint.resume",
                "exact resume state is required",
            ));
        }
        self.roots.validate()
    }
}

impl super::NativeResumeState {
    pub fn validate(&self) -> Result<(), NativeAbiError> {
        validate_zero("native.resume.flags", self.flags)?;
        self.source.validate()
    }
}

impl NativeCancellationReason {
    pub const fn is_known(self) -> bool {
        self.0 <= Self::SHUTDOWN.0
    }
}

impl NativeResumeKind {
    pub const fn is_known(self) -> bool {
        self.0 <= Self::OPTIMIZED_NATIVE.0
    }
}

impl NativeDeoptReason {
    pub const fn is_known(self) -> bool {
        self.0 <= Self::EXPLICIT_SLOW_PATH.0
    }
}

fn validate_version(version: u32) -> Result<(), NativeAbiError> {
    if version != NATIVE_ABI_VERSION.encoded() {
        return Err(NativeAbiError::new(
            "native.abi_version",
            "unsupported native ABI version",
        ));
    }
    Ok(())
}

fn validate_zero(field: &'static str, value: u32) -> Result<(), NativeAbiError> {
    if value != 0 {
        return Err(NativeAbiError::new(
            field,
            "reserved bits must be zero for this ABI revision",
        ));
    }
    Ok(())
}

pub(super) fn validate_slice<T>(
    field: &'static str,
    pointer: *const T,
    count: usize,
) -> Result<(), NativeAbiError> {
    if pointer.is_null() != (count == 0) {
        return Err(NativeAbiError::new(
            field,
            "pointer must be null exactly when count is zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{
        NativeCallKind, NativeCancellation, NativeException, NativeFrameKind, NativeResumeState,
        NativeRoot, NativeSourceLocation, NativeValueRef,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn call_and_frame_require_exact_borrowed_state() {
        let mut resume = NativeResumeState::default();
        let mut local = NativeValueRef::NULL;
        let root = NativeRoot {
            value: NativeValueRef::NULL,
            kind: crate::native::NativeRootKind::LOCAL,
            slot: 0,
        };
        let mut frame = NativeFrame {
            kind: NativeFrameKind::FUNCTION,
            locals: &mut local,
            local_count: 1,
            operands: std::ptr::null_mut(),
            operand_capacity: 0,
            roots: NativeRootSet {
                roots: &root,
                count: 1,
            },
            resume: &mut resume,
            ..NativeFrame::default()
        };
        assert_eq!(frame.validate(), Ok(()));

        let host = std::ptr::dangling::<crate::native::NativeHostVTable>();
        let call = NativeCall {
            kind: NativeCallKind::DIRECT,
            host,
            frame: &mut frame,
            ..NativeCall::default()
        };
        assert_eq!(call.validate(), Ok(()));

        let invalid = NativeCall {
            requested_outputs: 1,
            ..call
        };
        assert_eq!(
            invalid.validate().unwrap_err().field,
            "native.call.requested_outputs"
        );

        let invalid_kind = NativeCall {
            kind: NativeCallKind(u32::MAX),
            ..call
        };
        assert_eq!(
            invalid_kind.validate().unwrap_err().field,
            "native.call.kind"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn unknown_control_codes_fail_closed_without_invalid_rust_enums() {
        assert!(!NativeCancellationReason(u32::MAX).is_known());
        assert!(!NativeResumeKind(u32::MAX).is_known());
        assert!(!NativeDeoptReason(u32::MAX).is_known());
        assert!(!NativeValueRef {
            handle: 1,
            generation: 0
        }
        .is_valid());

        let invalid_resume = NativeResumeState {
            flags: 1,
            ..NativeResumeState::default()
        };
        assert_eq!(
            invalid_resume.validate().unwrap_err().field,
            "native.resume.flags"
        );

        let invalid_source = NativeSourceLocation {
            start: 2,
            end: 1,
            ..NativeSourceLocation::default()
        };
        assert_eq!(
            invalid_source.validate().unwrap_err().field,
            "native.source_location"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn transactional_exit_variants_validate_their_selected_payload() {
        assert_eq!(NativeExit::completed(2).validate(), Ok(()));
        assert_eq!(
            NativeExit::exception(NativeException {
                handle: 7,
                generation: 1,
                source: NativeSourceLocation::default(),
            })
            .validate(),
            Ok(())
        );
        assert_eq!(
            NativeExit::cancelled(NativeCancellation {
                reason: NativeCancellationReason::DEADLINE,
                reserved: 0,
                generation: 2,
            })
            .validate(),
            Ok(())
        );
        assert!(NativeExit::exception(NativeException::default())
            .validate()
            .is_err());

        let mut resume = NativeResumeState::default();
        let mut frame = NativeFrame {
            resume: &mut resume,
            ..NativeFrame::default()
        };
        let mut result = NativeValueRef::NULL;
        let call = NativeCall {
            requested_outputs: 1,
            host: std::ptr::dangling::<crate::native::NativeHostVTable>(),
            frame: &mut frame,
            results: &mut result,
            result_capacity: 1,
            ..NativeCall::default()
        };
        assert_eq!(call.validate_exit(&NativeExit::completed(1)), Ok(()));
        assert_eq!(
            call.validate_exit(&NativeExit::completed(2))
                .unwrap_err()
                .field,
            "native.exit.produced_outputs"
        );

        let mut non_transactional = NativeExit::cancelled(NativeCancellation {
            reason: NativeCancellationReason::REQUESTED,
            reserved: 0,
            generation: 1,
        });
        non_transactional.produced_outputs = 1;
        assert_eq!(
            call.validate_exit(&non_transactional).unwrap_err().field,
            "native.exit.produced_outputs"
        );
    }
}
