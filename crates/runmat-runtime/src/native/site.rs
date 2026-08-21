use super::{NativeCall, NativeExit, NativeHostStatus};
use std::ffi::c_void;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSitePhase(pub u32);

impl NativeSitePhase {
    pub const RVALUE: Self = Self(0);
    pub const STATEMENT: Self = Self(1);
    pub const TERMINATOR_RVALUE: Self = Self(2);
    pub const TERMINATOR: Self = Self(3);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::TERMINATOR.0
    }
}

/// Stable, path-independent identity of one executable Native IR site.
///
/// The host maps this identity back to the immutable Native IR product. The
/// ABI deliberately does not embed MIR, Rust enums, pointers, or serialized
/// operation payloads in generated code.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSiteRequest {
    pub function: u32,
    pub block: u32,
    pub position: u32,
    pub phase: NativeSitePhase,
    pub ordinal: u32,
    pub reserved: u32,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSiteOutcomeKind(pub u32);

impl NativeSiteOutcomeKind {
    /// Continue with the next instruction in the current compiled block.
    pub const CONTINUE: Self = Self(0);
    /// Transfer to the indexed outgoing edge of the current terminator.
    pub const EDGE: Self = Self(1);
    /// Return from generated code with the initialized `NativeExit`.
    pub const EXIT: Self = Self(2);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::EXIT.0
    }
}

/// Host decision for one Native IR site.
///
/// `edge` is meaningful only for `EDGE`; zero is required for `CONTINUE` and
/// `EXIT`. The generated function treats `NativeHostStatus` as an ABI/service
/// result and this value as semantic control flow.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSiteOutcome {
    pub kind: NativeSiteOutcomeKind,
    pub edge: u32,
    pub flags: u32,
    pub reserved: u32,
}

impl NativeSiteOutcome {
    pub const fn continue_execution() -> Self {
        Self::new(NativeSiteOutcomeKind::CONTINUE, 0)
    }

    pub const fn edge(edge: u32) -> Self {
        Self::new(NativeSiteOutcomeKind::EDGE, edge)
    }

    pub const fn exit() -> Self {
        Self::new(NativeSiteOutcomeKind::EXIT, 0)
    }

    const fn new(kind: NativeSiteOutcomeKind, edge: u32) -> Self {
        Self {
            kind,
            edge,
            flags: 0,
            reserved: 0,
        }
    }
}

/// Execute exactly one immutable Native IR site.
///
/// The callback must initialize `outcome` when it returns
/// `NativeHostStatus::OK`, and must initialize `exit` when that outcome is
/// `EXIT`. Panics and platform unwinding may not cross this boundary. `call`
/// retains the frame, opaque value arena, result window, and host table for the
/// complete generated invocation.
pub type NativeExecuteSiteFn = unsafe extern "C" fn(
    context: *mut c_void,
    call: *mut NativeCall,
    request: *const NativeSiteRequest,
    outcome: *mut NativeSiteOutcome,
    exit: *mut NativeExit,
) -> NativeHostStatus;

impl NativeSiteRequest {
    pub fn validate(&self) -> Result<(), super::NativeAbiError> {
        if !self.phase.is_known() {
            return Err(super::NativeAbiError::new(
                "native.site.phase",
                "site phase is unknown",
            ));
        }
        if self.reserved != 0 {
            return Err(super::NativeAbiError::new(
                "native.site.reserved",
                "reserved bits must be zero for this ABI revision",
            ));
        }
        Ok(())
    }
}

impl NativeSiteOutcome {
    pub fn validate(&self) -> Result<(), super::NativeAbiError> {
        if !self.kind.is_known() {
            return Err(super::NativeAbiError::new(
                "native.site_outcome.kind",
                "site outcome kind is unknown",
            ));
        }
        if self.flags != 0 || self.reserved != 0 {
            return Err(super::NativeAbiError::new(
                "native.site_outcome.reserved",
                "flags and reserved bits must be zero for this ABI revision",
            ));
        }
        if self.kind != NativeSiteOutcomeKind::EDGE && self.edge != 0 {
            return Err(super::NativeAbiError::new(
                "native.site_outcome.edge",
                "only an edge outcome may carry an edge index",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn site_identity_and_outcomes_reject_reserved_or_ambiguous_states() {
        let request = NativeSiteRequest {
            function: 7,
            block: 3,
            position: 2,
            phase: NativeSitePhase::STATEMENT,
            ordinal: 1,
            reserved: 0,
        };
        assert_eq!(request.validate(), Ok(()));
        assert_eq!(NativeSiteOutcome::continue_execution().validate(), Ok(()));
        assert_eq!(NativeSiteOutcome::edge(4).validate(), Ok(()));
        assert_eq!(NativeSiteOutcome::exit().validate(), Ok(()));

        let invalid_request = NativeSiteRequest {
            phase: NativeSitePhase(u32::MAX),
            ..request
        };
        assert_eq!(
            invalid_request.validate().unwrap_err().field,
            "native.site.phase"
        );

        let invalid_outcome = NativeSiteOutcome {
            kind: NativeSiteOutcomeKind::CONTINUE,
            edge: 1,
            flags: 0,
            reserved: 0,
        };
        assert_eq!(
            invalid_outcome.validate().unwrap_err().field,
            "native.site_outcome.edge"
        );
    }
}
