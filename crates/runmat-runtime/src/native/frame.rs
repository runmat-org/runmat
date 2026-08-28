use super::{NativeRootSet, NativeSourceLocation, NativeValueRef};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeFrameKind(pub u32);

impl NativeFrameKind {
    pub const FUNCTION: Self = Self(0);
    pub const SCRIPT: Self = Self(1);
    pub const CALLBACK: Self = Self(2);
    pub const CONTINUATION: Self = Self(3);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::CONTINUATION.0
    }
}

/// Exact interpreter-materializable state at a native program point.
///
/// `pc`, stack depth, locals, side-effect epoch, and source location are all
/// explicit so resume/deoptimization never replays a completed effect.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeResumeState {
    pub function: u32,
    pub block: u32,
    pub position: u32,
    pub phase: u32,
    pub ordinal: u32,
    pub flags: u32,
    pub reserved: u32,
    pub bytecode_pc: u64,
    pub operand_depth: u32,
    pub local_count: u32,
    pub side_effect_epoch: u64,
    pub source: NativeSourceLocation,
}

/// Runtime-owned frame visible to generated code and host helpers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeFrame {
    pub abi_version: u32,
    pub kind: NativeFrameKind,
    pub flags: u32,
    pub reserved: u32,
    pub caller: *mut NativeFrame,
    pub locals: *mut NativeValueRef,
    pub local_count: usize,
    pub operands: *mut NativeValueRef,
    pub operand_capacity: usize,
    pub roots: NativeRootSet,
    pub resume: *mut NativeResumeState,
}

impl Default for NativeFrame {
    fn default() -> Self {
        Self {
            abi_version: super::NATIVE_ABI_VERSION.encoded(),
            kind: NativeFrameKind::FUNCTION,
            flags: 0,
            reserved: 0,
            caller: std::ptr::null_mut(),
            locals: std::ptr::null_mut(),
            local_count: 0,
            operands: std::ptr::null_mut(),
            operand_capacity: 0,
            roots: NativeRootSet::default(),
            resume: std::ptr::null_mut(),
        }
    }
}
