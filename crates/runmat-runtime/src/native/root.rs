use super::NativeValueRef;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRootKind(pub u32);

impl NativeRootKind {
    pub const LOCAL: Self = Self(0);
    pub const OPERAND: Self = Self(1);
    pub const TEMPORARY: Self = Self(2);
    pub const CLOSURE: Self = Self(3);
    pub const CONTINUATION: Self = Self(4);
    pub const HOST: Self = Self(5);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::HOST.0
    }
}

/// One precise GC root visible at a native safepoint.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRoot {
    pub value: NativeValueRef,
    pub kind: NativeRootKind,
    pub slot: u32,
}

/// Borrowed root slice. The host owns the backing allocation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeRootSet {
    pub roots: *const NativeRoot,
    pub count: usize,
}

impl Default for NativeRootSet {
    fn default() -> Self {
        Self {
            roots: std::ptr::null(),
            count: 0,
        }
    }
}
