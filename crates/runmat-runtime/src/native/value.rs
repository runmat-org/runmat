/// Opaque, generation-checked reference to a runtime-owned RunMat value.
///
/// Native code may copy this token but must use runtime helpers to inspect,
/// materialize, retain, or release the underlying value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeValueRef {
    pub handle: u64,
    pub generation: u64,
}

impl NativeValueRef {
    pub const NULL: Self = Self {
        handle: 0,
        generation: 0,
    };

    pub const fn is_null(self) -> bool {
        self.handle == 0
    }

    pub const fn is_valid(self) -> bool {
        (self.handle == 0) == (self.generation == 0)
    }
}
