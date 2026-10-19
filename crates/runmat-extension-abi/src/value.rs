#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunMatValueKind {
    Unknown = 0,
    Scalar = 1,
    Dense = 2,
    Sparse = 3,
    Logical = 4,
    Character = 5,
    String = 6,
    Cell = 7,
    Structure = 8,
    Object = 9,
    Callable = 10,
    Foreign = 11,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunMatElementType {
    Unknown = 0,
    F64 = 1,
    F32 = 2,
    I8 = 3,
    I16 = 4,
    I32 = 5,
    I64 = 6,
    U8 = 7,
    U16 = 8,
    U32 = 9,
    U64 = 10,
    Logical = 11,
    CharacterU32 = 12,
    ComplexF64 = 13,
    ComplexF32 = 14,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunMatBufferView {
    pub data: *const u8,
    pub byte_length: usize,
    pub shape: *const usize,
    pub rank: usize,
    pub element_type: RunMatElementType,
    pub flags: u32,
}

/// A buffer view and the explicit lifetime token that keeps it valid.
///
/// The view, including its shape pointer, remains valid until the matching
/// host `release_buffer` call succeeds. Releasing the source value handle does
/// not implicitly release this lease.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RunMatBufferLease {
    pub view: RunMatBufferView,
    pub handle: crate::RunMatBufferLeaseHandle,
}

pub const RUNMAT_BUFFER_READ_ONLY: u32 = 1 << 0;
pub const RUNMAT_BUFFER_COLUMN_MAJOR: u32 = 1 << 1;
pub const RUNMAT_BUFFER_INTERLEAVED_COMPLEX: u32 = 1 << 2;
