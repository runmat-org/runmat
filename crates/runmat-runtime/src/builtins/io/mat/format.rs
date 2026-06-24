//! Shared MAT-file format helpers.

/// MAT-file data types used in Level 5 files.
pub const MI_INT8: u32 = 1;
pub const MI_UINT8: u32 = 2;
pub const MI_INT16: u32 = 3;
pub const MI_UINT16: u32 = 4;
pub const MI_INT32: u32 = 5;
pub const MI_UINT32: u32 = 6;
pub const MI_SINGLE: u32 = 7;
pub const MI_DOUBLE: u32 = 9;
pub const MI_INT64: u32 = 12;
pub const MI_UINT64: u32 = 13;
pub const MI_MATRIX: u32 = 14;
pub const MI_COMPRESSED: u32 = 15;
pub const MI_UTF8: u32 = 16;
pub const MI_UTF16: u32 = 17;
pub const MI_UTF32: u32 = 18;

/// Number of bytes in the MAT-file header.
pub const MAT_HEADER_LEN: usize = 128;

/// Logical flag bit in the array flags word.
pub const FLAG_LOGICAL: u32 = 1 << 9;
/// Complex flag bit in the array flags word.
pub const FLAG_COMPLEX: u32 = 1 << 10;

/// Representation of a MAT-file array.
#[derive(Clone, Debug)]
pub struct MatArray {
    pub class: MatClass,
    pub dims: Vec<usize>,
    pub data: MatData,
}

/// MAT-file array class (mirrors MATLAB mxClassID values we emit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatClass {
    Double,
    Single,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Logical,
    Char,
    Cell,
    Struct,
    Sparse,
}

impl MatClass {
    /// Convert the class to the numeric code used in MAT-file array flags.
    pub fn class_code(&self) -> u32 {
        match self {
            MatClass::Double => 6,
            MatClass::Single => 7,
            MatClass::Int8 => 8,
            MatClass::UInt8 => 9,
            MatClass::Int16 => 10,
            MatClass::UInt16 => 11,
            MatClass::Int32 => 12,
            MatClass::UInt32 => 13,
            MatClass::Int64 => 14,
            MatClass::UInt64 => 15,
            MatClass::Logical => 9,
            MatClass::Char => 4,
            MatClass::Cell => 1,
            MatClass::Struct => 2,
            MatClass::Sparse => 5,
        }
    }

    /// Convert from a numeric class code.
    pub fn from_class_code(code: u32) -> Option<Self> {
        match code {
            6 => Some(MatClass::Double),
            7 => Some(MatClass::Single),
            8 => Some(MatClass::Int8),
            9 => Some(MatClass::UInt8),
            10 => Some(MatClass::Int16),
            11 => Some(MatClass::UInt16),
            12 => Some(MatClass::Int32),
            13 => Some(MatClass::UInt32),
            14 => Some(MatClass::Int64),
            15 => Some(MatClass::UInt64),
            4 => Some(MatClass::Char),
            1 => Some(MatClass::Cell),
            2 => Some(MatClass::Struct),
            5 => Some(MatClass::Sparse),
            _ => None,
        }
    }
}

/// MAT-file data payload variants we currently support.
#[derive(Clone, Debug)]
pub enum MatData {
    Double {
        real: Vec<f64>,
        imag: Option<Vec<f64>>,
    },
    Numeric {
        real: Vec<f64>,
        imag: Option<Vec<f64>>,
    },
    Logical {
        data: Vec<u8>,
    },
    Char {
        data: Vec<u16>,
    },
    Cell {
        elements: Vec<MatArray>,
    },
    Struct {
        field_names: Vec<String>,
        field_values: Vec<MatArray>,
    },
    Sparse {
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    },
}
