//! Shared MAT-file format helpers.

/// MAT-file data types used in Level 5 files.
pub const MI_INT8: u32 = 1;
pub const MI_UINT8: u32 = 2;
pub const MI_UINT16: u32 = 4;
pub const MI_INT32: u32 = 5;
pub const MI_UINT32: u32 = 6;
pub const MI_DOUBLE: u32 = 9;
pub const MI_MATRIX: u32 = 14;

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
    Logical,
    Char,
    Cell,
    Struct,
}

impl MatClass {
    /// Convert the class to the numeric code used in MAT-file array flags.
    pub fn class_code(&self) -> u32 {
        match self {
            MatClass::Double => 6,
            MatClass::Logical => 9,
            MatClass::Char => 4,
            MatClass::Cell => 1,
            MatClass::Struct => 2,
        }
    }

    /// Convert from a numeric class code.
    pub fn from_class_code(code: u32) -> Option<Self> {
        match code {
            6 => Some(MatClass::Double),
            9 => Some(MatClass::Logical),
            4 => Some(MatClass::Char),
            1 => Some(MatClass::Cell),
            2 => Some(MatClass::Struct),
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
}
