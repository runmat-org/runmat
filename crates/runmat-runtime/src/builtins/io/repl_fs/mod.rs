//! REPL-facing filesystem builtins.

pub mod addpath;
pub mod cd;
pub mod compat;
pub mod copyfile;
pub mod delete;
pub mod dir;
pub mod exist;
pub(crate) mod file_dialog;
pub mod fullfile;
pub mod genpath;
pub mod getenv;
pub mod ls;
pub mod mkdir;
pub mod movefile;
pub mod open;
pub mod opentoline;
pub mod path;
pub mod pcode;
pub mod pwd;
pub mod rmdir;
pub mod rmpath;
pub mod run;
pub mod savepath;
pub mod setenv;
pub mod tempdir;
pub mod tempname;
pub mod uigetdir;
pub mod uigetfile;
pub mod uiputfile;
pub mod xml;
use once_cell::sync::Lazy;
use runmat_builtins::Tensor;

use crate::builtins::common::tensor;
use std::path::Path;
use std::sync::Mutex;

pub static REPL_FS_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

pub(crate) fn is_rooted_path(path: &Path) -> bool {
    path.is_absolute() || path.has_root()
}

pub(crate) fn tensor_char_codes_to_string(value: &Tensor) -> Option<String> {
    let codes = tensor::tensor_values_f64(value);
    let mut text = String::with_capacity(codes.len());
    for code in codes {
        if !code.is_finite() {
            return None;
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return None;
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return None;
        }
        text.push(char::from_u32(int_code as u32)?);
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn tensor_char_codes_to_string_reads_typed_integer_storage_exactly() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![82, 77]), vec![1, 2]).expect("tensor");
        tensor.data = vec![65.0, 66.0];
        assert_eq!(tensor_char_codes_to_string(&tensor).as_deref(), Some("RM"));

        let mut negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("tensor");
        negative.data = vec![65.0];
        assert!(tensor_char_codes_to_string(&negative).is_none());

        let mut invalid =
            Tensor::new_integer(IntegerStorage::U32(vec![0x11_0000]), vec![1, 1]).expect("tensor");
        invalid.data = vec![65.0];
        assert!(tensor_char_codes_to_string(&invalid).is_none());
    }
}
