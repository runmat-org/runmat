pub const RUNMAT_EXTENSION_HEADER: &str = include_str!("../include/runmat_extension.h");

pub fn generated_header() -> String {
    RUNMAT_EXTENSION_HEADER.to_owned()
}
