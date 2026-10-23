//! Core string builtins.

pub mod char;
pub mod blanks;
pub(crate) mod classification;
pub(crate) mod comparison_observation;
pub mod compat;
pub(crate) mod conversion;
pub mod convert_chars_to_strings;
pub mod convert_contained_strings_to_chars;
pub mod convert_strings_to_chars;
pub mod compose;
pub mod genvarname;
pub mod int2str;
pub mod isletter;
pub mod is_string_scalar;
pub mod isspace;
pub mod isstrprop;
pub mod num2str;
pub(crate) mod numeric_format_work;
pub mod newline;
pub(crate) mod numeric_length;
pub mod patterns;
pub mod pattern;
pub mod regexp_pattern;
pub mod digits_pattern;
pub mod letters_pattern;
pub mod wildcard_pattern;
pub mod text_boundary;
pub mod sprintf;
pub mod str2double;
pub mod strcmp;
pub mod strcmpi;
pub mod string;
#[path = "string.empty.rs"]
pub mod string_empty;
pub mod strings;
pub mod strlength;
pub mod strncmp;
pub mod strncmpi;
pub mod unicode2native;

#[cfg(test)]
#[path = "character_property_tests.rs"]
mod character_property_tests;
