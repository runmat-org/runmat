//! One encoding policy for batch text readers and incremental file input.

use encoding_rs::{Encoding, UTF_8};

mod input;
pub(crate) use input::{EncodedInput, EncodedInputStep, TextInputBox};

#[derive(Clone, Copy)]
pub(crate) enum TextEncoding {
    Bytes,
    Ascii,
    Encoded(&'static Encoding),
}

impl TextEncoding {
    pub(crate) fn parse(label: &str) -> Result<Self, String> {
        let label = label.trim();
        if label.is_empty()
            || label.eq_ignore_ascii_case("utf-8")
            || label.eq_ignore_ascii_case("utf8")
        {
            return Ok(Self::Encoded(UTF_8));
        }
        if ["binary", "latin1", "latin-1", "iso-8859-1"]
            .iter()
            .any(|alias| label.eq_ignore_ascii_case(alias))
        {
            return Ok(Self::Bytes);
        }
        if ["ascii", "us-ascii", "us_ascii", "usascii"]
            .iter()
            .any(|alias| label.eq_ignore_ascii_case(alias))
        {
            return Ok(Self::Ascii);
        }
        if label.eq_ignore_ascii_case("system") {
            return Ok(Self::Encoded(if cfg!(windows) {
                encoding_rs::WINDOWS_1252
            } else {
                UTF_8
            }));
        }
        Encoding::for_label(label.as_bytes())
            .map(Self::Encoded)
            .ok_or_else(|| format!("unsupported encoding '{label}'"))
    }
}
