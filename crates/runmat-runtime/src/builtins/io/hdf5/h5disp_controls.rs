//! Typed public argument forms for HDF5 display.

use super::{hdf5_error, hdf5_path, string_scalar, ERR_ARGUMENT, H5DISP_NAME};
use crate::BuiltinResult;
use runmat_value::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum H5DispMode {
    Simple,
    Min,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum H5DispTextEncoding {
    Utf8,
}

#[derive(Debug)]
pub(super) struct H5DispControls {
    pub location: String,
    pub mode: H5DispMode,
    pub text_encoding: Option<H5DispTextEncoding>,
}

pub(super) fn parse(rest: &[Value]) -> BuiltinResult<H5DispControls> {
    let mut controls = H5DispControls {
        location: "/".to_string(),
        mode: H5DispMode::Simple,
        text_encoding: None,
    };
    let (location, mode, encoding) =
        match rest {
            [] => (None, None, None),
            [location] => (Some(location), None, None),
            [first, value] if option_name(first)? => (None, None, Some((first, value))),
            [location, mode] => (Some(location), Some(mode), None),
            [location, option, value] => (Some(location), None, Some((option, value))),
            [location, mode, option, value] => (Some(location), Some(mode), Some((option, value))),
            _ => return Err(hdf5_error(
                H5DISP_NAME,
                &ERR_ARGUMENT,
                "h5disp: expected filename, optional location/mode, and optional TextEncoding pair",
            )),
        };
    if let Some(location) = location {
        controls.location = hdf5_path(location, H5DISP_NAME, "location")?;
    }
    if let Some(mode) = mode {
        let spelling = string_scalar(mode, H5DISP_NAME, "display mode")?;
        controls.mode = if spelling.eq_ignore_ascii_case("simple") {
            H5DispMode::Simple
        } else if spelling.eq_ignore_ascii_case("min") {
            H5DispMode::Min
        } else {
            return Err(hdf5_error(
                H5DISP_NAME,
                &ERR_ARGUMENT,
                "h5disp: display mode must be 'simple' or 'min'",
            ));
        };
    }
    if let Some((option, value)) = encoding {
        if !option_name(option)? {
            return Err(hdf5_error(
                H5DISP_NAME,
                &ERR_ARGUMENT,
                "h5disp: expected the TextEncoding option name",
            ));
        }
        let spelling = string_scalar(value, H5DISP_NAME, "TextEncoding value")?;
        if !spelling.eq_ignore_ascii_case("UTF-8") {
            return Err(hdf5_error(
                H5DISP_NAME,
                &ERR_ARGUMENT,
                "h5disp: TextEncoding currently accepts only 'UTF-8'",
            ));
        }
        controls.text_encoding = Some(H5DispTextEncoding::Utf8);
    }
    Ok(controls)
}

fn option_name(value: &Value) -> BuiltinResult<bool> {
    Ok(
        string_scalar(value, H5DISP_NAME, "location or option name")?
            .eq_ignore_ascii_case("TextEncoding"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documented_forms_and_invalid_suffixes() {
        let text = |value: &str| Value::String(value.to_string());
        let base = parse(&[]).unwrap();
        assert_eq!(base.location, "/");
        assert_eq!(base.mode, H5DispMode::Simple);
        assert!(base.text_encoding.is_none());
        let minimal = parse(&[text("/data"), text("min")]).unwrap();
        assert_eq!(minimal.location, "/data");
        assert_eq!(minimal.mode, H5DispMode::Min);
        let encoded = parse(&[text("TextEncoding"), text("UTF-8")]).unwrap();
        assert_eq!(encoded.location, "/");
        assert_eq!(encoded.text_encoding, Some(H5DispTextEncoding::Utf8));
        let full = parse(&[
            text("/data"),
            text("simple"),
            text("TextEncoding"),
            text("UTF-8"),
        ])
        .unwrap();
        assert_eq!(full.location, "/data");
        assert_eq!(full.mode, H5DispMode::Simple);
        assert!(parse(&[text("/data"), text("not-a-mode")]).is_err());
        assert!(parse(&[text("/data"), text("Other"), text("UTF-8")]).is_err());
        assert!(parse(&[text("TextEncoding"), text("Latin-1")]).is_err());
    }
}
