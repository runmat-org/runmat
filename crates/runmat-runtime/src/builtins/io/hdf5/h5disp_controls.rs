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

pub(super) fn parse(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<H5DispControls> {
    let mut controls = H5DispControls {
        location: "/".to_string(),
        mode: H5DispMode::Simple,
        text_encoding: None,
    };
    let (location, mode, encoding) =
        match rest.len() {
            0 => (None, None, None),
            1 => (Some(&rest[0]), None, None),
            2 if option_name(&rest[0])? => (None, None, Some((&rest[0], &rest[1]))),
            2 => (Some(&rest[0]), Some(&rest[1]), None),
            3 => (Some(&rest[0]), None, Some((&rest[1], &rest[2]))),
            4 => (Some(&rest[0]), Some(&rest[1]), Some((&rest[2], &rest[3]))),
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
        let text = |value: &str| Value::String(value.to_string().into());
        let base = parse(crate::call::FunctionArgumentView::from_slice(&[])).unwrap();
        assert_eq!(base.location, "/");
        assert_eq!(base.mode, H5DispMode::Simple);
        assert!(base.text_encoding.is_none());
        let minimal = parse(crate::call::FunctionArgumentView::from_slice(&[
            text("/data"),
            text("min"),
        ]))
        .unwrap();
        assert_eq!(minimal.location, "/data");
        assert_eq!(minimal.mode, H5DispMode::Min);
        let encoded = parse(crate::call::FunctionArgumentView::from_slice(&[
            text("TextEncoding"),
            text("UTF-8"),
        ]))
        .unwrap();
        assert_eq!(encoded.location, "/");
        assert_eq!(encoded.text_encoding, Some(H5DispTextEncoding::Utf8));
        let full = parse(crate::call::FunctionArgumentView::from_slice(&[
            text("/data"),
            text("simple"),
            text("TextEncoding"),
            text("UTF-8"),
        ]))
        .unwrap();
        assert_eq!(full.location, "/data");
        assert_eq!(full.mode, H5DispMode::Simple);
        assert!(parse(crate::call::FunctionArgumentView::from_slice(&[
            text("/data"),
            text("not-a-mode")
        ]))
        .is_err());
        assert!(parse(crate::call::FunctionArgumentView::from_slice(&[
            text("/data"),
            text("Other"),
            text("UTF-8")
        ]))
        .is_err());
        assert!(parse(crate::call::FunctionArgumentView::from_slice(&[
            text("TextEncoding"),
            text("Latin-1")
        ]))
        .is_err());
    }
}
