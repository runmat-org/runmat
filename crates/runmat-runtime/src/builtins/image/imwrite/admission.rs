use super::*;
use runmat_builtins::{ImwriteAdmissionFailure, ImwriteEncoderFacts, ImwriteFormat};

/// Facts available from admitted host values or a resident descriptor.
pub(super) struct EncoderFacts {
    pub format: ImageFormat,
    pub channels: usize,
    pub sixteen_bit: bool,
    pub indexed_source: bool,
    pub alpha_applied: bool,
    pub append: bool,
}

pub(super) fn typed_format(format: ImageFormat) -> BuiltinResult<ImwriteFormat> {
    match format {
        ImageFormat::Png => Ok(ImwriteFormat::Png),
        ImageFormat::Jpeg => Ok(ImwriteFormat::Jpeg),
        ImageFormat::Bmp => Ok(ImwriteFormat::Bmp),
        ImageFormat::Gif => Ok(ImwriteFormat::Gif),
        ImageFormat::Tiff => Ok(ImwriteFormat::Tiff),
        _ => Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_FORMAT,
            "unsupported image format",
        )),
    }
}

pub(super) fn validate(facts: &EncoderFacts) -> BuiltinResult<()> {
    ImwriteEncoderFacts {
        format: typed_format(facts.format)?,
        channels: facts.channels,
        sixteen_bit: facts.sixteen_bit,
        indexed_source: facts.indexed_source,
        alpha_applied: facts.alpha_applied,
        append: facts.append,
    }
    .validate()
    .map_err(|failure| match failure {
        ImwriteAdmissionFailure::InvalidImage(detail) => {
            imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_IMAGE, detail)
        }
        ImwriteAdmissionFailure::InvalidOption(detail) => {
            imwrite_error_with_detail(&IMWRITE_ERROR_INVALID_OPTION, detail)
        }
    })
}
