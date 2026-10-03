use super::*;

/// Facts available from admitted host values or a resident handle descriptor
/// before a provider download. No pixel contents are presumed here.
pub(super) struct EncoderFacts {
    pub format: ImageFormat,
    pub channels: usize,
    pub sixteen_bit: bool,
    pub indexed_source: bool,
    pub alpha_applied: bool,
    pub append: bool,
}

pub(super) fn validate(facts: &EncoderFacts) -> BuiltinResult<()> {
    if facts.channels == 4 && !facts.alpha_applied && facts.format != ImageFormat::Png {
        let detail = if facts.format == ImageFormat::Tiff {
            "CMYK TIFF encoding is not supported by this RunMat encoder"
        } else {
            "direct four-channel input is supported for PNG only"
        };
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_IMAGE,
            detail,
        ));
    }
    if facts.append && facts.format != ImageFormat::Gif {
        return Err(imwrite_error_with_detail(
            &IMWRITE_ERROR_INVALID_OPTION,
            "WriteMode 'append' is supported for GIF files only",
        ));
    }
    match facts.format {
        ImageFormat::Gif => {
            if (facts.channels == 3 && !facts.indexed_source) || facts.sixteen_bit {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_IMAGE,
                    "GIF direct input must be grayscale/indexed 8-bit data",
                ));
            }
        }
        ImageFormat::Jpeg => {
            if facts.channels == 4 {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "JPEG does not support alpha channels",
                ));
            }
            if facts.sixteen_bit {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_IMAGE,
                    "16-bit JPEG encoding is not supported by this RunMat encoder",
                ));
            }
        }
        ImageFormat::Bmp => {
            if facts.channels == 4 {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_OPTION,
                    "BMP alpha output is not supported",
                ));
            }
            if facts.sixteen_bit {
                return Err(imwrite_error_with_detail(
                    &IMWRITE_ERROR_INVALID_IMAGE,
                    "BMP does not support this 16-bit image input",
                ));
            }
        }
        ImageFormat::Tiff if facts.channels == 4 => {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_IMAGE,
                "CMYK TIFF encoding is not supported by this RunMat encoder",
            ));
        }
        ImageFormat::Png | ImageFormat::Tiff => {}
        _ => {
            return Err(imwrite_error_with_detail(
                &IMWRITE_ERROR_INVALID_FORMAT,
                "unsupported image format",
            ));
        }
    }
    Ok(())
}
