/// The public spelling boundary for writer name-value options. Advanced
/// recognized names remain distinct from unknown names so they can retain
/// the stable unsupported-option error rather than being parsed as a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteOption {
    Alpha,
    Quality,
    WriteMode,
    DelayTime,
    LoopCount,
    UnsupportedAdvanced,
}

impl ImwriteOption {
    pub fn from_name(name: &str) -> Option<Self> {
        let normalized: String = name
            .chars()
            .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
            .flat_map(char::to_lowercase)
            .collect();
        match normalized.as_str() {
            "alpha" => Some(Self::Alpha),
            "quality" => Some(Self::Quality),
            "writemode" => Some(Self::WriteMode),
            "delaytime" => Some(Self::DelayTime),
            "loopcount" => Some(Self::LoopCount),
            "compression" | "bitdepth" | "mode" | "disposalmethod" | "backgroundcolor"
            | "comment" | "transparentcolor" => Some(Self::UnsupportedAdvanced),
            _ => None,
        }
    }

    pub fn validate_numeric(self, number: f64) -> Result<(), &'static str> {
        match self {
            Self::Quality if !number.is_finite() || !(0.0..=100.0).contains(&number) => {
                Err("Quality must be a scalar from 0 to 100")
            }
            Self::DelayTime if !number.is_finite() || number < 0.0 || number >= u64::MAX as f64 => {
                Err("DelayTime must be a finite non-negative duration representable by the GIF encoder")
            }
            Self::LoopCount if number.is_nan() || number < 0.0 => {
                Err("LoopCount must be an integer between 0 and 65535, or Inf")
            }
            Self::LoopCount if number.is_finite() && {
                let rounded = number.round();
                (rounded - number).abs() > 1e-6 || rounded > u16::MAX as f64
            } => Err("LoopCount must be an integer between 0 and 65535, or Inf"),
            Self::LoopCount if number.is_infinite() && number.is_sign_negative() => {
                Err("LoopCount must be an integer between 0 and 65535, or Inf")
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteWriteMode {
    Overwrite,
    Append,
}

impl ImwriteWriteMode {
    pub fn from_name(mode: &str) -> Option<Self> {
        let mode = mode.trim();
        if mode.eq_ignore_ascii_case("overwrite") {
            Some(Self::Overwrite)
        } else if mode.eq_ignore_ascii_case("append") {
            Some(Self::Append)
        } else {
            None
        }
    }
}

/// Supported encoder identity, decoded once from public format spellings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteFormat {
    Png,
    Jpeg,
    Bmp,
    Gif,
    Tiff,
}

impl ImwriteFormat {
    pub fn from_hint(hint: &str) -> Option<Self> {
        let label = hint.trim().trim_start_matches('.');
        if label.eq_ignore_ascii_case("png") {
            Some(Self::Png)
        } else if ["jpg", "jpeg", "jpe"]
            .iter()
            .any(|name| label.eq_ignore_ascii_case(name))
        {
            Some(Self::Jpeg)
        } else if label.eq_ignore_ascii_case("bmp") {
            Some(Self::Bmp)
        } else if label.eq_ignore_ascii_case("gif") {
            Some(Self::Gif)
        } else if ["tif", "tiff"]
            .iter()
            .any(|name| label.eq_ignore_ascii_case(name))
        {
            Some(Self::Tiff)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImwriteAdmissionFailure {
    InvalidImage(&'static str),
    InvalidOption(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImwriteEncoderFacts {
    pub format: ImwriteFormat,
    pub channels: usize,
    pub sixteen_bit: bool,
    pub indexed_source: bool,
    pub alpha_applied: bool,
    pub append: bool,
}

impl ImwriteEncoderFacts {
    pub fn validate(self) -> Result<(), ImwriteAdmissionFailure> {
        use ImwriteAdmissionFailure::{InvalidImage, InvalidOption};
        use ImwriteFormat::*;
        if self.channels == 4 && !self.alpha_applied && self.format != Png {
            return Err(InvalidImage(if self.format == Tiff {
                "CMYK TIFF encoding is not supported by this RunMat encoder"
            } else {
                "direct four-channel input is supported for PNG only"
            }));
        }
        if self.append && self.format != Gif {
            return Err(InvalidOption(
                "WriteMode 'append' is supported for GIF files only",
            ));
        }
        match self.format {
            Gif if (self.channels == 3 && !self.indexed_source) || self.sixteen_bit => Err(
                InvalidImage("GIF direct input must be grayscale/indexed 8-bit data"),
            ),
            Jpeg if self.channels == 4 => {
                Err(InvalidOption("JPEG does not support alpha channels"))
            }
            Jpeg if self.sixteen_bit => Err(InvalidImage(
                "16-bit JPEG encoding is not supported by this RunMat encoder",
            )),
            Bmp if self.channels == 4 => Err(InvalidOption("BMP alpha output is not supported")),
            Bmp if self.sixteen_bit => {
                Err(InvalidImage("BMP does not support this 16-bit image input"))
            }
            Tiff if self.channels == 4 => Err(InvalidImage(
                "CMYK TIFF encoding is not supported by this RunMat encoder",
            )),
            _ => Ok(()),
        }
    }
}
