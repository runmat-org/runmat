#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rgb2LabOptionName {
    ColorSpace,
    WhitePoint,
}

impl Rgb2LabOptionName {
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("ColorSpace") {
            Some(Self::ColorSpace)
        } else if name.eq_ignore_ascii_case("WhitePoint") {
            Some(Self::WhitePoint)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RgbColorSpace {
    Srgb,
    AdobeRgb1998,
    ProphotoRgb,
    LinearRgb,
}

impl RgbColorSpace {
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("srgb") {
            Some(Self::Srgb)
        } else if name.eq_ignore_ascii_case("adobe-rgb-1998") {
            Some(Self::AdobeRgb1998)
        } else if name.eq_ignore_ascii_case("prophoto-rgb") {
            Some(Self::ProphotoRgb)
        } else if name.eq_ignore_ascii_case("linear-rgb") {
            Some(Self::LinearRgb)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RgbWhitePoint {
    A,
    C,
    E,
    D50,
    D55,
    D65,
    Icc,
}

impl RgbWhitePoint {
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("a") {
            Some(Self::A)
        } else if name.eq_ignore_ascii_case("c") {
            Some(Self::C)
        } else if name.eq_ignore_ascii_case("e") {
            Some(Self::E)
        } else if name.eq_ignore_ascii_case("d50") {
            Some(Self::D50)
        } else if name.eq_ignore_ascii_case("d55") {
            Some(Self::D55)
        } else if name.eq_ignore_ascii_case("d65") {
            Some(Self::D65)
        } else if name.eq_ignore_ascii_case("icc") {
            Some(Self::Icc)
        } else {
            None
        }
    }

    pub const fn coordinates(self) -> [f64; 3] {
        match self {
            Self::A => [1.0985, 1.0, 0.3558],
            Self::C => [0.9807, 1.0, 1.1822],
            Self::E => [1.0, 1.0, 1.0],
            Self::D50 => [0.9642, 1.0, 0.8251],
            Self::D55 => [0.9568, 1.0, 0.9214],
            Self::D65 => [0.95047, 1.0, 1.08883],
            Self::Icc => [31595.0 / 32768.0, 1.0, 27030.0 / 32768.0],
        }
    }
}
