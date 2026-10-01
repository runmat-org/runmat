/// The public numeric flag is decoded once at the call boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RidgeScaling {
    Standardized,
    Original,
}

impl RidgeScaling {
    pub fn from_float(value: f64) -> Option<Self> {
        const TOLERANCE: f64 = 1.0e-12;
        if value.abs() <= TOLERANCE {
            Some(Self::Original)
        } else if (value - 1.0).abs() <= TOLERANCE {
            Some(Self::Standardized)
        } else {
            None
        }
    }

    pub fn from_integer(value: i128) -> Option<Self> {
        match value {
            0 => Some(Self::Original),
            1 => Some(Self::Standardized),
            _ => None,
        }
    }

    pub fn from_bool(value: bool) -> Self {
        if value {
            Self::Standardized
        } else {
            Self::Original
        }
    }
}
