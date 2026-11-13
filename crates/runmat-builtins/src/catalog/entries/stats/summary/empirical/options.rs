//! Public empirical-distribution option spellings decode once at the boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmpiricalBuiltin {
    Ecdf,
    Cdfplot,
}

impl EmpiricalBuiltin {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ecdf => "ecdf",
            Self::Cdfplot => "cdfplot",
        }
    }
}

impl std::fmt::Display for EmpiricalBuiltin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EcdfOptionName {
    Function,
    Censoring,
    Frequency,
    Alpha,
    Bounds,
    IntervalIterationLimit,
    IntervalTolerance,
    IntervalIcmFrequency,
}

impl EcdfOptionName {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("function") {
            Some(Self::Function)
        } else if text.eq_ignore_ascii_case("censoring") {
            Some(Self::Censoring)
        } else if text.eq_ignore_ascii_case("frequency") {
            Some(Self::Frequency)
        } else if text.eq_ignore_ascii_case("alpha") {
            Some(Self::Alpha)
        } else if text.eq_ignore_ascii_case("bounds") {
            Some(Self::Bounds)
        } else if text.eq_ignore_ascii_case("iterationlimit") {
            Some(Self::IntervalIterationLimit)
        } else if text.eq_ignore_ascii_case("tolerance") {
            Some(Self::IntervalTolerance)
        } else if text.eq_ignore_ascii_case("icmfrequency") {
            Some(Self::IntervalIcmFrequency)
        } else {
            None
        }
    }

    pub fn is_interval_only(self) -> bool {
        matches!(
            self,
            Self::IntervalIterationLimit | Self::IntervalTolerance | Self::IntervalIcmFrequency
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EcdfFunction {
    Cdf,
    Survivor,
    CumulativeHazard,
}

impl EcdfFunction {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("cdf") {
            Some(Self::Cdf)
        } else if text.eq_ignore_ascii_case("survivor") {
            Some(Self::Survivor)
        } else if text.eq_ignore_ascii_case("cumulative hazard")
            || text.eq_ignore_ascii_case("cumhazard")
            || text.eq_ignore_ascii_case("hazard")
        {
            Some(Self::CumulativeHazard)
        } else {
            None
        }
    }

    pub fn y_label(self) -> &'static str {
        match self {
            Self::Cdf => "F(x)",
            Self::Survivor => "S(x)",
            Self::CumulativeHazard => "H(x)",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Cdf => "Empirical cdf",
            Self::Survivor => "Empirical survivor function",
            Self::CumulativeHazard => "Empirical cumulative hazard",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EcdfBounds {
    On,
    Off,
}

impl EcdfBounds {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("on") {
            Some(Self::On)
        } else if text.eq_ignore_ascii_case("off") {
            Some(Self::Off)
        } else {
            None
        }
    }
}
