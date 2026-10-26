//! Public simulated-annealing choices are typed roles, not algorithm aliases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimAnnealing {
    Fast,
    Boltzmann,
}
impl OptimAnnealing {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "annealingfast" => Some(Self::Fast),
            "annealingboltz" => Some(Self::Boltzmann),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fast => "annealingfast",
            Self::Boltzmann => "annealingboltz",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimAcceptance {
    Annealing,
}
impl OptimAcceptance {
    pub fn parse(name: &str) -> Option<Self> {
        name.trim()
            .eq_ignore_ascii_case("acceptancesa")
            .then_some(Self::Annealing)
    }
    pub const fn name(self) -> &'static str {
        "acceptancesa"
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimTemperature {
    Exponential,
    Fast,
    Boltzmann,
}
impl OptimTemperature {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "temperatureexp" => Some(Self::Exponential),
            "temperaturefast" => Some(Self::Fast),
            "temperatureboltz" => Some(Self::Boltzmann),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exponential => "temperatureexp",
            Self::Fast => "temperaturefast",
            Self::Boltzmann => "temperatureboltz",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimAnnealData {
    Double,
    Custom,
}
impl OptimAnnealData {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "double" => Some(Self::Double),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Custom => "custom",
        }
    }
}
