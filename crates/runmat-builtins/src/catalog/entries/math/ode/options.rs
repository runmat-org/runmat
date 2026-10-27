/// Recognized fields of the public ODE options structure. Field spellings are
/// decoded once at admission; the solver carries these identities, not names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeOptionField {
    RelTol,
    AbsTol,
    InitialStep,
    MaxStep,
    MaxSteps,
    Refine,
}

impl OdeOptionField {
    pub const ALL: [Self; 6] = [
        Self::RelTol,
        Self::AbsTol,
        Self::InitialStep,
        Self::MaxStep,
        Self::MaxSteps,
        Self::Refine,
    ];

    pub fn from_public_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("RelTol") {
            Some(Self::RelTol)
        } else if name.eq_ignore_ascii_case("AbsTol") {
            Some(Self::AbsTol)
        } else if name.eq_ignore_ascii_case("InitialStep") {
            Some(Self::InitialStep)
        } else if name.eq_ignore_ascii_case("MaxStep") {
            Some(Self::MaxStep)
        } else if name.eq_ignore_ascii_case("MaxSteps") {
            Some(Self::MaxSteps)
        } else if name.eq_ignore_ascii_case("Refine") {
            Some(Self::Refine)
        } else {
            None
        }
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::RelTol => "RelTol",
            Self::AbsTol => "AbsTol",
            Self::InitialStep => "InitialStep",
            Self::MaxStep => "MaxStep",
            Self::MaxSteps => "MaxSteps",
            Self::Refine => "Refine",
        }
    }

    pub const fn is_floating_control(self) -> bool {
        !matches!(self, Self::MaxSteps | Self::Refine)
    }
}
