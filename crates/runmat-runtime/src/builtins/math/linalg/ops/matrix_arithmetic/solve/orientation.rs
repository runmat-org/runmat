#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::linalg) enum SolveOrientation {
    Left,
    Right,
}

impl SolveOrientation {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Left => "mldivide",
            Self::Right => "mrdivide",
        }
    }
}
