/// Closed option vocabulary shared by admission and static inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TtestOptionName {
    Alpha,
    Dim,
    Tail,
}
impl TtestOptionName {
    pub fn from_public(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Alpha") {
            Some(Self::Alpha)
        } else if name.eq_ignore_ascii_case("Dim") {
            Some(Self::Dim)
        } else if name.eq_ignore_ascii_case("Tail") {
            Some(Self::Tail)
        } else {
            None
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TtestTail {
    Both,
    Right,
    Left,
}
impl TtestTail {
    pub fn from_public(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("both") {
            Some(Self::Both)
        } else if value.eq_ignore_ascii_case("right") {
            Some(Self::Right)
        } else if value.eq_ignore_ascii_case("left") {
            Some(Self::Left)
        } else {
            None
        }
    }
}
