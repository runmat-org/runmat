//! Original external row identity retained by native numerical solver data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearRowRole {
    Equality(usize),
    Inequality(usize),
    Lower(usize),
    Upper(usize),
}
