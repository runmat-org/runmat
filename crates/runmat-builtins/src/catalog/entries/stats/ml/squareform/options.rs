//! Closed squareform conversion directions; textual spellings are decoded only at the call boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SquareformDirection {
    Auto,
    ToMatrix,
    ToVector,
}

impl SquareformDirection {
    pub fn from_keyword(keyword: &str) -> Option<Self> {
        if keyword.eq_ignore_ascii_case("tomatrix") {
            Some(Self::ToMatrix)
        } else if keyword.eq_ignore_ascii_case("tovector") {
            Some(Self::ToVector)
        } else {
            None
        }
    }
}

/// Checked condensed-distance geometry shared by inference and execution.
pub fn condensed_length(size: usize) -> Option<usize> {
    let previous = size.saturating_sub(1);
    if size.is_multiple_of(2) {
        (size / 2).checked_mul(previous)
    } else {
        size.checked_mul(previous / 2)
    }
}

pub fn matrix_size_for_condensed_length(length: usize) -> Option<usize> {
    if length == 0 {
        return Some(1);
    }
    let mut lower = 2usize;
    let mut upper = usize::MAX;
    while lower <= upper {
        let middle = lower + (upper - lower) / 2;
        match condensed_length(middle) {
            Some(candidate) if candidate == length => return Some(middle),
            Some(candidate) if candidate < length => lower = middle.checked_add(1)?,
            _ => upper = middle - 1,
        }
    }
    None
}
