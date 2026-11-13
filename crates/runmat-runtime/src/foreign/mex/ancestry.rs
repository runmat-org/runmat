/// Number of MEX callbacks in the logical invocation ancestry. Pending
/// siblings share the same parent depth; they do not increment this fact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MexCallbackAncestry(u16);

pub(super) const MAX_CALLBACK_DEPTH: u16 = 64;

impl MexCallbackAncestry {
    pub(crate) const ROOT: Self = Self(0);

    pub(crate) fn from_wire(depth: u16) -> Option<Self> {
        (depth <= MAX_CALLBACK_DEPTH).then_some(Self(depth))
    }

    pub(crate) fn child(self) -> Option<Self> {
        self.0
            .checked_add(1)
            .filter(|depth| *depth <= MAX_CALLBACK_DEPTH)
            .map(Self)
    }

    pub(crate) fn wire(self) -> u16 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_sixty_four_callback_ancestors_are_admitted() {
        let mut depth = MexCallbackAncestry::ROOT;
        for expected in 1..=MAX_CALLBACK_DEPTH {
            depth = depth.child().expect("ancestor within limit");
            assert_eq!(depth.wire(), expected);
            assert_eq!(MexCallbackAncestry::from_wire(expected), Some(depth));
        }
        assert!(depth.child().is_none());
        assert!(MexCallbackAncestry::from_wire(MAX_CALLBACK_DEPTH + 1).is_none());
    }
}
