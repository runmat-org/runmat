/// Registry identity used by method, protocol, event, and name admissions.
/// A selected immutable class product can never collide with a mutable
/// session generation, even if either counter wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClassRegistryRevision {
    Current(u64),
    Selected(u64),
}

#[cfg(test)]
mod tests {
    use super::ClassRegistryRevision;

    #[test]
    fn selected_and_current_revisions_cannot_collide_at_the_same_serial() {
        assert_ne!(
            ClassRegistryRevision::Current(1),
            ClassRegistryRevision::Selected(1)
        );
    }
}
