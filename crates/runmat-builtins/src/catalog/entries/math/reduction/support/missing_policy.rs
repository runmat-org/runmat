pub use runmat_types::ReductionNaN;

#[cfg(test)]
mod tests {
    use super::ReductionNaN;

    #[test]
    fn public_missing_policy_spellings_share_one_typed_identity() {
        for option in ["omitnan", "omitmissing", "OMITNAN"] {
            assert_eq!(
                ReductionNaN::from_public_option(option),
                Some(ReductionNaN::Omit)
            );
        }
        for option in ["includenan", "includemissing", "INCLUDEMISSING"] {
            assert_eq!(
                ReductionNaN::from_public_option(option),
                Some(ReductionNaN::Include)
            );
        }
        assert_eq!(ReductionNaN::from_public_option("native"), None);
    }
}
