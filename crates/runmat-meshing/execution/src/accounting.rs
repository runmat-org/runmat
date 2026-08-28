pub(crate) fn checked_sum_lengths(mut lengths: impl Iterator<Item = usize>) -> Option<u64> {
    lengths.try_fold(0_u64, |total, length| {
        u64::try_from(length)
            .ok()
            .and_then(|length| total.checked_add(length))
    })
}
