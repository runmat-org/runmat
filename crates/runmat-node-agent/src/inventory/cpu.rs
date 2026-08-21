pub(super) fn millicores() -> u64 {
    u64::try_from(
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    )
    .unwrap_or(1)
    .saturating_mul(1_000)
}
