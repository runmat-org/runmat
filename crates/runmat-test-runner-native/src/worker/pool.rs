pub(super) fn validate_capacity(max_workers: usize) -> Result<(), &'static str> {
    if max_workers == 0 {
        Err("native worker capacity must be greater than zero")
    } else {
        Ok(())
    }
}
