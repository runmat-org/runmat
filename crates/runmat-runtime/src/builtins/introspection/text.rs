pub(super) fn decode_utf16_with_error<F>(
    units: &[runmat_value::Utf16CodeUnit],
    error: F,
) -> crate::BuiltinResult<String>
where
    F: FnOnce() -> crate::RuntimeError,
{
    runmat_value::decode_utf16_code_units(units).map_err(|source| {
        let mut error = error();
        error.source = Some(Box::new(source));
        error
    })
}
