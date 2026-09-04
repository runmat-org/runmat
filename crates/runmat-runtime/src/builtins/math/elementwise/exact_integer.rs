use runmat_value::IntegerStorage;

const MAX_EXACT_BINARY64_INTEGER: i128 = 1_i128 << 53;

pub(super) fn is_exact_binary64(storage: &IntegerStorage) -> bool {
    match storage {
        IntegerStorage::I8(_) | IntegerStorage::I16(_) | IntegerStorage::I32(_) => true,
        IntegerStorage::I64(values) => values.iter().all(|&value| {
            (-MAX_EXACT_BINARY64_INTEGER..=MAX_EXACT_BINARY64_INTEGER).contains(&i128::from(value))
        }),
        IntegerStorage::U8(_) | IntegerStorage::U16(_) | IntegerStorage::U32(_) => true,
        IntegerStorage::U64(values) => values
            .iter()
            .all(|&value| u128::from(value) <= MAX_EXACT_BINARY64_INTEGER as u128),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_signed_and_unsigned_binary64_boundaries() {
        let boundary = 1_u64 << 53;
        assert!(is_exact_binary64(&IntegerStorage::I64(vec![
            -(boundary as i64),
            boundary as i64,
        ])));
        assert!(is_exact_binary64(&IntegerStorage::U64(vec![boundary])));
    }

    #[test]
    fn rejects_values_beyond_binary64_exact_integer_range() {
        let beyond = (1_u64 << 53) + 1;
        assert!(!is_exact_binary64(&IntegerStorage::I64(vec![
            -(beyond as i64),
            beyond as i64,
        ])));
        assert!(!is_exact_binary64(&IntegerStorage::U64(vec![beyond])));
    }

    #[test]
    fn all_narrow_integer_storage_is_exact() {
        assert!(is_exact_binary64(&IntegerStorage::I32(vec![
            i32::MIN,
            i32::MAX
        ])));
        assert!(is_exact_binary64(&IntegerStorage::U32(vec![
            u32::MIN,
            u32::MAX
        ])));
    }
}
