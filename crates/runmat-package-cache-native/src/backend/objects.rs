use runmat_package::ContentDigest;
use rusqlite::{OptionalExtension, Transaction};

pub(crate) fn read(
    transaction: &Transaction<'_>,
    digest: &ContentDigest,
) -> rusqlite::Result<Option<Vec<u8>>> {
    transaction
        .query_row(
            "SELECT bytes FROM object_payloads WHERE digest = ?1",
            [digest.to_string()],
            |row| row.get(0),
        )
        .optional()
}

pub(crate) fn stored_bytes(transaction: &Transaction<'_>) -> rusqlite::Result<u64> {
    let total: i64 = transaction.query_row(
        "SELECT COALESCE(SUM(length(bytes)), 0) FROM object_payloads",
        [],
        |row| row.get(0),
    )?;
    u64::try_from(total).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Integer,
            Box::new(error),
        )
    })
}
