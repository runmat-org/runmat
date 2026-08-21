use runmat_package_cache::CacheState;
use rusqlite::Connection;

pub(crate) const DATABASE_SCHEMA_VERSION: i64 = 1;

pub(crate) fn initialize(connection: &Connection) -> rusqlite::Result<()> {
    connection.execute_batch(
        "
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS cache_state (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            revision INTEGER NOT NULL CHECK (revision >= 0),
            state_json BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS object_payloads (
            digest TEXT PRIMARY KEY,
            bytes BLOB NOT NULL
        );
        ",
    )?;
    let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version == 0 {
        connection.pragma_update(None, "user_version", DATABASE_SCHEMA_VERSION)?;
    } else if version != DATABASE_SCHEMA_VERSION {
        return Err(rusqlite::Error::InvalidParameterName(format!(
            "unsupported cache database schema {version}"
        )));
    }
    let initial = serde_json::to_vec(&CacheState::default())
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    connection.execute(
        "INSERT OR IGNORE INTO cache_state (singleton, revision, state_json) VALUES (1, 0, ?1)",
        [initial],
    )?;
    Ok(())
}
