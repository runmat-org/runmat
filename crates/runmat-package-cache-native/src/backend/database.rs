use super::schema;
use crate::{NativeCacheConfig, NativeCacheError};
use runmat_package_cache::CacheState;
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

pub struct SqliteCacheBackend {
    pub(crate) connection: Mutex<Connection>,
    pub(crate) quota_bytes: Option<u64>,
}

impl std::fmt::Debug for SqliteCacheBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteCacheBackend")
            .field("quota_bytes", &self.quota_bytes)
            .finish_non_exhaustive()
    }
}

impl SqliteCacheBackend {
    pub fn open(config: &NativeCacheConfig) -> Result<Self, NativeCacheError> {
        config.validate()?;
        let layout = config.layout();
        layout.create()?;
        Self::open_path(&layout.database, config.quota_bytes)
    }

    pub fn open_path(
        path: impl AsRef<Path>,
        quota_bytes: Option<u64>,
    ) -> Result<Self, NativeCacheError> {
        let connection = Connection::open(path)?;
        configure(&connection)?;
        schema::initialize(&connection)?;
        validate_state(&connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
            quota_bytes,
        })
    }

    pub fn open_in_memory(quota_bytes: Option<u64>) -> Result<Self, NativeCacheError> {
        let connection = Connection::open_in_memory()?;
        configure(&connection)?;
        schema::initialize(&connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
            quota_bytes,
        })
    }
}

fn configure(connection: &Connection) -> Result<(), NativeCacheError> {
    connection.busy_timeout(Duration::from_secs(30))?;
    connection.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

fn validate_state(connection: &Connection) -> Result<(), NativeCacheError> {
    let bytes: Vec<u8> = connection.query_row(
        "SELECT state_json FROM cache_state WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    let state: CacheState = serde_json::from_slice(&bytes)?;
    state
        .validate()
        .map_err(|error| NativeCacheError::Config(error.to_string()))
}
