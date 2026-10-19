use super::*;
use rusqlite::{params, DatabaseName, OptionalExtension};
use std::io::Read;

impl BoundedMeshingBlobCache {
    pub(crate) fn read<'a>(
        &self,
        key: &[u8; 32],
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<CachedBlob<'a>>, CacheError> {
        self.with_connection(control, |connection| {
            let transaction = connection.transaction().map_err(storage)?;
            let metadata = transaction
                .query_row(
                    "SELECT rowid,payload_bytes,materialization_bytes,digest,length(payload)
                 FROM cache_entries WHERE cache_key=?1",
                    params![key.as_slice()],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, u64>(1)?,
                            row.get::<_, u64>(2)?,
                            row.get::<_, Vec<u8>>(3)?,
                            row.get::<_, u64>(4)?,
                        ))
                    },
                )
                .optional()
                .map_err(storage)?;
            let Some((rowid, length, workspace, expected, actual)) = metadata else {
                return Ok(None);
            };
            if length != actual || length > MAXIMUM_ENTRY_BYTES || expected.len() != 32 {
                return Err(storage("numerical cache blob metadata is invalid"));
            }
            let inventory = length
                .checked_add(workspace)
                .ok_or_else(|| storage("cache materialization inventory overflow"))?;
            let allocation_reservation =
                match runmat_geometry_core::GeometryAllocationReservation::new(control, inventory) {
                    Ok(reservation) => reservation,
                    Err(error)
                        if error.kind == GeometryEvaluationErrorKind::AllocationBudgetExceeded =>
                    {
                        return Ok(None)
                    }
                    Err(error) => return Err(CacheError::Control(error)),
                };
            let mut bytes = vec![0_u8; usize::try_from(length).map_err(storage)?];
            let mut blob = transaction
                .blob_open(DatabaseName::Main, "cache_entries", "payload", rowid, true)
                .map_err(storage)?;
            for chunk in bytes.chunks_mut(IO_CHUNK_BYTES) {
                control.checkpoint().map_err(CacheError::Control)?;
                blob.read_exact(chunk).map_err(storage)?;
            }
            drop(blob);
            transaction.commit().map_err(storage)?;
            if digest(&bytes, control)?.as_slice() != expected.as_slice() {
                return Err(storage(
                    "numerical cache blob digest differs from its retained record",
                ));
            }
            // Cache recency is operational metadata. Contention may skip this update;
            // it cannot invalidate a successfully read, immutable numerical record.
            if let Err(error) = connection.execute(
                "UPDATE cache_entries SET touched=max(touched,?1) WHERE rowid=?2",
                params![touched(), rowid],
            ) {
                if !busy(&error) {
                    return Err(storage(error));
                }
            }
            Ok(Some(CachedBlob {
                bytes,
                storage: allocation_reservation,
                materialization_resident_bytes: workspace,
            }))
        })
        .map(Option::flatten)
    }
}
