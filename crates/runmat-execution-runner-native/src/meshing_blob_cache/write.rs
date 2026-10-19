use super::*;
use rusqlite::{params, DatabaseName, OptionalExtension, TransactionBehavior};
use std::io::Write;

impl BoundedMeshingBlobCache {
    pub(crate) fn write(
        &self,
        key: &[u8; 32],
        bytes: &[u8],
        workspace: u64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, CacheError> {
        let length = bytes.len() as u64;
        if length > MAXIMUM_ENTRY_BYTES
            || length > self.maximum_payload_bytes
            || self.maximum_entries == 0
        {
            return Ok(false);
        }
        let hash = digest(bytes, control)?;
        self.with_connection(control, |connection| {
            let transaction = match connection.transaction_with_behavior(TransactionBehavior::Immediate) {
                Ok(value) => value,
                Err(error) if busy(&error) => return Ok(false),
                Err(error) => return Err(storage(error)),
            };
            let previous = transaction.query_row(
                "SELECT rowid,digest,payload_bytes,materialization_bytes FROM cache_entries WHERE cache_key=?1",
                params![key.as_slice()], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,Vec<u8>>(1)?,row.get::<_,u64>(2)?,row.get::<_,u64>(3)?)),
            ).optional().map_err(storage)?;
            if let Some((rowid,old_hash,old_length,old_workspace)) = previous {
                if old_hash.as_slice()!=hash || old_length!=length || old_workspace!=workspace {
                    return Err(storage("one numerical cache identity produced different accepted bytes or storage inventory"));
                }
                // A fresh derivation can repair damaged storage in refresh mode.
                // Different accepted bytes under the same input still fail above.
                transaction.execute("UPDATE cache_entries SET payload=zeroblob(?1),touched=?2 WHERE rowid=?3",params![length,touched(),rowid]).map_err(storage)?;
                write_payload(&transaction,rowid,bytes,control)?;
                control.checkpoint().map_err(CacheError::Control)?;
                transaction.commit().map_err(storage)?;
                return Ok(true);
            }
            let (mut total,mut count) = transaction.query_row(
                "SELECT payload_bytes,entry_count FROM cache_state WHERE singleton=1",[],
                |row| Ok((row.get::<_,u64>(0)?,row.get::<_,u64>(1)?)),
            ).map_err(storage)?;
            if total>self.maximum_payload_bytes || count>self.maximum_entries { return Err(storage("cache quota state is invalid")); }
            while total.checked_add(length).is_none_or(|value| value>self.maximum_payload_bytes) || count>=self.maximum_entries {
                control.checkpoint().map_err(CacheError::Control)?;
                control.consume_search_work(1).map_err(CacheError::Control)?;
                let (rowid,evicted) = transaction.query_row(
                    "SELECT rowid,payload_bytes FROM cache_entries ORDER BY touched,cache_key LIMIT 1",[],
                    |row| Ok((row.get::<_,i64>(0)?,row.get::<_,u64>(1)?)),
                ).map_err(storage)?;
                transaction.execute("DELETE FROM cache_entries WHERE rowid=?1",params![rowid]).map_err(storage)?;
                total=total.checked_sub(evicted).ok_or_else(|| storage("cache eviction byte accounting underflow"))?;
                count=count.checked_sub(1).ok_or_else(|| storage("cache eviction count accounting underflow"))?;
            }
            transaction.execute(
                "INSERT INTO cache_entries(cache_key,payload,digest,payload_bytes,materialization_bytes,touched)
                 VALUES(?1,zeroblob(?2),?3,?2,?4,?5)",params![key.as_slice(),length,hash.as_slice(),workspace,touched()],
            ).map_err(storage)?;
            let rowid=transaction.last_insert_rowid();
            write_payload(&transaction,rowid,bytes,control)?;
            transaction.execute(
                "UPDATE cache_state SET payload_bytes=?1,entry_count=?2 WHERE singleton=1",params![total+length,count+1],
            ).map_err(storage)?;
            control.checkpoint().map_err(CacheError::Control)?;
            transaction.commit().map_err(storage)?;
            Ok(true)
        }).map(|value| value.unwrap_or(false))
    }
}

fn write_payload(
    transaction: &rusqlite::Transaction<'_>,
    rowid: i64,
    bytes: &[u8],
    control: &dyn GeometryEvaluationControl,
) -> Result<(), CacheError> {
    let mut blob = transaction
        .blob_open(DatabaseName::Main, "cache_entries", "payload", rowid, false)
        .map_err(storage)?;
    for chunk in bytes.chunks(IO_CHUNK_BYTES) {
        control.checkpoint().map_err(CacheError::Control)?;
        blob.write_all(chunk).map_err(storage)?;
    }
    Ok(())
}
