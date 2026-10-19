//! Incremental cache lifecycle under the shared kernel lease. The durable epoch
//! invalidates another process's inventory before any filesystem mutation. A
//! failed publication discards its inventory; recovery scans accepted receipts.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
mod inventory;
use inventory::{Inventory, Receipt};
const MAXIMUM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAXIMUM_REFERENCES: usize = 262_144;
const MAXIMUM_INDEX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Default)]
pub(super) struct Lifecycle {
    inventory: Option<Inventory>,
    #[cfg(test)]
    pub(super) discoveries: usize,
}
pub(super) struct Reservation {
    inventory: Inventory,
}
impl Lifecycle {
    /// Caller holds the cross-process cache lease through complete publication.
    /// No receipt-count heuristic may evict a byte/reference-feasible working set.
    pub(super) fn reserve(
        &mut self,
        index: &Path,
        objects: &Path,
        requested: u64,
        references: usize,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<Option<Reservation>, MeshingStageExecutionError> {
        check()?;
        if requested > MAXIMUM_BYTES || references > MAXIMUM_REFERENCES {
            return Ok(None);
        }
        let epoch_path = index.parent().unwrap().join("lifecycle-epoch");
        let epoch = match NativeFsProvider.read_bounded_sync(&epoch_path, 16) {
            Ok(bytes) if bytes.len() == 16 => Some(bytes),
            Ok(_) => return Err(error("cache lifecycle epoch has invalid size")),
            Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => None,
            Err(failure) => return Err(error(failure)),
        };
        let previous = self.inventory.take();
        let mut inventory = match previous {
            Some(inventory) if epoch.as_deref() == Some(inventory.epoch.as_slice()) => inventory,
            _ => {
                #[cfg(test)]
                {
                    self.discoveries += 1;
                }
                Inventory::read(index, objects, check)?
            }
        };
        // Publish a fresh invalidation token BEFORE deleting or creating files.
        // An interrupted owner has no usable in-memory inventory on its next call.
        inventory.epoch = *uuid::Uuid::new_v4().as_bytes();
        NativeFsProvider
            .atomic_replace_sync(&epoch_path, &inventory.epoch)
            .map_err(error)?;
        inventory.remove_orphans(check)?;
        while !inventory.fits(requested, references) {
            if !inventory.evict_oldest(objects, check)? {
                return Ok(None);
            }
        }
        Ok(Some(Reservation { inventory }))
    }
    pub(super) fn published(
        &mut self,
        mut reservation: Reservation,
        path: &Path,
        objects: &Path,
        receipt: &AcceptedMeshingStageReceipt,
        written: impl Iterator<Item = Digest>,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<(), MeshingStageExecutionError> {
        check()?;
        let length = std::fs::metadata(path).map_err(error)?.len();
        let digests = receipt
            .object_digests()
            .map(filename)
            .collect::<BTreeSet<_>>();
        reservation.inventory.add(
            path.to_owned(),
            Receipt {
                bytes: length,
                digests,
            },
            objects,
            check,
        )?;
        reservation
            .inventory
            .remove_unreferenced(objects, written, check)?;
        if !reservation.inventory.fits(0, 0) {
            return Err(error(
                "published cache closure exceeds its reserved lifecycle bounds",
            ));
        }
        self.inventory = Some(reservation.inventory);
        Ok(())
    }
}
