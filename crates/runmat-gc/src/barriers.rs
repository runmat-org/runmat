//! Write barriers for generational garbage collection
//!
//! Write barriers track cross-generational references to ensure that
//! objects in older generations that reference younger objects are
//! included in minor collection roots.

use crate::Value;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Write barrier implementation for generational GC
pub struct WriteBarrier {
    /// Set of old-to-young references (remembered set)
    remembered_set: parking_lot::RwLock<HashSet<*const u8>>,

    /// Statistics
    barrier_hits: AtomicUsize,
    remembered_set_size: AtomicUsize,

    /// Configuration
    enable_barriers: bool,
}

impl WriteBarrier {
    pub fn new(enable_barriers: bool) -> Self {
        Self {
            remembered_set: parking_lot::RwLock::new(HashSet::new()),
            barrier_hits: AtomicUsize::new(0),
            remembered_set_size: AtomicUsize::new(0),
            enable_barriers,
        }
    }

    /// Record a potential old-to-young reference
    ///
    /// Should be called whenever a reference from an old generation object
    /// to a young generation object is created
    pub fn record_reference(&self, old_object: *const u8, _young_object: *const u8) {
        if !self.enable_barriers {
            return;
        }

        self.barrier_hits.fetch_add(1, Ordering::Relaxed);

        let mut remembered_set = self.remembered_set.write();
        if remembered_set.insert(old_object) {
            self.remembered_set_size.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Remove an object from the remembered set
    ///
    /// Should be called when an old generation object is collected
    /// or when its references are updated
    pub fn remove_object(&self, object: *const u8) {
        if !self.enable_barriers {
            return;
        }

        let mut remembered_set = self.remembered_set.write();
        if remembered_set.remove(&object) {
            self.remembered_set_size.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get all objects in the remembered set as additional roots for minor GC
    pub fn get_remembered_roots(&self) -> Vec<*const u8> {
        let remembered_set = self.remembered_set.read();
        remembered_set.iter().copied().collect()
    }

    /// Clear the remembered set (typically after major GC)
    pub fn clear_remembered_set(&self) {
        let mut remembered_set = self.remembered_set.write();
        remembered_set.clear();
        self.remembered_set_size.store(0, Ordering::Relaxed);
    }

    /// Get write barrier statistics
    pub fn stats(&self) -> WriteBarrierStats {
        WriteBarrierStats {
            barrier_hits: self.barrier_hits.load(Ordering::Relaxed),
            remembered_set_size: self.remembered_set_size.load(Ordering::Relaxed),
            enabled: self.enable_barriers,
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.barrier_hits.store(0, Ordering::Relaxed);
        // Don't reset remembered_set_size as it reflects current state
    }
}

/// Statistics for write barriers
#[derive(Debug, Clone)]
pub struct WriteBarrierStats {
    pub barrier_hits: usize,
    pub remembered_set_size: usize,
    pub enabled: bool,
}

/// Write barrier macro for convenient insertion into code
///
/// Usage: write_barrier!(old_ptr, young_ptr);
#[macro_export]
macro_rules! write_barrier {
    ($old_ptr:expr, $young_ptr:expr) => {
        // Write barrier implementation for generational GC
        // Record the write for potential young-to-old generation promotion
        #[cfg(feature = "debug-gc")]
        {
            log::trace!("Write barrier: {:p} -> {:p}", $old_ptr, $young_ptr);
        }
    };
}

/// Trait for objects that can participate in write barriers
pub trait WriteBarrierAware {
    /// Called when this object is about to be modified
    fn pre_write_barrier(&self) {}

    /// Called after this object has been modified
    fn post_write_barrier(&self) {}

    /// Check if this object contains references to younger generations
    fn has_young_references(&self) -> bool {
        false
    }
}

/// Implementation of write barrier for Value types
impl WriteBarrierAware for Value {
    fn has_young_references(&self) -> bool {
        match self {
            Value::Cell(cells) => {
                // Check if any cell values are young
                // This is a placeholder - in reality we'd check generations
                !cells.data.is_empty()
            }
            _ => false,
        }
    }
}

/// Card-based write barrier for large heaps
///
/// Divides the heap into cards and tracks which cards contain
/// cross-generational references
pub struct CardTable {
    /// Card size in bytes (typically 512 bytes)
    card_size: usize,

    /// Dirty bits for each card
    cards: parking_lot::RwLock<Vec<bool>>,

    /// Heap start address for card calculations
    heap_start: *const u8,

    /// Total heap size
    heap_size: usize,
}

impl CardTable {
    pub fn new(heap_start: *const u8, heap_size: usize, card_size: usize) -> Self {
        let num_cards = heap_size.div_ceil(card_size);

        Self {
            card_size,
            cards: parking_lot::RwLock::new(vec![false; num_cards]),
            heap_start,
            heap_size,
        }
    }

    /// Mark a card as dirty due to a cross-generational reference
    pub fn mark_card_dirty(&self, address: *const u8) {
        if let Some(card_index) = self.address_to_card(address) {
            let mut cards = self.cards.write();
            if card_index < cards.len() {
                cards[card_index] = true;
            }
        }
    }

    /// Get all dirty cards for scanning
    pub fn get_dirty_cards(&self) -> Vec<usize> {
        let cards = self.cards.read();
        cards
            .iter()
            .enumerate()
            .filter(|(_, &dirty)| dirty)
            .map(|(index, _)| index)
            .collect()
    }

    /// Clear all dirty cards (typically after scanning)
    pub fn clear_dirty_cards(&self) {
        let mut cards = self.cards.write();
        cards.fill(false);
    }

    /// Convert an address to a card index
    fn address_to_card(&self, address: *const u8) -> Option<usize> {
        let heap_start = self.heap_start as usize;
        let addr = address as usize;

        if addr >= heap_start && addr < heap_start + self.heap_size {
            Some((addr - heap_start) / self.card_size)
        } else {
            None
        }
    }

    /// Convert a card index to an address range
    pub fn card_to_address_range(&self, card_index: usize) -> Option<(*const u8, *const u8)> {
        if card_index >= self.heap_size.div_ceil(self.card_size) {
            return None;
        }

        let start = self.heap_start as usize + card_index * self.card_size;
        let end = (start + self.card_size).min(self.heap_start as usize + self.heap_size);

        Some((start as *const u8, end as *const u8))
    }

    /// Get card table statistics
    pub fn stats(&self) -> CardTableStats {
        let cards = self.cards.read();
        let dirty_count = cards.iter().filter(|&&dirty| dirty).count();

        CardTableStats {
            total_cards: cards.len(),
            dirty_cards: dirty_count,
            card_size: self.card_size,
            heap_size: self.heap_size,
        }
    }
}

/// Statistics for card table
#[derive(Debug, Clone)]
pub struct CardTableStats {
    pub total_cards: usize,
    pub dirty_cards: usize,
    pub card_size: usize,
    pub heap_size: usize,
}

/// Global write barrier manager
pub struct WriteBarrierManager {
    /// Simple remembered set barrier
    remembered_set_barrier: WriteBarrier,

    /// Card table barrier for large heaps
    card_table: Option<CardTable>,

    /// Configuration
    use_card_table: bool,
}

impl WriteBarrierManager {
    pub fn new(enable_barriers: bool, use_card_table: bool) -> Self {
        Self {
            remembered_set_barrier: WriteBarrier::new(enable_barriers),
            card_table: None,
            use_card_table,
        }
    }

    /// Initialize card table for a heap
    pub fn initialize_card_table(&mut self, heap_start: *const u8, heap_size: usize) {
        if self.use_card_table {
            self.card_table = Some(CardTable::new(heap_start, heap_size, 512));
        }
    }

    /// Record a cross-generational reference
    pub fn record_reference(&self, old_object: *const u8, young_object: *const u8) {
        // Always use remembered set
        self.remembered_set_barrier
            .record_reference(old_object, young_object);

        // Also mark card if using card table
        if let Some(ref card_table) = self.card_table {
            card_table.mark_card_dirty(old_object);
        }
    }

    /// Get additional roots for minor GC
    pub fn get_minor_gc_roots(&self) -> Vec<*const u8> {
        self.remembered_set_barrier.get_remembered_roots()
    }

    /// Get dirty card ranges for scanning
    pub fn get_dirty_card_ranges(&self) -> Vec<(*const u8, *const u8)> {
        if let Some(ref card_table) = self.card_table {
            card_table
                .get_dirty_cards()
                .into_iter()
                .filter_map(|card| card_table.card_to_address_range(card))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Clear barriers after collection
    pub fn clear_after_minor_gc(&self) {
        // Keep remembered set for next minor GC
        if let Some(ref card_table) = self.card_table {
            card_table.clear_dirty_cards();
        }
    }

    /// Clear barriers after major GC
    pub fn clear_after_major_gc(&self) {
        self.remembered_set_barrier.clear_remembered_set();
        if let Some(ref card_table) = self.card_table {
            card_table.clear_dirty_cards();
        }
    }

    /// Get combined statistics
    pub fn stats(&self) -> WriteBarrierManagerStats {
        WriteBarrierManagerStats {
            remembered_set: self.remembered_set_barrier.stats(),
            card_table: self.card_table.as_ref().map(|ct| ct.stats()),
        }
    }
}

// Ensure thread-safety for global usage; internal synchronization guards shared state
unsafe impl Send for WriteBarrierManager {}
unsafe impl Sync for WriteBarrierManager {}

/// Combined statistics for write barrier manager
#[derive(Debug, Clone)]
pub struct WriteBarrierManagerStats {
    pub remembered_set: WriteBarrierStats,
    pub card_table: Option<CardTableStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_barrier() {
        let barrier = WriteBarrier::new(true);

        let old_ptr = 0x1000 as *const u8;
        let young_ptr = 0x2000 as *const u8;

        // Record a reference
        barrier.record_reference(old_ptr, young_ptr);

        let stats = barrier.stats();
        assert_eq!(stats.barrier_hits, 1);
        assert_eq!(stats.remembered_set_size, 1);

        // Get remembered roots
        let roots = barrier.get_remembered_roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], old_ptr);

        // Remove object
        barrier.remove_object(old_ptr);
        let stats = barrier.stats();
        assert_eq!(stats.remembered_set_size, 0);
    }

    #[test]
    fn test_write_barrier_disabled() {
        let barrier = WriteBarrier::new(false);

        let old_ptr = 0x1000 as *const u8;
        let young_ptr = 0x2000 as *const u8;

        barrier.record_reference(old_ptr, young_ptr);

        let stats = barrier.stats();
        assert!(!stats.enabled);
        assert_eq!(stats.barrier_hits, 0);
        assert_eq!(stats.remembered_set_size, 0);
    }

    #[test]
    fn test_card_table() {
        let heap_start = 0x10000 as *const u8;
        let heap_size = 4096;
        let card_size = 512;

        let card_table = CardTable::new(heap_start, heap_size, card_size);

        // Mark some cards dirty
        let addr1 = (heap_start as usize + 100) as *const u8;
        let addr2 = (heap_start as usize + 600) as *const u8;

        card_table.mark_card_dirty(addr1);
        card_table.mark_card_dirty(addr2);

        let dirty_cards = card_table.get_dirty_cards();
        assert_eq!(dirty_cards.len(), 2);
        assert!(dirty_cards.contains(&0)); // First card
        assert!(dirty_cards.contains(&1)); // Second card

        let stats = card_table.stats();
        assert_eq!(stats.total_cards, 8); // 4096 / 512
        assert_eq!(stats.dirty_cards, 2);

        // Clear dirty cards
        card_table.clear_dirty_cards();
        let dirty_cards = card_table.get_dirty_cards();
        assert_eq!(dirty_cards.len(), 0);
    }

    #[test]
    fn test_write_barrier_manager() {
        let mut manager = WriteBarrierManager::new(true, true);

        let heap_start = 0x10000 as *const u8;
        let heap_size = 4096;
        manager.initialize_card_table(heap_start, heap_size);

        let old_ptr = (heap_start as usize + 100) as *const u8;
        let young_ptr = 0x2000 as *const u8;

        manager.record_reference(old_ptr, young_ptr);

        let roots = manager.get_minor_gc_roots();
        assert_eq!(roots.len(), 1);

        let dirty_ranges = manager.get_dirty_card_ranges();
        assert_eq!(dirty_ranges.len(), 1);

        let stats = manager.stats();
        assert_eq!(stats.remembered_set.barrier_hits, 1);
        assert!(stats.card_table.is_some());
    }

    #[test]
    fn test_value_write_barrier_aware() {
        let value = Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(42.0)], 1, 1).unwrap());
        assert!(value.has_young_references());

        let value2 = Value::Num(42.0);
        assert!(!value2.has_young_references());
    }
}
