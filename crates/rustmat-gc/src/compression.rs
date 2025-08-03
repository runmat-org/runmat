//! Pointer compression for memory efficiency
//!
//! Implements compressed pointers to reduce memory overhead on 64-bit platforms.
//! Pointers are compressed to 32 bits by using a base address and offset encoding.

use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Global heap base address for pointer compression
static HEAP_BASE: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

/// Maximum compressed heap size (4GB with 32-bit offsets)
pub const MAX_COMPRESSED_HEAP_SIZE: usize = 4 * 1024 * 1024 * 1024; // 4GB

/// A compressed pointer that uses 32 bits instead of 64 bits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompressedPtr {
    /// 32-bit offset from heap base
    offset: u32,
}

impl CompressedPtr {
    /// Create a null compressed pointer
    pub fn null() -> Self {
        Self { offset: 0 }
    }

    /// Check if this pointer is null
    pub fn is_null(&self) -> bool {
        self.offset == 0
    }

    /// Compress a raw pointer
    pub fn compress(ptr: *const u8) -> Self {
        if ptr.is_null() {
            return Self::null();
        }

        let heap_base = HEAP_BASE.load(Ordering::Relaxed);
        if heap_base.is_null() {
            // Initialize heap base on first use
            initialize_heap_base(ptr);
            let heap_base = HEAP_BASE.load(Ordering::Relaxed);
            if heap_base.is_null() {
                log::warn!("Failed to initialize heap base for pointer compression");
                return Self::null();
            }
        }

        let base_addr = heap_base as usize;
        let ptr_addr = ptr as usize;

        if ptr_addr < base_addr {
            log::warn!("Pointer {:p} is below heap base {:p}", ptr, heap_base);
            return Self::null();
        }

        let offset = ptr_addr - base_addr;
        if offset > u32::MAX as usize {
            log::warn!("Pointer {:p} offset {} exceeds 32-bit range", ptr, offset);
            return Self::null();
        }

        Self {
            offset: offset as u32,
        }
    }

    /// Decompress to a raw pointer
    pub fn decompress(&self) -> *const u8 {
        if self.is_null() {
            return ptr::null();
        }

        let heap_base = HEAP_BASE.load(Ordering::Relaxed);
        if heap_base.is_null() {
            log::error!("Heap base not initialized for decompression");
            return ptr::null();
        }

        let base_addr = heap_base as usize;
        let ptr_addr = base_addr + (self.offset as usize);

        ptr_addr as *const u8
    }

    /// Get the raw offset value
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Create a compressed pointer from an offset (unsafe)
    ///
    /// # Safety
    ///
    /// The caller must ensure the offset is valid within the compressed heap
    pub unsafe fn from_offset(offset: u32) -> Self {
        Self { offset }
    }

    /// Add an offset to this compressed pointer
    pub fn add_offset(&self, additional_offset: usize) -> Option<Self> {
        let new_offset = self.offset as usize + additional_offset;
        if new_offset <= u32::MAX as usize {
            Some(Self {
                offset: new_offset as u32,
            })
        } else {
            None
        }
    }

    /// Calculate the distance between two compressed pointers
    pub fn offset_from(&self, other: &CompressedPtr) -> Option<isize> {
        if self.offset >= other.offset {
            Some((self.offset - other.offset) as isize)
        } else {
            Some(-((other.offset - self.offset) as isize))
        }
    }
}

impl Default for CompressedPtr {
    fn default() -> Self {
        Self::null()
    }
}

/// Initialize the heap base address for compression
fn initialize_heap_base(first_ptr: *const u8) {
    // Use the first allocated pointer as the heap base, aligned down to a page boundary
    let page_size = page_size::get();
    let aligned_base = (first_ptr as usize) & !(page_size - 1);

    let base_ptr = aligned_base as *mut u8;

    // Try to set the heap base atomically
    match HEAP_BASE.compare_exchange(
        ptr::null_mut(),
        base_ptr,
        Ordering::Relaxed,
        Ordering::Relaxed,
    ) {
        Ok(_) => {
            log::info!(
                "Initialized pointer compression heap base at {:p}",
                base_ptr
            );
        }
        Err(existing) => {
            log::debug!(
                "Heap base already initialized at {:p}, first ptr was {:p}",
                existing,
                first_ptr
            );
        }
    }
}

/// Get the current heap base address
pub fn get_heap_base() -> *const u8 {
    HEAP_BASE.load(Ordering::Relaxed) as *const u8
}

/// Set the heap base address (for testing)
#[cfg(test)]
pub fn set_heap_base_for_test(base: *mut u8) {
    HEAP_BASE.store(base, Ordering::Relaxed);
}

/// Reset the heap base (for testing)
#[cfg(test)]
pub fn reset_heap_base_for_test() {
    HEAP_BASE.store(ptr::null_mut(), Ordering::Relaxed);
}

/// Check if pointer compression is beneficial
pub fn is_compression_beneficial() -> bool {
    // Compression is beneficial on 64-bit platforms where we can save 4 bytes per pointer
    std::mem::size_of::<*const u8>() == 8
}

/// Calculate memory savings from compression
pub fn calculate_compression_savings(num_pointers: usize) -> usize {
    if is_compression_beneficial() {
        num_pointers * 4 // Save 4 bytes per pointer
    } else {
        0
    }
}

/// Statistics for pointer compression
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub heap_base: *const u8,
    pub max_offset: u32,
    pub compression_ratio: f64,
    pub memory_saved_bytes: usize,
    pub pointers_compressed: usize,
}

impl CompressionStats {
    pub fn new() -> Self {
        Self {
            heap_base: get_heap_base(),
            max_offset: 0,
            compression_ratio: 0.5, // 32-bit vs 64-bit
            memory_saved_bytes: 0,
            pointers_compressed: 0,
        }
    }

    pub fn record_compression(&mut self, _ptr: *const u8) {
        self.pointers_compressed += 1;
        self.memory_saved_bytes = calculate_compression_savings(self.pointers_compressed);
    }

    /// Check if compression is working effectively
    pub fn is_effective(&self) -> bool {
        !self.heap_base.is_null() && self.memory_saved_bytes > 0
    }

    /// Get compression efficiency as a percentage
    pub fn efficiency_percentage(&self) -> f64 {
        if self.pointers_compressed == 0 {
            0.0
        } else {
            self.compression_ratio * 100.0
        }
    }
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed pointer array for efficient storage of many pointers
pub struct CompressedPtrArray {
    /// Compressed pointers
    pointers: Vec<CompressedPtr>,

    /// Statistics
    stats: CompressionStats,
}

impl CompressedPtrArray {
    pub fn new() -> Self {
        Self {
            pointers: Vec::new(),
            stats: CompressionStats::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pointers: Vec::with_capacity(capacity),
            stats: CompressionStats::new(),
        }
    }

    /// Add a pointer to the array
    pub fn push(&mut self, ptr: *const u8) {
        let compressed = CompressedPtr::compress(ptr);
        self.pointers.push(compressed);
        self.stats.record_compression(ptr);
    }

    /// Get a pointer from the array
    pub fn get(&self, index: usize) -> Option<*const u8> {
        self.pointers.get(index).map(|cp| cp.decompress())
    }

    /// Get the compressed pointer directly
    pub fn get_compressed(&self, index: usize) -> Option<CompressedPtr> {
        self.pointers.get(index).copied()
    }

    /// Length of the array
    pub fn len(&self) -> usize {
        self.pointers.len()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    /// Clear the array
    pub fn clear(&mut self) {
        self.pointers.clear();
        self.stats = CompressionStats::new();
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Calculate memory usage compared to uncompressed
    pub fn memory_usage(&self) -> (usize, usize) {
        let compressed_size = self.pointers.len() * std::mem::size_of::<CompressedPtr>();
        let uncompressed_size = self.pointers.len() * std::mem::size_of::<*const u8>();
        (compressed_size, uncompressed_size)
    }
}

impl Default for CompressedPtrArray {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_ptr_null() {
        let null_ptr = CompressedPtr::null();
        assert!(null_ptr.is_null());
        assert_eq!(null_ptr.decompress(), ptr::null());
    }

    #[test]
    fn test_compressed_ptr_basic() {
        reset_heap_base_for_test();

        // Create a test pointer
        let test_data = [1, 2, 3, 4];
        let ptr = test_data.as_ptr();

        // Compress and decompress
        let compressed = CompressedPtr::compress(ptr);
        assert!(!compressed.is_null());

        let decompressed = compressed.decompress();
        assert_eq!(ptr, decompressed);
    }

    #[test]
    fn test_compressed_ptr_offset() {
        reset_heap_base_for_test();

        let test_data = [1u8; 1000];
        let base_ptr = test_data.as_ptr();
        let offset_ptr = unsafe { base_ptr.add(100) };

        // Set heap base manually for test
        set_heap_base_for_test(base_ptr as *mut u8);

        let compressed = CompressedPtr::compress(offset_ptr);
        assert!(!compressed.is_null());
        assert_eq!(compressed.offset(), 100);

        let decompressed = compressed.decompress();
        assert_eq!(offset_ptr, decompressed);
    }

    #[test]
    fn test_compressed_ptr_array() {
        reset_heap_base_for_test();

        let mut array = CompressedPtrArray::new();
        assert!(array.is_empty());

        let test_data = [1u8, 2, 3, 4, 5];
        for (i, item) in test_data.iter().enumerate() {
            array.push(item as *const u8);
            assert_eq!(array.len(), i + 1);
        }

        // Verify we can retrieve the pointers
        for (i, expected) in test_data.iter().enumerate() {
            let retrieved = array.get(i).expect("should have pointer");
            assert_eq!(retrieved, expected as *const u8);
        }

        let stats = array.stats();
        assert_eq!(stats.pointers_compressed, 5);

        let (compressed_size, uncompressed_size) = array.memory_usage();
        assert!(compressed_size <= uncompressed_size);
    }

    #[test]
    fn test_compression_savings() {
        let savings = calculate_compression_savings(1000);
        if is_compression_beneficial() {
            assert_eq!(savings, 4000); // 1000 pointers * 4 bytes saved each
        } else {
            assert_eq!(savings, 0);
        }
    }

    #[test]
    fn test_compressed_ptr_add_offset() {
        let ptr = CompressedPtr { offset: 100 };

        let new_ptr = ptr.add_offset(50).expect("should add offset");
        assert_eq!(new_ptr.offset(), 150);

        // Test overflow
        let large_ptr = CompressedPtr {
            offset: u32::MAX - 10,
        };
        assert!(large_ptr.add_offset(20).is_none());
    }

    #[test]
    fn test_compressed_ptr_offset_from() {
        let ptr1 = CompressedPtr { offset: 100 };
        let ptr2 = CompressedPtr { offset: 150 };

        let diff = ptr2.offset_from(&ptr1).expect("should calculate offset");
        assert_eq!(diff, 50);

        let diff_rev = ptr1.offset_from(&ptr2).expect("should calculate offset");
        assert_eq!(diff_rev, -50);
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::new();
        assert_eq!(stats.pointers_compressed, 0);
        assert_eq!(stats.memory_saved_bytes, 0);

        let dummy_ptr = 0x1000 as *const u8;
        stats.record_compression(dummy_ptr);

        assert_eq!(stats.pointers_compressed, 1);
        if is_compression_beneficial() {
            assert_eq!(stats.memory_saved_bytes, 4);
        }
    }
}
