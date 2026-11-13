//! Measured encoder peak against the encoding workspace profile, in its own test binary.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    collections::BTreeMap,
};

use runmat_canonical_codec::{encode_contract, profile_canonical_encoding, CanonicalLimits};
use serde::{ser::SerializeMap, Serialize};

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

struct PeakAllocator;
#[global_allocator]
static ALLOCATOR: PeakAllocator = PeakAllocator;

fn record(delta: i64) {
    let _ = TRACKING.try_with(|enabled| {
        if enabled.get() {
            let _ = LIVE.try_with(|live| {
                live.set(live.get() + delta);
                let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
            });
        }
    });
}

unsafe impl GlobalAlloc for PeakAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size() as i64);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size() as i64);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // Count the transient copy a moving reallocation may hold.
        record(size as i64);
        let moved = unsafe { System.realloc(pointer, layout, size) };
        record(-(layout.size() as i64));
        moved
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(-(layout.size() as i64));
        unsafe { System.dealloc(pointer, layout) }
    }
}

const LIMITS: CanonicalLimits = CanonicalLimits::new(256 << 20, 20_000_000, 1 << 20, 64);

fn measured_peak<T: Serialize>(value: &T) -> (u64, u64, usize) {
    let profile = profile_canonical_encoding(value, LIMITS).unwrap();
    LIVE.with(|live| live.set(0));
    PEAK.with(|peak| peak.set(0));
    TRACKING.with(|enabled| enabled.set(true));
    let bytes = encode_contract(b"peak\0", "workspace", value, LIMITS).unwrap();
    TRACKING.with(|enabled| enabled.set(false));
    let peak = PEAK.with(Cell::get) as u64;
    (peak, profile.resident_workspace_bytes_estimate, bytes.len())
}

/// Columns declared in canonical order stream straight into the reserved output.
#[derive(Serialize)]
struct Canonical {
    a_ids: Vec<u64>,
    b_points: Vec<[f64; 3]>,
    c_rows: Vec<Vec<u32>>,
}

/// Columns declared out of canonical order: early fields are held.
#[derive(Serialize)]
struct Declared {
    zeta: Vec<f64>,
    schema_version: u16,
    alpha: Vec<u64>,
    nested: Vec<Canonical>,
}

struct UnknownLength(Vec<u32>);
impl Serialize for UnknownLength {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().filter(|_| true))
    }
}

struct Unordered(Vec<(String, Vec<u16>)>);
impl Serialize for Unordered {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

fn canonical(count: usize) -> Canonical {
    Canonical {
        a_ids: (0..count as u64).collect(),
        b_points: (0..count).map(|i| [i as f64, 0.5, -1.0]).collect(),
        c_rows: (0..count).map(|i| vec![i as u32; 4]).collect(),
    }
}

#[test]
fn encoder_peak_stays_within_the_profiled_workspace() {
    let streamed = canonical(200_000);
    let (peak, profile, wire) = measured_peak(&streamed);
    assert!(
        peak <= profile,
        "streamed peak {peak} exceeds profile {profile}"
    );
    // Streaming holds no second copy of the output.
    assert!(
        peak < 2 * wire as u64,
        "streamed peak {peak} for {wire} wire bytes"
    );

    let declared = Declared {
        zeta: vec![0.25; 100_000],
        schema_version: 3,
        alpha: (0..100_000).collect(),
        nested: vec![canonical(1_000), canonical(10)],
    };
    let (peak, profile, _) = measured_peak(&declared);
    assert!(
        peak <= profile,
        "held-field peak {peak} exceeds profile {profile}"
    );

    let unknown = vec![UnknownLength((0..50_000).collect()); 1];
    let (peak, profile, _) = measured_peak(&unknown.iter().collect::<Vec<_>>());
    assert!(
        peak <= profile,
        "unknown-length peak {peak} exceeds profile {profile}"
    );

    let unordered = Unordered(
        (0..2_000)
            .rev()
            .map(|i| (format!("key{i:05}"), vec![i as u16; 32]))
            .collect(),
    );
    let (peak, profile, _) = measured_peak(&unordered);
    assert!(
        peak <= profile,
        "fallback map peak {peak} exceeds profile {profile}"
    );

    let ordered = BTreeMap::from_iter((0..2_000).map(|i| (format!("key{i:05}"), vec![i; 32])));
    let (peak, profile, _) = measured_peak(&ordered);
    assert!(
        peak <= profile,
        "ordered map peak {peak} exceeds profile {profile}"
    );
}

impl Clone for UnknownLength {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
