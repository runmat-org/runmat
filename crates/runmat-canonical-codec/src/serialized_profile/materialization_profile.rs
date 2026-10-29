//! Borrowed exact-value and value-independent shape materialization profiles.
use super::*;

/// Projects canonical materialization residency directly from a borrowed Serde value.
///
/// The traversal itself allocates no value tree, output buffer, or owned strings. Custom
/// `Serialize`/`Display` implementations (including Serde-generated buffering for some advanced
/// representations) may allocate independently. Errors allocate their diagnostic message.
/// Counters, per-container lengths, string lengths, and nesting are checked during traversal.
/// Encoded bytes reserve the largest valid canonical envelope and the encoder's actual scalar/header widths (unknown-length collections retain a nine-byte header bound),
/// so this can conservatively reject values that a particular smaller envelope would accept.
/// Nesting also counts transparent `Some`/newtype wrappers, bounding traversal stack depth even
/// though those wrappers do not add nodes to the canonical value tree.
/// Nonfinite numbers, integers outside the JSON number range, and non-scalar map keys fail closed.
/// This is the shared materialization estimate, not an exact typed-object heap measurement.
pub fn serialized_materialization_resident_bytes_estimate<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<u64, CanonicalCodecError> {
    Ok(profile_serialized(value, limits)?.materialization_resident_bytes_estimate)
}

/// Cooperative materialization profiling over the same shared Serde traversal.
pub fn profile_serialized_with_control<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
    control: &dyn crate::CanonicalCodecControl,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    profile_serialized(&crate::controlled_serde::tracked(value, control), limits)
}

/// Profiles both canonical wire size and materialization without constructing either.
///
/// Uses the same traversal, conservative envelope, validation and custom-serializer caveats as
/// [`serialized_materialization_resident_bytes_estimate`]. The wire bound includes one contract
/// envelope; separately encoded partitions must each be profiled with their own envelope.
pub fn profile_serialized<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    profile(value, limits, false)
}

/// Profiles a fixed serialization shape before integer values are known.
///
/// Integer values reserve their entire Serde type range; integer map keys reserve
/// twenty decimal characters (forty for 128-bit key types). Floating values retain
/// their nine-byte bound. Strings,
/// collection counts and enum/optional variants remain supplied by the caller, which
/// must choose their complete domain envelope. Exact encoding bytes are unchanged.
/// Uses the same validation, cancellation-adapter support and custom-serializer
/// caveats as [`profile_serialized`]. No value tree or numerical array is allocated.
pub fn profile_serialized_shape<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    profile(value, limits, true)
}

/// Resident estimate of an owned typed value from its serialization shape: eight bytes per
/// scalar, a collection header per array or string, string bytes, and inline structs, with the
/// fixed per-contract allowance. This prices typed storage that already exists or is decoded
/// directly; it builds neither a value tree nor an encoding.
pub fn profile_typed_resident_bytes<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<u64, CanonicalCodecError> {
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        encoded_bytes: 0,
        limits,
        live_map_metadata: 0,
        peak_map_metadata: 0,
        integer_value_bounds: false,
        child_peak: 0,
    };
    counter.encoded(ENVELOPE_BYTES)?;
    value.serialize(Profiler::value(&mut counter, 0))?;
    crate::materialization::typed_resident_bytes_estimate(counter.profile)
}

/// Direct-codec materialization of an owned typed value: the typed storage the direct decoder
/// builds (or the producer already holds), priced from the serialization shape exactly as
/// [`crate::canonical_payload_typed_resident_bytes_estimate`] prices the resulting wire, plus
/// its canonical bytes and the direct encoder's peak workspace (reserved output, held child
/// buffers, live sorted-map metadata). The direct encoder and decoder build no value tree, so
/// owners of typed values charge this, not the value-tree [`profile_serialized`] estimate,
/// which costs tens of bytes per scalar (about 80 bytes per element of an embedded byte vector).
pub fn profile_direct_materialization<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        encoded_bytes: 0,
        limits,
        live_map_metadata: 0,
        peak_map_metadata: 0,
        integer_value_bounds: false,
        child_peak: 0,
    };
    counter.encoded(ENVELOPE_BYTES)?;
    value.serialize(Profiler::value(&mut counter, 0))?;
    let overflow = || {
        CanonicalCodecError::resource_limit("direct materialization profile", "byte count overflow")
    };
    let typed = crate::materialization::typed_resident_bytes_estimate(counter.profile)?;
    let encoding = counter
        .encoded_bytes
        .checked_add(counter.child_peak)
        .and_then(|bytes| bytes.checked_add(counter.peak_map_metadata))
        .ok_or_else(overflow)?;
    let materialization = typed
        .checked_add(counter.encoded_bytes)
        .and_then(|bytes| bytes.checked_add(encoding))
        .ok_or_else(overflow)?;
    Ok(SerializedCanonicalProfile {
        encoded_bytes_upper_bound: counter.encoded_bytes,
        materialization_resident_bytes_estimate: materialization,
    })
}

fn profile<T: Serialize + ?Sized>(
    value: &T,
    limits: CanonicalLimits,
    integer_value_bounds: bool,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    let mut counter = Counter {
        profile: MaterializationProfile::default(),
        encoded_bytes: 0,
        limits,
        live_map_metadata: 0,
        peak_map_metadata: 0,
        integer_value_bounds,
        child_peak: 0,
    };
    counter.encoded(ENVELOPE_BYTES)?;
    value.serialize(Profiler::value(&mut counter, 0))?;
    Ok(SerializedCanonicalProfile {
        encoded_bytes_upper_bound: counter.encoded_bytes,
        materialization_resident_bytes_estimate: materialization_resident_bytes_estimate(
            counter.profile,
            counter.encoded_bytes,
        )?,
    })
}
