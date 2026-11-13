//! Checked accounting for heap allocations created by immutable plot snapshots.

pub(super) fn add(total: &mut usize, bytes: usize) -> Option<()> {
    *total = total.checked_add(bytes)?;
    Some(())
}

pub(super) fn vec<T>(capacity: usize) -> Option<usize> {
    capacity.checked_mul(std::mem::size_of::<T>())
}

pub(super) fn arc_slice<T>(len: usize) -> Option<usize> {
    let alignment = std::mem::align_of::<T>().max(std::mem::align_of::<usize>());
    let header_and_padding =
        (2 * std::mem::size_of::<usize>()).checked_add(alignment.saturating_sub(1))?;
    header_and_padding.checked_add(vec::<T>(len)?)
}

pub(super) fn string(value: &String) -> usize {
    value.capacity()
}

pub(super) fn optional_string(value: Option<&String>) -> usize {
    value.map_or(0, String::capacity)
}

pub(super) fn strings(values: &Vec<String>) -> Option<usize> {
    let mut total = vec::<String>(values.capacity())?;
    for value in values {
        add(&mut total, string(value))?;
    }
    Some(total)
}

pub(super) fn nested_vec<T>(values: &Vec<Vec<T>>) -> Option<usize> {
    let mut total = vec::<Vec<T>>(values.capacity())?;
    for value in values {
        add(&mut total, vec::<T>(value.capacity())?)?;
    }
    Some(total)
}

pub(super) fn owned_axis(value: &crate::gpu::axis::OwnedAxisData) -> Option<usize> {
    match value {
        crate::gpu::axis::OwnedAxisData::F32(values) => vec::<f32>(values.capacity()),
        crate::gpu::axis::OwnedAxisData::F64(values) => vec::<f64>(values.capacity()),
        crate::gpu::axis::OwnedAxisData::Buffer(_) => Some(0),
    }
}
