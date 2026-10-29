//! Allocation-free counting and iteration over the original ordered time sources.
use super::TimeScheduleError as Error;

pub(super) struct OrderedUnion<'a> {
    sources: [&'a [f64]; 3],
    positions: [usize; 3],
    /// Coordinates closer than this are one time: a uniform grid point computed as
    /// start + k·Δt and the authored time it stands for (0.1 s against 0.09999999999999999 s)
    /// must not leave a rounding-width interval between them.
    tolerance: f64,
}
impl<'a> OrderedUnion<'a> {
    pub(super) fn new(sources: [&'a [f64]; 3]) -> Self {
        let scale = sources
            .iter()
            .flat_map(|s| s.first().into_iter().chain(s.last()))
            .fold(0.0f64, |m, v| m.max(v.abs()));
        Self {
            sources,
            positions: [0; 3],
            tolerance: 1e-12 * scale,
        }
    }
}
impl Iterator for OrderedUnion<'_> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        let value = self
            .sources
            .iter()
            .zip(self.positions)
            .filter_map(|(s, p)| s.get(p))
            .copied()
            .min_by(f64::total_cmp)?;
        // One coordinate per cluster; an authored event or required time (later sources)
        // names it over a computed grid point.
        let mut chosen = value;
        for (index, (source, position)) in self.sources.iter().zip(&mut self.positions).enumerate()
        {
            while let Some(v) = source.get(*position).copied() {
                if v - value > self.tolerance {
                    break;
                }
                if index > 0 {
                    chosen = v;
                }
                *position += 1;
            }
        }
        Some(if chosen == 0.0 { 0.0 } else { chosen })
    }
}

pub(super) fn counts(
    grid: &[f64],
    events: &[f64],
    required: &[f64],
    maximum: usize,
) -> Result<(usize, usize), Error> {
    if grid.len() < 2
        || grid.iter().any(|t| !t.is_finite())
        || grid.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(Error::Invalid(
            "time grid must be finite and strictly increasing",
        ));
    }
    for source in [events, required] {
        if source
            .iter()
            .any(|t| !t.is_finite() || *t < grid[0] || *t > grid[grid.len() - 1])
            || source.windows(2).any(|p| p[0] > p[1])
        {
            return Err(Error::Invalid(
                "time schedule source must be ordered inside the authored interval",
            ));
        }
    }
    let points = count(OrderedUnion::new([grid, events, required]), maximum)?;
    let events = count(OrderedUnion::new([events, &[], &[]]), maximum)?;
    Ok((points, events))
}
fn count(values: OrderedUnion<'_>, maximum: usize) -> Result<usize, Error> {
    let mut count = 0usize;
    for _ in values {
        if count.is_multiple_of(1024) && crate::is_fea_cancelled() {
            return Err(Error::Invalid("time schedule construction cancelled"));
        }
        count = count
            .checked_add(1)
            .ok_or(Error::Invalid("time schedule count overflow"))?;
        if count > maximum {
            return Err(Error::Ceiling {
                required: count,
                maximum,
            });
        }
    }
    Ok(count)
}
pub(super) fn reserve(count: usize) -> Result<Vec<f64>, Error> {
    crate::topology::storage::reserve(count)
        .map_err(|_| Error::Invalid("time schedule storage unavailable or cancelled"))
}
pub(super) fn resident_bytes(points: usize, events: usize) -> Result<u64, Error> {
    use crate::topology::storage::{add, slots};
    slots::<f64>(points)
        .and_then(|p| slots::<f64>(events).and_then(|e| add(p, e)))
        .map_err(|_| Error::Invalid("time schedule storage exceeds addressable slots"))
}
