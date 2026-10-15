//! Original admitted sum geometry, not provider-effective arithmetic or timing.
use super::SumInvocation;
use crate::{NumericClass, NumericDomain};

pub struct SumWorkControls<'a> {
    pub shape: &'a [usize],
    pub axes: &'a [u64],
    pub input_class: NumericClass,
    pub domain: NumericDomain,
    pub accumulation_class: NumericClass,
    pub publication_class: NumericClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumWorkGeometry {
    /// Same scalar-lane scan units as the existing reduction calibration.
    pub scan_lanes: u64,
    /// Separate cold publication/conversion stages, not calibrated scan units.
    pub publication_conversion_stages: u64,
    pub output_elements: u64,
}

impl SumInvocation {
    pub fn original_work_geometry(self, controls: SumWorkControls<'_>) -> Option<SumWorkGeometry> {
        let SumWorkControls {
            shape,
            axes,
            input_class,
            domain,
            accumulation_class,
            publication_class,
        } = controls;
        // Native integer accumulation is a real, separately admitted sum form.
        // Complex-integer arithmetic is not an admitted host sum form.
        let native_integer = input_class.integer_class().is_some()
            && accumulation_class == input_class
            && publication_class == input_class;
        if (domain == NumericDomain::Complex && input_class.integer_class().is_some())
            || (!native_integer && accumulation_class != NumericClass::Double)
            || (!native_integer
                && !matches!(
                    publication_class,
                    NumericClass::Single | NumericClass::Double
                ))
        {
            return None;
        }
        let mut reduced = vec![false; shape.len()];
        for &axis in axes {
            let axis = usize::try_from(axis).ok()?;
            if axis >= reduced.len() || reduced[axis] {
                return None;
            }
            reduced[axis] = true;
        }
        let output_shape = shape
            .iter()
            .zip(reduced)
            .map(|(&n, r)| if r { 1 } else { n })
            .collect::<Vec<_>>();
        let inputs = elements(shape)?;
        let outputs = elements(&output_shape)?;
        let lanes = if domain == NumericDomain::Complex {
            2
        } else {
            1
        };
        let scan_lanes = inputs.checked_mul(lanes)?;
        let output_lanes = outputs.checked_mul(lanes)?;
        let input_conversions = if input_class != accumulation_class {
            scan_lanes
        } else {
            0
        };
        let output_conversions = if accumulation_class != publication_class {
            output_lanes
        } else {
            0
        };
        Some(SumWorkGeometry {
            scan_lanes,
            publication_conversion_stages: output_lanes
                .checked_add(input_conversions)?
                .checked_add(output_conversions)?,
            output_elements: outputs,
        })
    }
}

fn elements(shape: &[usize]) -> Option<u64> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape.iter().try_fold(1_u64, |n, &extent| {
        n.checked_mul(u64::try_from(extent).ok()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_integer_complex_and_empty_publication_geometry() {
        for (class, domain, accumulator, publication, lanes) in [
            (
                NumericClass::Single,
                NumericDomain::Real,
                NumericClass::Double,
                NumericClass::Single,
                1,
            ),
            (
                NumericClass::Int64,
                NumericDomain::Real,
                NumericClass::Int64,
                NumericClass::Int64,
                1,
            ),
            (
                NumericClass::Double,
                NumericDomain::Complex,
                NumericClass::Double,
                NumericClass::Double,
                2,
            ),
        ] {
            let work = SumInvocation::Sum
                .original_work_geometry(SumWorkControls {
                    shape: &[0, 7],
                    axes: &[0],
                    input_class: class,
                    domain,
                    accumulation_class: accumulator,
                    publication_class: publication,
                })
                .unwrap();
            assert_eq!(work.scan_lanes, 0);
            assert_eq!(work.output_elements, 7);
            assert!(work.publication_conversion_stages >= 7 * lanes);
        }
        assert!(SumInvocation::NanSum
            .original_work_geometry(SumWorkControls {
                shape: &[3, 2],
                axes: &[0, 0],
                input_class: NumericClass::Double,
                domain: NumericDomain::Real,
                accumulation_class: NumericClass::Double,
                publication_class: NumericClass::Double,
            })
            .is_none());
    }
}
