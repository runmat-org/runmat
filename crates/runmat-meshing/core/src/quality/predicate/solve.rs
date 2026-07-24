use crate::quality::tolerance::MeshingTolerance;

use super::types::Point3;

pub fn solve_3x3(
    mut a: [[f64; 3]; 3],
    mut b: [f64; 3],
    tolerance: MeshingTolerance,
) -> Option<Point3> {
    for pivot in 0..3 {
        let mut pivot_row = pivot;
        for row in (pivot + 1)..3 {
            if a[row][pivot].abs() > a[pivot_row][pivot].abs() {
                pivot_row = row;
            }
        }
        if a[pivot_row][pivot].abs() <= tolerance.absolute_m {
            return None;
        }
        if pivot_row != pivot {
            a.swap(pivot, pivot_row);
            b.swap(pivot, pivot_row);
        }
        let pivot_value = a[pivot][pivot];
        for column in pivot..3 {
            a[pivot][column] /= pivot_value;
        }
        b[pivot] /= pivot_value;
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for column in pivot..3 {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}
