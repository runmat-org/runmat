use std::collections::BTreeMap;

use super::*;
use crate::planning::resource::LocalThreadCount;

fn plan(threads: u16) -> LocalExecutionPlan {
    LocalExecutionPlan::from_thread_count(LocalThreadCount::new(threads).unwrap()).unwrap()
}

/// Deterministic pseudo-random stream (SplitMix64).
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Values spanning many magnitudes so that any change of summation order changes the bits.
fn value(seed: u64) -> f64 {
    let bits = mix(seed);
    let mantissa = (bits >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
    mantissa * 10f64.powi((bits % 17) as i32 - 8)
}

/// Mesh-like map: elements of 4..=10 local DOFs drawn from a sliding window, with occasional
/// duplicate and skipped local DOFs.
fn test_map(dof_count: usize, element_count: usize) -> ElementDofMap {
    let mut offsets = vec![0];
    let mut dofs = Vec::new();
    for element in 0..element_count {
        let n = 4 + (mix(element as u64) % 7) as usize;
        let base = (mix(element as u64 + 99) as usize) % dof_count;
        for local in 0..n {
            let r = mix((element * 31 + local) as u64);
            let dof = if r % 23 == 0 {
                SKIPPED_DOF
            } else if r % 29 == 0 && local > 0 {
                dofs[dofs.len() - 1]
            } else {
                ((base + (r as usize % 40)) % dof_count) as u32
            };
            dofs.push(dof);
        }
        offsets.push(dofs.len());
    }
    ElementDofMap::new(dof_count, offsets, dofs).unwrap()
}

fn kernel(element: usize, dofs: &[u32], out: &mut ElementOutputs<'_, f64>) -> Result<(), String> {
    let n = dofs.len();
    for matrix in 0..2 {
        let local = out.matrix(matrix);
        for i in 0..n {
            for j in 0..n {
                local[i * n + j] = value((element * 7919 + matrix * 104_729 + i * n + j) as u64);
            }
        }
    }
    for (i, slot) in out.vector(0).iter_mut().enumerate() {
        *slot = value((element * 13 + i) as u64 + 1_000_000_007);
    }
    *out.scalar(0) = value(element as u64 + 77);
    Ok(())
}

/// Serial reference: element loop, rows outer, columns inner, `+=` into ordered maps.
fn reference(map: &ElementDofMap) -> (Vec<BTreeMap<usize, f64>>, Vec<BTreeMap<usize, f64>>, Vec<f64>, f64) {
    let n_dofs = map.dof_count();
    let mut a = vec![BTreeMap::new(); n_dofs];
    let mut b = vec![BTreeMap::new(); n_dofs];
    let mut vector = vec![0.0; n_dofs];
    let mut scalar = 0.0;
    for element in 0..map.element_count() {
        let dofs = map.element(element);
        let n = dofs.len();
        let mut values = vec![0.0; 2 * n * n + n + 1];
        let shape = ElementOutputShape {
            matrices: 2,
            vectors: 1,
            scalars: 1,
        };
        kernel(element, dofs, &mut ElementOutputs { n, shape, values: &mut values }).unwrap();
        for i in 0..n {
            if dofs[i] == SKIPPED_DOF {
                continue;
            }
            let row = dofs[i] as usize;
            for j in 0..n {
                if dofs[j] == SKIPPED_DOF {
                    continue;
                }
                let column = dofs[j] as usize;
                *a[row].entry(column).or_insert(0.0) += values[i * n + j];
                *b[row].entry(column).or_insert(0.0) += values[n * n + i * n + j];
            }
            vector[row] += values[2 * n * n + i];
        }
        scalar += values[2 * n * n + n];
    }
    (a, b, vector, scalar)
}

fn to_csr_values(pattern: &SparsePattern, rows: &[BTreeMap<usize, f64>]) -> Vec<f64> {
    let mut values = Vec::new();
    for (row, entries) in rows.iter().enumerate() {
        let columns = &pattern.column_indices()[pattern.row_range(row)];
        assert_eq!(
            columns.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            entries.keys().copied().collect::<Vec<_>>(),
            "row {row} pattern"
        );
        values.extend(entries.values().copied());
    }
    values
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

// witness: FEA-SCALE-PARALLEL-001/shared_parallel_kernels @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: Element assembly is bit-identical to the serial loop at 1, 2, 3, 7 and 16 threads on
//   a synthetic 3,000-element map; no family operator is assembled here.
// witness: FEA-SHARED-EXECUTION-001/bounded_shared_memory_assembly @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: Parallel element assembly into partition-local buffers merges to a bit-identical
//   result for every thread count; memory ceilings are not asserted.
// witness: FEA-SHARED-EXECUTION-001/serial_reference_and_equivalence_policy @
//   shared_infrastructure/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: Parallel assembly equals the serial reference bit for bit; the policy for remote and
//   distributed equivalence is not exercised.
#[test]
fn assembly_is_bitwise_identical_to_the_serial_element_loop_for_every_thread_count() {
    let map = test_map(3_000, 9_000);
    let (a, b, vector, scalar) = reference(&map);
    let shape = ElementOutputShape {
        matrices: 2,
        vectors: 1,
        scalars: 1,
    };
    for threads in [1, 2, 3, 7, 16] {
        let pattern = map.symbolic_pattern(plan(threads)).unwrap();
        let expected_a = to_csr_values(&pattern, &a);
        let expected_b = to_csr_values(&pattern, &b);
        let assembled =
            assemble_elements(&map, &pattern, shape, plan(threads), kernel, no_visit).unwrap();
        assert_eq!(bits(&assembled.matrices[0]), bits(&expected_a), "threads {threads}");
        assert_eq!(bits(&assembled.matrices[1]), bits(&expected_b), "threads {threads}");
        assert_eq!(bits(&assembled.vectors[0]), bits(&vector), "threads {threads}");
        assert_eq!(assembled.scalars[0].to_bits(), scalar.to_bits());
    }
}

#[test]
fn parallel_map_construction_preserves_element_order() {
    let reference = test_map(500, 3_000);
    for threads in [1, 4] {
        let built = ElementDofMap::from_elements(500, 3_000, plan(threads), |element, out| {
            out.extend(
                reference
                    .element(element)
                    .iter()
                    .map(|&dof| (dof != SKIPPED_DOF).then_some(dof as usize)),
            );
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(built, reference);
    }
}

#[test]
fn kernel_failures_report_the_lowest_failing_element_for_every_thread_count() {
    let map = test_map(800, 5_000);
    let pattern = map.symbolic_pattern(plan(1)).unwrap();
    for threads in [1, 3, 8] {
        let error = assemble_elements(
            &map,
            &pattern,
            ElementOutputShape {
                matrices: 1,
                ..Default::default()
            },
            plan(threads),
            |element, _, _: &mut ElementOutputs<'_, f64>| {
                if element == 1_234 || element == 4_321 || element == 1_500 {
                    Err(element)
                } else {
                    Ok(())
                }
            },
            no_visit,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ElementAssemblyError::Element {
                element: 1_234,
                error: 1_234
            }
        );
    }
}

#[test]
fn ordered_visitor_sees_every_element_in_canonical_order() {
    let map = test_map(400, 2_000);
    let pattern = map.symbolic_pattern(plan(4)).unwrap();
    let mut seen = Vec::new();
    assemble_elements(
        &map,
        &pattern,
        ElementOutputShape {
            matrices: 1,
            vectors: 0,
            scalars: 1,
        },
        plan(4),
        |element, _, out: &mut ElementOutputs<'_, f64>| {
            *out.scalar(0) = element as f64;
            Ok::<_, ()>(())
        },
        |view: &ElementView<'_, f64>| {
            assert_eq!(*view.scalar(0), view.element as f64);
            seen.push(view.element);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, (0..2_000).collect::<Vec<_>>());
}

#[test]
fn entries_outside_a_foreign_pattern_fail_closed() {
    let map = test_map(300, 600);
    let diagonal = SparsePattern::try_new(
        300,
        300,
        (0..=300).collect(),
        (0..300).collect(),
    )
    .unwrap();
    let error = assemble_elements(
        &map,
        &diagonal,
        ElementOutputShape {
            matrices: 1,
            ..Default::default()
        },
        plan(2),
        |_, _, out: &mut ElementOutputs<'_, f64>| {
            out.matrix(0).fill(1.0);
            Ok::<_, ()>(())
        },
        no_visit,
    )
    .unwrap_err();
    assert!(matches!(error, ElementAssemblyError::OutsidePattern { .. }));
    // Exact-zero contributions carry no support and never fail.
    assemble_elements(
        &map,
        &diagonal,
        ElementOutputShape {
            matrices: 1,
            ..Default::default()
        },
        plan(2),
        |_, dofs, out: &mut ElementOutputs<'_, f64>| {
            let n = dofs.len();
            for local in 0..n {
                out.matrix(0)[local * n + local] = 1.0;
            }
            Ok::<_, ()>(())
        },
        no_visit,
    )
    .unwrap();
}
