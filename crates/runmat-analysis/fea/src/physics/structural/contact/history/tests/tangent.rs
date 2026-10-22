use super::*;
// witness: FEA-CONTACT-OPERATOR-001/friction_tangent_history @
//   penalty_coulomb_contact/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Complete stick and slip tangents include frame terms and match residual changes for
//   both element orders.
#[test]
fn complete_stick_and_slip_tangents_include_frame_terms_and_match_residual_changes() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for (mu, mode) in [
            (0.3, ContactPointMode::Stick),
            (1e-5, ContactPointMode::Slip),
        ] {
            let f = Fixture::new(order, mu);
            let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interface).unwrap();
            let inventory = ContactReferenceInventory::derive(&surfaces, 1000).unwrap();
            let binding = ContactPointBinding::derive(&inventory, 0, f.law(), policy()).unwrap();
            let accepted = binding
                .initial_history(f.primary, f.barycentric, &f.positions)
                .unwrap();
            let current: Vec<_> = f
                .positions
                .iter()
                .map(|p| {
                    [
                        p[0] + 0.004 * p[1],
                        p[1] + 0.006 * p[2],
                        p[2] + 0.005 * p[0],
                    ]
                })
                .collect();
            let bary = f.footprint(&binding, &current);
            let trial = binding
                .trial(&accepted, 0, &f.positions, &current, bary)
                .unwrap();
            assert_eq!(trial.point_response().mode, mode);
            binding
                .validate_history(trial.history(), 1, &current)
                .unwrap();
            let force = f.force(&trial);
            let mut resultant = [0.; 3];
            let mut moment = [0.; 3];
            for (p, r) in current.iter().zip(&force) {
                for k in 0..3 {
                    resultant[k] += r[k];
                }
                moment[0] += p[1] * r[2] - p[2] * r[1];
                moment[1] += p[2] * r[0] - p[0] * r[2];
                moment[2] += p[0] * r[1] - p[1] * r[0];
            }
            for x in resultant.into_iter().chain(moment) {
                near(x, 0., 1e-9);
            }
            let velocities: Vec<_> = f
                .positions
                .iter()
                .map(|p| [0.03 * p[2] + 0.004 * p[1], -0.04 * p[0], 0.02 * p[1]])
                .collect();
            let primary: Vec<_> = f.projection.boundary_faces[f.primary]
                .node_indices
                .iter()
                .map(|&i| velocities[i])
                .collect();
            let query = std::array::from_fn(|k| {
                f.projection.boundary_faces[f.secondary]
                    .node_indices
                    .iter()
                    .zip(binding.reference.shape_values())
                    .map(|(&i, n)| velocities[i][k] * n)
                    .sum()
            });
            let action = trial.tangent_action(query, &primary).unwrap();
            let mut derivative = vec![[0.; 3]; current.len()];
            for (&i, r) in f.projection.boundary_faces[f.primary]
                .node_indices
                .iter()
                .zip(action.primary_force_increment_n())
            {
                for (out, value) in derivative[i].iter_mut().zip(r) {
                    *out += value;
                }
            }
            for (&i, n) in f.projection.boundary_faces[f.secondary]
                .node_indices
                .iter()
                .zip(binding.reference.shape_values())
            {
                for k in 0..3 {
                    derivative[i][k] += n * action.query_force_increment_n[k];
                }
            }
            for step in [1e-5, 1e-6] {
                let evaluate = |sign: f64| {
                    let positions: Vec<_> = current
                        .iter()
                        .zip(&velocities)
                        .map(|(p, v)| std::array::from_fn(|k| p[k] + sign * step * v[k]))
                        .collect();
                    let bary = f.footprint(&binding, &positions);
                    let trial = binding
                        .trial(&accepted, 0, &f.positions, &positions, bary)
                        .unwrap();
                    assert_eq!(trial.point_response().mode, mode);
                    f.force(&trial)
                };
                let plus = evaluate(1.);
                let minus = evaluate(-1.);
                for ((exact, a), b) in derivative
                    .iter()
                    .flatten()
                    .zip(plus.iter().flatten())
                    .zip(minus.iter().flatten())
                {
                    near(*exact, (*a - *b) / (2. * step), 1e-6);
                }
            }
            let replay = binding
                .trial(&accepted, 0, &f.positions, &current, bary)
                .unwrap();
            assert_eq!(trial.history(), replay.history());
            assert_eq!(trial.point_response(), replay.point_response());
            assert_eq!(
                trial.query_internal_force_n(),
                replay.query_internal_force_n()
            );
        }
    }
}
#[test]
fn compiled_original_inventory_storage_scope_budgets_and_cancel_are_explicit() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, 0.3);
        let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interface).unwrap();
        let (inventory, retained, peak) = crate::topology::measure_test_allocations(|| {
            ContactReferenceInventory::derive(&surfaces, 1000).unwrap()
        });
        assert_eq!(retained as u64, inventory.resident_bytes().unwrap());
        assert!(peak as u64 <= inventory.construction_resident_bytes().unwrap());
        assert_eq!(inventory.point_count(), 7);
        let binding = ContactPointBinding::derive(&inventory, 0, f.law(), policy()).unwrap();
        let accepted = binding
            .initial_history(f.primary, f.barycentric, &f.positions)
            .unwrap();
        let next = ContactPointBinding::derive(&inventory, 1, f.law(), policy()).unwrap();
        assert!(next.validate_history(&accepted, 0, &f.positions).is_err());
        assert!(matches!(
            ContactReferenceInventory::derive(&surfaces, 6),
            Err(ContactHistoryError::ReferenceBudget)
        ));
        let (trial, retained, peak) = crate::topology::measure_test_allocations(|| {
            binding
                .trial(&accepted, 0, &f.positions, &f.positions, f.barycentric)
                .unwrap()
        });
        assert_eq!((retained, peak), (0, 0));
        let count = f.projection.boundary_faces[f.primary].node_indices.len();
        let (_, retained, peak) = crate::topology::measure_test_allocations(|| {
            trial
                .tangent_action([0.; 3], &[[0.; 3]; 6][..count])
                .unwrap()
        });
        assert_eq!((retained, peak), (0, 0));
        let token = crate::progress::FeaCancellationToken::new();
        token.cancel();
        let _guard =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                cancellation: token,
                ..Default::default()
            });
        let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
            trial.tangent_action([0.; 3], &[[0.; 3]; 6][..count])
        });
        assert!(matches!(
            result,
            Err(ContactHistoryError::Geometry(
                BoundarySearchError::Cancelled
            ))
        ));
        assert_eq!((retained, peak), (0, 0));
    }
}

#[test]
fn global_planned_contact_scatter_matches_original_residual_derivatives_and_inventory_outlives_selection(
) {
    use crate::algebra::{spmv, MatrixProperties, OperatorRole, RealSparseAssembler};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, 0.3);
        let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interface).unwrap();
        let rows: Vec<_> = (0..f.positions.len())
            .map(|n| [3 * n, 3 * n + 1, 3 * n + 2])
            .collect();
        let budget = ContactSupportBudget {
            maximum_face_pairs: 1,
            maximum_coordinates: 1296,
        };
        let (support, retained, peak) = crate::topology::measure_test_allocations(|| {
            ContactTangentSupport::derive(&surfaces, &rows, 3 * rows.len(), budget).unwrap()
        });
        assert_eq!(retained as u64, support.resident_bytes().unwrap());
        assert!(peak as u64 <= support.construction_resident_bytes().unwrap());
        assert_eq!(support.face_pair_count(), 1);
        let (rejected, retained, peak) = crate::topology::measure_test_allocations(|| {
            ContactTangentSupport::derive(
                &surfaces,
                &rows,
                3 * rows.len(),
                ContactSupportBudget {
                    maximum_coordinates: 1,
                    ..budget
                },
            )
        });
        assert!(matches!(rejected, Err(ContactSupportError::Budget)));
        assert_eq!((retained, peak), (0, 0));
        let inventory = ContactReferenceInventory::derive(&surfaces, 1000).unwrap();
        drop(surfaces);
        let binding = ContactPointBinding::derive(&inventory, 0, f.law(), policy()).unwrap();
        let accepted = binding
            .initial_history(f.primary, f.barycentric, &f.positions)
            .unwrap();
        let current: Vec<_> = f
            .positions
            .iter()
            .map(|p| {
                [
                    p[0] + 0.004 * p[1],
                    p[1] + 0.006 * p[2],
                    p[2] + 0.005 * p[0],
                ]
            })
            .collect();
        let trial = binding
            .trial(
                &accepted,
                0,
                &f.positions,
                &current,
                f.footprint(&binding, &current),
            )
            .unwrap();
        let primary: Vec<_> = f.projection.boundary_faces[f.primary]
            .node_indices
            .iter()
            .map(|&n| rows[n])
            .collect();
        let secondary: Vec<_> = f.projection.boundary_faces[f.secondary]
            .node_indices
            .iter()
            .map(|&n| rows[n])
            .collect();
        let mut residual = vec![0.; 3 * rows.len()];
        let mut assembler = RealSparseAssembler::with_pattern(
            support.pattern().try_clone().unwrap(),
            OperatorRole::EffectiveTangent,
            MatrixProperties::GENERAL,
        )
        .unwrap();
        let invalid_primary = vec![[residual.len(); 3]; primary.len()];
        let (invalid, retained, peak) = crate::topology::measure_test_allocations(|| {
            trial.assemble_into(&invalid_primary, &secondary, &mut residual, &mut assembler)
        });
        assert!(matches!(invalid, Err(ContactHistoryError::InvalidScope)));
        assert_eq!((retained, peak), (0, 0));
        assert!(residual.iter().all(|r| *r == 0.));
        let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
            trial.assemble_into(&primary, &secondary, &mut residual, &mut assembler)
        });
        result.unwrap();
        assert_eq!((retained, peak), (0, 0));
        let matrix = assembler.finish().unwrap();
        for (actual, expected) in residual.iter().zip(f.force(&trial).iter().flatten()) {
            near(*actual, *expected, 1e-12);
        }
        let velocity: Vec<_> = current
            .iter()
            .map(|p| [0.03 * p[2] + 0.004 * p[1], -0.04 * p[0], 0.02 * p[1]])
            .collect();
        let flat: Vec<_> = velocity.iter().flatten().copied().collect();
        let action = spmv(&matrix, &flat).unwrap();
        for step in [1e-5, 1e-6] {
            let evaluate = |sign: f64| {
                let positions: Vec<_> = current
                    .iter()
                    .zip(&velocity)
                    .map(|(p, v)| std::array::from_fn(|k| p[k] + sign * step * v[k]))
                    .collect();
                f.force(
                    &binding
                        .trial(
                            &accepted,
                            0,
                            &f.positions,
                            &positions,
                            f.footprint(&binding, &positions),
                        )
                        .unwrap(),
                )
            };
            let plus = evaluate(1.);
            let minus = evaluate(-1.);
            for ((actual, a), b) in action
                .iter()
                .zip(plus.iter().flatten())
                .zip(minus.iter().flatten())
            {
                near(*actual, (*a - *b) / (2. * step), 1e-6);
            }
        }
    }
}

#[test]
fn potential_pair_support_is_independent_of_selection_order_and_checks_pair_cap_before_allocation()
{
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut f = Fixture::new(order, 0.3);
        let entity = |i: usize| {
            f.projection.boundary_faces[i]
                .provenance
                .iter()
                .find(|p| p.kind == PersistentEntityKind::Face)
                .unwrap()
                .clone()
        };
        f.interface
            .primary
            .bind_resolved_entities(vec![entity(0), entity(1)])
            .unwrap();
        f.interface
            .secondary
            .bind_resolved_entities(vec![entity(2), entity(3)])
            .unwrap();
        let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interface).unwrap();
        let rows: Vec<_> = (0..f.positions.len())
            .map(|n| [3 * n, 3 * n + 1, 3 * n + 2])
            .collect();
        let budget = ContactSupportBudget {
            maximum_face_pairs: 4,
            maximum_coordinates: 5184,
        };
        let support =
            ContactTangentSupport::derive(&surfaces, &rows, 3 * rows.len(), budget).unwrap();
        assert_eq!(support.face_pair_count(), 4);
        let n = 3 * rows.len();
        assert_eq!(support.pattern().column_indices().len(), n * n);
        let (rejected, retained, peak) = crate::topology::measure_test_allocations(|| {
            ContactTangentSupport::derive(
                &surfaces,
                &rows,
                n,
                ContactSupportBudget {
                    maximum_face_pairs: 3,
                    ..budget
                },
            )
        });
        assert!(matches!(rejected, Err(ContactSupportError::Budget)));
        assert_eq!((retained, peak), (0, 0));
        let mut reversed = f.interface.clone();
        reversed
            .primary
            .bind_resolved_entities(vec![entity(1), entity(0)])
            .unwrap();
        reversed
            .secondary
            .bind_resolved_entities(vec![entity(3), entity(2)])
            .unwrap();
        let reversed_surfaces = CanonicalContactSurfaces::derive(&f.projection, &reversed).unwrap();
        let reverse = ContactTangentSupport::derive(&reversed_surfaces, &rows, n, budget).unwrap();
        assert_eq!(
            support.pattern().identity().unwrap(),
            reverse.pattern().identity().unwrap()
        );
    }
}
