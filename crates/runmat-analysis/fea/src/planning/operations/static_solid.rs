//! Operations estimate of a linear static solid solve: assembly, factorization or multigrid
//! setup, the solve, and field recovery.
//!
//! Assembly reserves fixed batches per element: an affine element's reference counts, and for a
//! curved quadratic element the tensor Gauss rules its stiffness moments may need (8, 12 and 16
//! points per axis at 2,399 operations per point, the full sequence the shared Tet10 rule may
//! run). Quadratic strain recovery runs in the element kernels and charges assembly too; element
//! stress, energy and the static balance charge the solve.
//!
//! The proving factorization charges the reference work of the symbolic factor it builds, and the
//! factor admission refuses that factor when its work or stored nonzeros exceed the plan's
//! envelope. The envelope is therefore the factorization estimate, and its nonzero bound sizes
//! the substitutions of the direct solve; neither takes a further margin. The multigrid path
//! charges its hierarchy setup and each preconditioned CG iteration; the estimate covers
//! [`MULTIGRID_ITERATIONS_COVERED`] iterations.
//!
//! No kernel of a static solid run charges the `other` category, so its estimate is zero.

use super::FeaOperationEstimate;
use crate::planning::resource::{
    FeaResourceDemand, FeaResourcePlan, SolveResourcePolicy, StaticSolvePath,
};
use runmat_meshing_core::{ElementOrder, SolidMeshCounts, SolverMeshArtifact};
use runmat_simulation_usage::{CountOverflow, OperationCount};

/// PCG iterations the multigrid solve estimate covers. Elasticity on admitted meshes converges
/// well inside this count; a solve that needs more charges more than the estimate.
pub const MULTIGRID_ITERATIONS_COVERED: u32 = 200;

/// Assembly batches of one affine element (stiffness, consistent mass and their validity
/// checks): Tet4 and Tet10. The ledger records 18,116 and 92,885 on the structured cantilever.
const ASSEMBLY_PER_ELEMENT: [u64; 2] = [18_200, 93_000];
/// Additional assembly of one curved Tet10. An element whose moments converge between the
/// Gauss 8 and 12 rules records 5.47 million more than an affine one (2,240 points at 2,399
/// operations); the figure adds a quarter of the Gauss 16 rule (4,096 points) for elements that
/// need it. An element on the Gauss 16 rule charges up to 15.4 million.
const CURVED_TET10_ASSEMBLY: u64 = 8_000_000;
/// Global scatter and operator preparation per stored stiffness nonzero; the per-element
/// batches already cover them on the measured meshes.
const ASSEMBLY_PER_STIFFNESS_NONZERO: u64 = 0;
/// Forward and backward substitution per stored factor nonzero: 4 operations per off-diagonal
/// entry in each of three passes (the initial solve and the proving factor's two refinement
/// steps).
const SOLVE_PER_FACTOR_NONZERO: u64 = 12;
/// Residual products per stored stiffness nonzero around a direct solve: two around the
/// correction, up to three inside the refined solve and the static balance's internal force,
/// at 2 operations per entry.
const SOLVE_PER_STIFFNESS_NONZERO: u64 = 12;
/// Norms, correction updates and balance dot products per unknown around a direct solve.
const SOLVE_PER_UNKNOWN: u64 = 48;
/// Element stress, strain energy and averaging that recovery charges to the solve, per Tet4 and
/// Tet10 element (446 and 307 operations at four recovery points).
const SOLVE_RECOVERY_PER_ELEMENT: [u64; 2] = [600, 400];
/// Multigrid hierarchy setup per stored stiffness nonzero (aggregation, smoothed prolongation
/// and Galerkin products), beside the coarse factor.
const MULTIGRID_SETUP_PER_STIFFNESS_NONZERO: u64 = 200;
/// Rows of the multigrid hierarchy's coarsest level (`AmgOptions::coarse_size`), whose dense
/// factor bounds the coarse setup: at most `rows³ / 3` operations.
const MULTIGRID_COARSE_ROWS: u64 = 3_000;
/// One preconditioned CG iteration per stored stiffness nonzero: the operator product and the
/// V-cycle's smoother and grid-transfer products. The ledger records 11.6 on the curved Tet10
/// cantilever.
const MULTIGRID_ITERATION_PER_STIFFNESS_NONZERO: u64 = 14;

/// Pre-mesh model of the proving factor on a compact tetrahedral solid under nested dissection:
/// factor nonzeros ≈ `FILL · n^(4/3)` and reference work ≈ `WORK · n²` for `n` unknowns, the work
/// including the admission envelope's 25% headroom that a planned factor carries. A thin or
/// slender part fills less, so the model reads high for those.
const FACTOR_FILL_COEFFICIENT: f64 = 6.0;
const FACTOR_WORK_COEFFICIENT: f64 = 5.0;
/// Factor bytes per stored nonzero (value, column index) when choosing the path before meshing.
const FACTOR_BYTES_PER_NONZERO: f64 = 16.0;
/// Elements behind one boundary facet on a curved surface: the facet's own element and the
/// elements that share only one of its edges.
const CURVED_ELEMENTS_PER_CURVED_FACET: f64 = 2.0;
/// Margin on the measured per-element and per-nonzero figures (assembly and the multigrid path)
/// for counts the model does not trace batch by batch.
const MARGIN_NUMERATOR: u64 = 5;
const MARGIN_DENOMINATOR: u64 = 4;

/// The mesh and operator counts that size a static solid estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticSolidOperationCounts {
    pub order: ElementOrder,
    pub elements: u64,
    /// Quadratic elements with a midside node off its chord (curved boundary elements).
    pub curved_elements: u64,
    pub nodes: u64,
    /// Unknowns of the assembled operator.
    pub unknowns: u64,
    /// Stored nonzeros of the assembled stiffness (both triangles and the diagonal).
    pub stiffness_nonzeros: u64,
}

/// The solve path and the work it is planned with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticSolveOperations {
    /// The proving factorization: its symbolic work envelope and factor nonzero bound.
    Factorization {
        factor_work: u64,
        factor_nonzeros: u64,
    },
    /// Elastic rigidity proof with AMG-preconditioned CG.
    Multigrid,
}

pub fn estimate_static_solid_operations(
    counts: StaticSolidOperationCounts,
    solve: StaticSolveOperations,
) -> Result<FeaOperationEstimate, CountOverflow> {
    let order = usize::from(counts.order == ElementOrder::Tet10);
    let curved = if counts.order == ElementOrder::Tet10 {
        counts.curved_elements
    } else {
        0
    };
    let nonzeros = counts.stiffness_nonzeros;
    let assembly = sum([
        product(counts.elements, ASSEMBLY_PER_ELEMENT[order])?,
        product(curved, CURVED_TET10_ASSEMBLY)?,
        product(nonzeros, ASSEMBLY_PER_STIFFNESS_NONZERO)?,
    ])?;
    let (factorization, solve, solve_iterations) = match solve {
        // Admission bounds: the factor's work and nonzeros never exceed them, so they take no
        // margin.
        StaticSolveOperations::Factorization {
            factor_work,
            factor_nonzeros,
        } => (
            OperationCount::from_operations(factor_work),
            OperationCount::from_operations(sum([
                product(factor_nonzeros, SOLVE_PER_FACTOR_NONZERO)?,
                product(nonzeros, SOLVE_PER_STIFFNESS_NONZERO)?,
                product(counts.unknowns, SOLVE_PER_UNKNOWN)?,
                product(counts.elements, SOLVE_RECOVERY_PER_ELEMENT[order])?,
            ])?),
            None,
        ),
        StaticSolveOperations::Multigrid => (
            with_margin(sum([
                product(nonzeros, MULTIGRID_SETUP_PER_STIFFNESS_NONZERO)?,
                dense_factor_work(counts.unknowns.min(MULTIGRID_COARSE_ROWS))?,
            ])?)?,
            with_margin(product(
                product(nonzeros, MULTIGRID_ITERATION_PER_STIFFNESS_NONZERO)?,
                u64::from(MULTIGRID_ITERATIONS_COVERED),
            )?)?,
            Some(MULTIGRID_ITERATIONS_COVERED),
        ),
    };
    Ok(FeaOperationEstimate {
        meshing: OperationCount::ZERO,
        assembly: with_margin(assembly)?,
        factorization,
        solve,
        other: OperationCount::ZERO,
        solve_iterations,
    })
}

/// The estimate of a planned static solid solve from its admitted resource plan and mesh: the
/// plan's exact operator counts and solve path, and the mesh's curved elements. `None` for a
/// plan of another procedure.
pub fn estimate_static_solid_operations_from_plan(
    plan: &FeaResourcePlan,
    mesh: &SolverMeshArtifact,
) -> Result<Option<FeaOperationEstimate>, CountOverflow> {
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &plan.input.demand else {
        return Ok(None);
    };
    let operator = &demand.operator;
    let solve = match &operator.solve {
        SolveResourcePolicy::Iterative {
            path: StaticSolvePath::ElasticRigidityMultigrid,
            ..
        } => StaticSolveOperations::Multigrid,
        SolveResourcePolicy::Iterative { admission, .. }
        | SolveResourcePolicy::SaddlePointGmres { admission, .. } => {
            StaticSolveOperations::Factorization {
                factor_work: admission.update_count_upper_bound.0,
                factor_nonzeros: admission.factor_nonzero_upper_bound.0,
            }
        }
        _ => return Ok(None),
    };
    let Some(order) = mesh
        .topology
        .volume_elements
        .first()
        .map(|element| element.order)
    else {
        return Ok(None);
    };
    estimate_static_solid_operations(
        StaticSolidOperationCounts {
            order,
            elements: operator.topology.elements.0,
            curved_elements: curved_elements(mesh),
            nodes: operator.topology.nodes.0,
            unknowns: operator.sparse.rows.0,
            stiffness_nonzeros: operator.sparse.maximum_nonzeros(),
        },
        solve,
    )
    .map(Some)
}

/// The estimate before meshing, from the estimated mesh counts. `curved_area_fraction` is the
/// share of the boundary area on curved surfaces; the path is the one the planner chooses
/// within `memory_ceiling_bytes`: the factorization when its modelled factor fits.
pub fn estimate_static_solid_operations_from_counts(
    mesh: SolidMeshCounts,
    curved_area_fraction: f64,
    memory_ceiling_bytes: u64,
) -> Result<FeaOperationEstimate, CountOverflow> {
    let nodes = mesh.nodes();
    let unknowns = nodes.checked_mul(3).ok_or(CountOverflow)?;
    let node_nonzeros = mesh
        .node_pairs()
        .checked_mul(2)
        .and_then(|pairs| pairs.checked_add(nodes))
        .ok_or(CountOverflow)?;
    let curved_elements = if mesh.order == ElementOrder::Tet10 {
        (mesh.boundary_facets as f64
            * curved_area_fraction.clamp(0.0, 1.0)
            * CURVED_ELEMENTS_PER_CURVED_FACET)
            .ceil()
            .min(mesh.tetrahedra as f64) as u64
    } else {
        0
    };
    let n = unknowns as f64;
    let factor_nonzeros = FACTOR_FILL_COEFFICIENT * n.powf(4.0 / 3.0);
    let solve = if factor_nonzeros * FACTOR_BYTES_PER_NONZERO <= memory_ceiling_bytes as f64 {
        StaticSolveOperations::Factorization {
            factor_work: whole(FACTOR_WORK_COEFFICIENT * n * n)?,
            factor_nonzeros: whole(factor_nonzeros)?,
        }
    } else {
        StaticSolveOperations::Multigrid
    };
    estimate_static_solid_operations(
        StaticSolidOperationCounts {
            order: mesh.order,
            elements: mesh.tetrahedra,
            curved_elements,
            nodes,
            unknowns,
            stiffness_nonzeros: node_nonzeros.checked_mul(9).ok_or(CountOverflow)?,
        },
        solve,
    )
}

/// Quadratic elements whose midside nodes leave their edge chords. The check reads coordinates
/// only and charges nothing.
fn curved_elements(mesh: &SolverMeshArtifact) -> u64 {
    let positions: std::collections::HashMap<u64, [f64; 3]> = mesh
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect();
    mesh.topology
        .volume_elements
        .iter()
        .filter(|element| element.order == ElementOrder::Tet10 && element.node_ids.len() == 10)
        .filter(|element| {
            let mut nodes = [[0.0; 3]; 10];
            for (slot, id) in nodes.iter_mut().zip(&element.node_ids) {
                match positions.get(id) {
                    Some(position) => *slot = *position,
                    // An unresolved node cannot be proved affine; count the element as curved.
                    None => return true,
                }
            }
            !runmat_meshing_core::tetrahedron10_is_affine(&nodes)
        })
        .count() as u64
}

/// Dense Cholesky reference work of `rows` unknowns, `rows³ / 3` rounded up with its square
/// roots and substitutions covered by the quadratic term.
fn dense_factor_work(rows: u64) -> Result<u64, CountOverflow> {
    let cube = rows
        .checked_mul(rows)
        .and_then(|square| square.checked_mul(rows))
        .ok_or(CountOverflow)?;
    sum([cube.div_ceil(3), product(rows, rows)?])
}

fn product(count: u64, per_item: u64) -> Result<u64, CountOverflow> {
    count.checked_mul(per_item).ok_or(CountOverflow)
}

fn sum<const N: usize>(terms: [u64; N]) -> Result<u64, CountOverflow> {
    terms.into_iter().try_fold(0_u64, |total, term| {
        total.checked_add(term).ok_or(CountOverflow)
    })
}

fn with_margin(operations: u64) -> Result<OperationCount, CountOverflow> {
    operations
        .checked_mul(MARGIN_NUMERATOR)
        .map(|scaled| scaled.div_ceil(MARGIN_DENOMINATOR))
        .map(OperationCount::from_operations)
        .ok_or(CountOverflow)
}

fn whole(value: f64) -> Result<u64, CountOverflow> {
    if value.is_finite() && value >= 0.0 && value < u64::MAX as f64 {
        Ok(value.ceil() as u64)
    } else {
        Err(CountOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(order: ElementOrder, vertices: u64) -> SolidMeshCounts {
        SolidMeshCounts {
            vertices,
            tetrahedra: vertices * 5,
            boundary_facets: vertices / 2,
            order,
        }
    }

    #[test]
    fn the_factor_path_is_estimated_while_its_modelled_factor_fits_the_ceiling() {
        let counts = mesh(ElementOrder::Tet10, 20_000);
        let factor = estimate_static_solid_operations_from_counts(counts, 0.0, u64::MAX).unwrap();
        assert_eq!(factor.solve_iterations, None);
        let multigrid = estimate_static_solid_operations_from_counts(counts, 0.0, 1 << 20).unwrap();
        assert_eq!(
            multigrid.solve_iterations,
            Some(MULTIGRID_ITERATIONS_COVERED)
        );
        assert_eq!(factor.assembly, multigrid.assembly);
        assert!(factor.total().unwrap() > OperationCount::ZERO);
    }

    #[test]
    fn curved_surfaces_raise_only_quadratic_assembly() {
        let flat = estimate_static_solid_operations_from_counts(
            mesh(ElementOrder::Tet10, 5_000),
            0.0,
            u64::MAX,
        )
        .unwrap();
        let curved = estimate_static_solid_operations_from_counts(
            mesh(ElementOrder::Tet10, 5_000),
            0.5,
            u64::MAX,
        )
        .unwrap();
        // 2,500 facets, half curved, two elements behind each: 2,500 curved elements.
        assert!(
            curved.assembly.operations() - flat.assembly.operations()
                >= 2_500 * CURVED_TET10_ASSEMBLY
        );
        // Curved recovery points evaluate their Jacobian inside the element kernels, which
        // charge assembly; no static solid kernel charges `other`.
        assert_eq!(curved.other, OperationCount::ZERO);
        assert_eq!(flat.other, OperationCount::ZERO);
        assert_eq!(curved.factorization, flat.factorization);
        assert_eq!(curved.solve, flat.solve);
        let linear = |fraction| {
            estimate_static_solid_operations_from_counts(
                mesh(ElementOrder::Tet4, 5_000),
                fraction,
                u64::MAX,
            )
            .unwrap()
        };
        assert_eq!(linear(0.0), linear(1.0));
    }

    #[test]
    fn repeated_runs_scale_every_category() {
        let once = estimate_static_solid_operations_from_counts(
            mesh(ElementOrder::Tet4, 1_000),
            0.0,
            u64::MAX,
        )
        .unwrap();
        let thrice = once.repeated(3).unwrap();
        assert_eq!(
            thrice.total().unwrap().operations(),
            3 * once.total().unwrap().operations()
        );
    }
}
