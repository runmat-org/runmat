//! One validity rule for a published quadratic tetrahedron, shared by the mesher's Tet10
//! certification and every analysis consumer that integrates over the element.
//!
//! A ten-node tetrahedron maps the reference element by quadratic shape functions, so each
//! Jacobian row is affine in barycentric coordinates and det(J) is a cubic. The element is valid
//! when
//!
//! 1. its Jacobian determinant keeps a scale-relative margin: a lower bound on det(J) over the
//!    whole element, from the cubic's Bernstein coefficients on an adaptive bisection, is at least
//!    [`TET10_MINIMUM_JACOBIAN_RATIO`] of the largest det(J) over a fixed sample set; and
//! 2. its stiffness quadrature converges: the shape-gradient moments
//!    `G[i,a,j,b] = ∫ ∂ₐNᵢ ∂ᵦNⱼ dV`, from which any isotropic material stiffness is formed, agree
//!    between successive tensor Gauss rules to [`STIFFNESS_MOMENT_RELATIVE_TOLERANCE`] of their
//!    largest entry ([`tetrahedron10_stiffness_moments`]). An affine element is integrated exactly
//!    by the four-point rule.
//!
//! The mesher admits only elements this rule accepts, repairing or relaxing the rest, and
//! analysis assembles stiffness from the moments the same function returns, so an element the
//! mesher publishes is one analysis can integrate.
use crate::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};

pub enum Tet10CapacityError<E> {
    Invalidity(Tet10ElementInvalidity),
    Admission(E),
}

impl<E> From<LedgerError> for Tet10CapacityError<E> {
    fn from(error: LedgerError) -> Self {
        Self::Invalidity(Tet10ElementInvalidity::Accounting(error))
    }
}

impl<E> From<Tet10ElementInvalidity> for Tet10CapacityError<E> {
    fn from(error: Tet10ElementInvalidity) -> Self {
        Self::Invalidity(error)
    }
}

async fn ensure_capacity<E>(
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    count: OperationCount,
) -> Result<(), Tet10CapacityError<E>> {
    if let Some(owner) = owner {
        owner
            .ensure_capacity(count)
            .await
            .map_err(Tet10CapacityError::Admission)?;
    }
    Ok(())
}

fn immediate<R>(
    future: impl std::future::Future<Output = Result<R, Tet10CapacityError<std::convert::Infallible>>>,
) -> Result<R, Tet10ElementInvalidity> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            Tet10CapacityError::Invalidity(error) => error,
            Tet10CapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate Tet10 owner cannot suspend"),
    }
}

fn immediate_ledger<R>(
    future: impl std::future::Future<Output = Result<R, Tet10CapacityError<std::convert::Infallible>>>,
) -> Result<R, LedgerError> {
    immediate(future).map_err(|error| match error {
        Tet10ElementInvalidity::Accounting(error) => error,
        _ => unreachable!("ledger-only Tet10 phase returned geometry invalidity"),
    })
}

use super::boundary_face::{
    GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
};

/// Smallest admitted ratio of the element's minimum Jacobian determinant to its largest.
pub const TET10_MINIMUM_JACOBIAN_RATIO: f64 = 0.05;

/// Largest admitted change of any shape-gradient moment between successive Gauss rules, relative
/// to the largest moment.
pub const STIFFNESS_MOMENT_RELATIVE_TOLERANCE: f64 = 1.0e-8;

/// Number of shape-gradient moments `G[i,a,j,b]` (10 nodes × 3 axes, squared).
pub const TET10_GRADIENT_MOMENT_COUNT: usize = 900;

const MAXIMUM_BISECTION_DEPTH: usize = 5;

const BARYCENTRIC_DERIVATIVES: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

const FOUR_POINT_A: f64 = 0.585_410_196_624_968_5;
const FOUR_POINT_B: f64 = 0.138_196_601_125_010_5;
/// Degree-two four-point rule on the reference tetrahedron (weights 1/24 each).
pub const TETRAHEDRON_FOUR_POINT_RULE: [[f64; 4]; 4] = [
    [FOUR_POINT_A, FOUR_POINT_B, FOUR_POINT_B, FOUR_POINT_B],
    [FOUR_POINT_B, FOUR_POINT_A, FOUR_POINT_B, FOUR_POINT_B],
    [FOUR_POINT_B, FOUR_POINT_B, FOUR_POINT_A, FOUR_POINT_B],
    [FOUR_POINT_B, FOUR_POINT_B, FOUR_POINT_B, FOUR_POINT_A],
];

/// The quadrature that integrates an element's stiffness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tet10StiffnessRule {
    /// Straight-sided element: the four-point rule is exact.
    Affine,
    GaussTwelve,
    GaussSixteen,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tet10JacobianMargin {
    /// Lower bound on det(J) over the whole element.
    pub minimum_jacobian_lower_bound: f64,
    /// Largest det(J) over the fixed sample set.
    pub maximum_jacobian: f64,
}

impl Tet10JacobianMargin {
    pub fn is_valid(&self) -> bool {
        self.minimum_jacobian_lower_bound > 0.0
            && self.minimum_jacobian_lower_bound
                >= TET10_MINIMUM_JACOBIAN_RATIO * self.maximum_jacobian
    }
}

/// Converged shape-gradient moments of a valid element, in reference measure (physical volume),
/// with the rule they converged under. The moments give the stiffness of a material whose
/// elasticity tensor is constant over the element; anything else integrates on
/// [`Tet10StiffnessMoments::quadrature_points`].
#[derive(Debug, Clone, PartialEq)]
pub struct Tet10StiffnessMoments {
    pub rule: Tet10StiffnessRule,
    /// `moments[((3 * i + a) * 30) + 3 * j + b] = ∫ ∂ₐNᵢ ∂ᵦNⱼ dV`.
    pub moments: Box<[f64; TET10_GRADIENT_MOMENT_COUNT]>,
}

impl Tet10StiffnessRule {
    /// The rule's barycentric points and reference weights (summing to 1/6). Consumers whose
    /// integrand is not a constant-coefficient bilinear form of the shape gradients (nonlinear,
    /// plastic, temperature-dependent or graded materials, per-point state) integrate a valid
    /// element on these points.
    pub fn quadrature_points(self) -> Vec<([f64; 4], f64)> {
        match self {
            Self::Affine => TETRAHEDRON_FOUR_POINT_RULE
                .iter()
                .map(|point| (*point, 1.0 / 24.0))
                .collect(),
            Self::GaussTwelve => tensor_tetrahedron_rule(&GAUSS_LEGENDRE_TWELVE_UNIT).collect(),
            Self::GaussSixteen => tensor_tetrahedron_rule(&GAUSS_LEGENDRE_SIXTEEN_UNIT).collect(),
        }
    }
}

impl Tet10StiffnessMoments {
    /// The converged point set these moments were integrated on.
    pub fn quadrature_points(&self) -> Vec<([f64; 4], f64)> {
        self.rule.quadrature_points()
    }

    pub fn moment(&self, node_i: usize, axis_a: usize, node_j: usize, axis_b: usize) -> f64 {
        self.moments[(3 * node_i + axis_a) * 30 + 3 * node_j + axis_b]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tet10ElementInvalidity {
    /// Numerical work could not be admitted before its next batch.
    Accounting(LedgerError),
    /// Nonfinite coordinates, or a Jacobian determinant certified nonpositive somewhere.
    InvertedOrDegenerate,
    /// The Jacobian is not certified to keep the scale-relative margin.
    InsufficientJacobianMargin,
    /// The stiffness moments do not converge under the sixteen-point tensor rule.
    UnconvergedStiffnessQuadrature,
}

/// Whether the element passes the shared rule. Grant denial remains a typed
/// failure and cannot be mistaken for invalid geometry.
pub fn is_valid_tetrahedron10_element(coordinates: &[[f64; 3]; 10]) -> Result<bool, LedgerError> {
    match tetrahedron10_stiffness_moments(coordinates) {
        Ok(_) => Ok(true),
        Err(Tet10ElementInvalidity::Accounting(error)) => Err(error),
        Err(_) => Ok(false),
    }
}

/// The element's Jacobian margin, or `None` for nonfinite coordinates or a determinant that is
/// nonpositive at a corner. Numerical admission failure is returned separately.
pub fn tetrahedron10_jacobian_margin(
    coordinates: &[[f64; 3]; 10],
) -> Result<Option<Tet10JacobianMargin>, LedgerError> {
    tetrahedron10_jacobian_margin_for_category(coordinates, UsageCategory::Meshing)
}

pub fn tetrahedron10_jacobian_margin_for_category(
    coordinates: &[[f64; 3]; 10],
    category: UsageCategory,
) -> Result<Option<Tet10JacobianMargin>, LedgerError> {
    immediate_ledger(tetrahedron10_jacobian_margin_with_capacity::<
        std::convert::Infallible,
    >(coordinates, category, None))
}

pub async fn tetrahedron10_jacobian_margin_with_capacity<E>(
    coordinates: &[[f64; 3]; 10],
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<Option<Tet10JacobianMargin>, Tet10CapacityError<E>> {
    if coordinates.iter().flatten().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    ensure_capacity(owner, OperationCount::from_operations(15 * 277)).await?;
    let usage = UsageContext::capture();
    // Fifteen sample Jacobians: reference gradients (92), nine ten-term
    // Jacobian dots (171), and determinant (14) each.
    let mut sample_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(15 * 277))
        })
        .transpose()?;
    let maximum_jacobian =
        SAMPLE_POINTS
            .iter()
            .try_fold(f64::NEG_INFINITY, |maximum, barycentric| {
                let value = determinant(jacobian(coordinates, *barycentric));
                if let Some(batch) = &mut sample_batch {
                    batch.complete(OperationCount::from_operations(277))?;
                }
                Ok::<_, LedgerError>(maximum.max(value))
            })?;
    if !maximum_jacobian.is_finite() || maximum_jacobian <= 0.0 {
        return Ok(None);
    }
    ensure_capacity(owner, OperationCount::from_operations(1)).await?;
    let mut target_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(1))
        })
        .transpose()?;
    let target = TET10_MINIMUM_JACOBIAN_RATIO * maximum_jacobian;
    if let Some(batch) = &mut target_batch {
        batch.complete(OperationCount::from_operations(1))?;
    }
    let reference = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let Some(minimum_jacobian_lower_bound) =
        cell_lower_bound_with_capacity(coordinates, reference, target, 0, category, owner).await?
    else {
        return Ok(None);
    };
    Ok(Some(Tet10JacobianMargin {
        minimum_jacobian_lower_bound,
        maximum_jacobian,
    }))
}

/// Applies the whole rule and returns the converged stiffness moments of a valid element.
pub fn tetrahedron10_stiffness_moments(
    coordinates: &[[f64; 3]; 10],
) -> Result<Tet10StiffnessMoments, Tet10ElementInvalidity> {
    tetrahedron10_stiffness_moments_for_category(coordinates, UsageCategory::Meshing)
}

/// Selects the owner category while retaining the same shared reference rule.
pub fn tetrahedron10_stiffness_moments_for_category(
    coordinates: &[[f64; 3]; 10],
    category: UsageCategory,
) -> Result<Tet10StiffnessMoments, Tet10ElementInvalidity> {
    immediate(tetrahedron10_stiffness_moments_with_capacity::<
        std::convert::Infallible,
    >(coordinates, category, None))
}

pub async fn tetrahedron10_stiffness_moments_with_capacity<E>(
    coordinates: &[[f64; 3]; 10],
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<Tet10StiffnessMoments, Tet10CapacityError<E>> {
    let margin = tetrahedron10_jacobian_margin_with_capacity(coordinates, category, owner)
        .await?
        .ok_or(Tet10ElementInvalidity::InvertedOrDegenerate)?;
    ensure_capacity(owner, OperationCount::from_operations(1)).await?;
    let usage = UsageContext::capture();
    let mut validity_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(1))
        })
        .transpose()
        .map_err(Tet10ElementInvalidity::Accounting)?;
    let valid = margin.is_valid();
    if let Some(batch) = &mut validity_batch {
        batch
            .complete(OperationCount::from_operations(1))
            .map_err(Tet10ElementInvalidity::Accounting)?;
    }
    if !valid {
        return Err((if margin.minimum_jacobian_lower_bound <= 0.0
            && sampled_nonpositive_with_capacity(coordinates, category, owner).await?
        {
            Tet10ElementInvalidity::InvertedOrDegenerate
        } else {
            Tet10ElementInvalidity::InsufficientJacobianMargin
        })
        .into());
    }
    if admitted_is_affine_with_capacity(coordinates, category, owner).await? {
        let moments = integrate_with_capacity(
            coordinates,
            TETRAHEDRON_FOUR_POINT_RULE
                .iter()
                .map(|point| (*point, 1.0 / 24.0)),
            4,
            0,
            category,
            owner,
        )
        .await?
        .ok_or(Tet10ElementInvalidity::InvertedOrDegenerate)?;
        return Ok(Tet10StiffnessMoments {
            rule: Tet10StiffnessRule::Affine,
            moments,
        });
    }
    let eight = integrate_with_capacity(
        coordinates,
        tensor_tetrahedron_rule(&GAUSS_LEGENDRE_EIGHT_UNIT),
        8 * 8 * 8,
        13,
        category,
        owner,
    )
    .await?
    .ok_or(Tet10ElementInvalidity::InvertedOrDegenerate)?;
    let twelve = integrate_with_capacity(
        coordinates,
        tensor_tetrahedron_rule(&GAUSS_LEGENDRE_TWELVE_UNIT),
        12 * 12 * 12,
        13,
        category,
        owner,
    )
    .await?
    .ok_or(Tet10ElementInvalidity::InvertedOrDegenerate)?;
    if converged_with_capacity(&eight, &twelve, category, owner).await? {
        return Ok(Tet10StiffnessMoments {
            rule: Tet10StiffnessRule::GaussTwelve,
            moments: twelve,
        });
    }
    let sixteen = integrate_with_capacity(
        coordinates,
        tensor_tetrahedron_rule(&GAUSS_LEGENDRE_SIXTEEN_UNIT),
        16 * 16 * 16,
        13,
        category,
        owner,
    )
    .await?
    .ok_or(Tet10ElementInvalidity::InvertedOrDegenerate)?;
    if converged_with_capacity(&twelve, &sixteen, category, owner).await? {
        Ok(Tet10StiffnessMoments {
            rule: Tet10StiffnessRule::GaussSixteen,
            moments: sixteen,
        })
    } else {
        Err(Tet10ElementInvalidity::UnconvergedStiffnessQuadrature.into())
    }
}

/// Whether every midside node lies exactly at its edge's chord midpoint.
pub fn tetrahedron10_is_affine(coordinates: &[[f64; 3]; 10]) -> bool {
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS
        .iter()
        .enumerate()
        .all(|(edge, [left, right])| {
            (0..3).all(|axis| {
                coordinates[4 + edge][axis]
                    == 0.5 * (coordinates[*left][axis] + coordinates[*right][axis])
            })
        })
}

async fn admitted_is_affine_with_capacity<E>(
    coordinates: &[[f64; 3]; 10],
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<bool, Tet10CapacityError<E>> {
    ensure_capacity(owner, OperationCount::from_operations(36)).await?;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(36))
        })
        .transpose()?;
    let mut axes = 0_u64;
    let mut affine = true;
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
        for axis in 0..3 {
            axes += 1;
            if coordinates[4 + edge][axis]
                != 0.5 * (coordinates[*left][axis] + coordinates[*right][axis])
            {
                affine = false;
                break;
            }
        }
        if !affine {
            break;
        }
    }
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(axes * 2))?;
    }
    Ok(affine)
}

async fn sampled_nonpositive_with_capacity<E>(
    coordinates: &[[f64; 3]; 10],
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<bool, Tet10CapacityError<E>> {
    ensure_capacity(owner, OperationCount::from_operations(15 * 277)).await?;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(15 * 277))
        })
        .transpose()?;
    let mut visited = 0_u64;
    let mut nonpositive = false;
    for point in SAMPLE_POINTS {
        visited += 1;
        if determinant(jacobian(coordinates, point)) <= 0.0 {
            nonpositive = true;
            break;
        }
    }
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(visited * 277))?;
    }
    Ok(nonpositive)
}

/// Collapsed (Duffy) tensor Gauss rule on the reference tetrahedron: barycentric points and
/// weights that sum to the reference volume 1/6.
pub fn tensor_tetrahedron_rule(rule: &[(f64, f64)]) -> impl Iterator<Item = ([f64; 4], f64)> + '_ {
    rule.iter().flat_map(move |&(r, rw)| {
        rule.iter().flat_map(move |&(s, sw)| {
            rule.iter().map(move |&(t, tw)| {
                let a = 1.0 - r;
                let b = 1.0 - s;
                (
                    [a * b * (1.0 - t), r, a * s, a * b * t],
                    rw * sw * tw * a * a * b,
                )
            })
        })
    })
}

/// Reference shape-function gradients (∂/∂ξ, ∂/∂η, ∂/∂ζ) of the ten nodes.
pub fn tetrahedron10_reference_gradients(barycentric: [f64; 4]) -> [[f64; 3]; 10] {
    let mut gradients = [[0.0; 3]; 10];
    for node in 0..4 {
        let factor = 4.0 * barycentric[node] - 1.0;
        gradients[node] = BARYCENTRIC_DERIVATIVES[node].map(|derivative| factor * derivative);
    }
    for (local_edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        gradients[4 + local_edge] = std::array::from_fn(|axis| {
            4.0 * (barycentric[right] * BARYCENTRIC_DERIVATIVES[left][axis]
                + barycentric[left] * BARYCENTRIC_DERIVATIVES[right][axis])
        });
    }
    gradients
}

/// `J[reference_axis][physical_axis]` at a barycentric point.
fn jacobian(coordinates: &[[f64; 3]; 10], barycentric: [f64; 4]) -> [[f64; 3]; 3] {
    let reference = tetrahedron10_reference_gradients(barycentric);
    std::array::from_fn(|reference_axis| {
        std::array::from_fn(|physical_axis| {
            (0..10)
                .map(|node| coordinates[node][physical_axis] * reference[node][reference_axis])
                .sum()
        })
    })
}

fn determinant(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Lower bound on det(J) over a barycentric sub-cell, from the cubic's Bernstein coefficients on
/// the cell. A cell whose bound misses `target` is bisected along its longest edge, up to
/// [`MAXIMUM_BISECTION_DEPTH`]. Returns `None` when a corner value (an actual determinant) is
/// nonpositive or a coefficient is nonfinite.
async fn cell_lower_bound_with_capacity<E>(
    coordinates: &[[f64; 3]; 10],
    cell: [[f64; 4]; 4],
    target: f64,
    depth: usize,
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<Option<f64>, Tet10CapacityError<E>> {
    // det(J) is multilinear in J's rows and each row is affine on the cell, so the Bernstein
    // coefficient of a degree-three multi-index averages the determinants of the six row
    // assignments drawn from the cell's vertex Jacobians.
    ensure_capacity(owner, OperationCount::from_operations(4 * 263)).await?;
    let usage = UsageContext::capture();
    let mut vertex_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(4 * 263))
        })
        .transpose()?;
    let mut vertices = [[[0.0; 3]; 3]; 4];
    for (entry, vertex) in vertices.iter_mut().zip(cell) {
        *entry = jacobian(coordinates, vertex);
        if let Some(batch) = &mut vertex_batch {
            batch.complete(OperationCount::from_operations(263))?;
        }
    }
    ensure_capacity(owner, OperationCount::from_operations(20 * 90)).await?;
    let mut coefficient_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(20 * 90))
        })
        .transpose()?;
    let mut lowest = f64::INFINITY;
    for first in 0..4 {
        for second in first..4 {
            for third in second..4 {
                let sources = [first, second, third];
                let coefficient = PERMUTATIONS
                    .iter()
                    .map(|permutation| {
                        determinant([
                            vertices[sources[permutation[0]]][0],
                            vertices[sources[permutation[1]]][1],
                            vertices[sources[permutation[2]]][2],
                        ])
                    })
                    .sum::<f64>()
                    / 6.0;
                if let Some(batch) = &mut coefficient_batch {
                    batch.complete(OperationCount::from_operations(90))?;
                }
                if !coefficient.is_finite() || (first == third && coefficient <= 0.0) {
                    return Ok(None);
                }
                lowest = lowest.min(coefficient);
            }
        }
    }
    if lowest >= target || depth >= MAXIMUM_BISECTION_DEPTH {
        return Ok(Some(lowest));
    }
    ensure_capacity(owner, OperationCount::from_operations(74)).await?;
    let mut bisect_batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(74))
        })
        .transpose()?;
    let (left, right) = bisect(cell);
    if let Some(batch) = &mut bisect_batch {
        batch.complete(OperationCount::from_operations(74))?;
    }
    let Some(left) = Box::pin(cell_lower_bound_with_capacity(
        coordinates,
        left,
        target,
        depth + 1,
        category,
        owner,
    ))
    .await?
    else {
        return Ok(None);
    };
    let Some(right) = Box::pin(cell_lower_bound_with_capacity(
        coordinates,
        right,
        target,
        depth + 1,
        category,
        owner,
    ))
    .await?
    else {
        return Ok(None);
    };
    Ok(Some(left.min(right)))
}

fn bisect(mut cell: [[f64; 4]; 4]) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
    let mut selected = (0, 1);
    let mut maximum_length = -1.0;
    for left in 0..4 {
        for right in left + 1..4 {
            let length = (0..4)
                .map(|axis| (cell[left][axis] - cell[right][axis]).powi(2))
                .sum::<f64>();
            if length > maximum_length {
                maximum_length = length;
                selected = (left, right);
            }
        }
    }
    let midpoint =
        std::array::from_fn(|axis| 0.5 * (cell[selected.0][axis] + cell[selected.1][axis]));
    let mut right = cell;
    cell[selected.1] = midpoint;
    right[selected.0] = midpoint;
    (cell, right)
}

async fn integrate_with_capacity<E>(
    coordinates: &[[f64; 3]; 10],
    rule: impl Iterator<Item = ([f64; 4], f64)>,
    point_count: u64,
    rule_ops: u64,
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<Option<Box<[f64; TET10_GRADIENT_MOMENT_COUNT]>>, Tet10CapacityError<E>> {
    // At each point: two reference-gradient evaluations (2×92), nine
    // ten-term Jacobian dots (171), determinant (14), inverse cofactors (36),
    // ten physical gradients (150), weight scale (1), 30 left scales (30),
    // and 900 moment multiply/add updates (1800): 2386 operations.
    // A generated Duffy point additionally has 13 real operations.
    let per_point = OperationCount::from_operations(2_386 + rule_ops);
    let maximum = per_point
        .checked_mul(point_count)
        .map_err(LedgerError::from)?;
    ensure_capacity(owner, maximum).await?;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| usage.ledger().reserve(category, maximum))
        .transpose()?;
    let mut moments = Box::new([0.0; TET10_GRADIENT_MOMENT_COUNT]);
    for (barycentric, weight) in rule {
        let reference = tetrahedron10_reference_gradients(barycentric);
        let j = jacobian(coordinates, barycentric);
        let det = determinant(j);
        if !det.is_finite() || det <= 0.0 {
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(369 + rule_ops))?;
            }
            return Ok(None);
        }
        // ∇ξN = J g for J[reference][physical], so the physical gradient is g = J⁻¹ ∇ξN.
        let inverse = [
            [
                (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det,
                (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det,
                (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det,
            ],
            [
                (j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det,
                (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det,
                (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det,
            ],
            [
                (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det,
                (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det,
                (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det,
            ],
        ];
        let gradients: [[f64; 3]; 10] = reference.map(|gradient| {
            std::array::from_fn(|axis| {
                inverse[axis][0] * gradient[0]
                    + inverse[axis][1] * gradient[1]
                    + inverse[axis][2] * gradient[2]
            })
        });
        let scale = weight * det;
        for (i, gi) in gradients.iter().enumerate() {
            for (a, gradient_i) in gi.iter().enumerate() {
                let left = scale * gradient_i;
                let row = (3 * i + a) * 30;
                for (j_node, gj) in gradients.iter().enumerate() {
                    for (b, gradient_j) in gj.iter().enumerate() {
                        moments[row + 3 * j_node + b] += left * gradient_j;
                    }
                }
            }
        }
        if let Some(batch) = &mut batch {
            batch.complete(per_point)?;
        }
    }
    Ok(moments
        .iter()
        .all(|value| value.is_finite())
        .then_some(moments))
}

async fn converged_with_capacity<E>(
    previous: &[f64; TET10_GRADIENT_MOMENT_COUNT],
    next: &[f64; TET10_GRADIENT_MOMENT_COUNT],
    category: UsageCategory,
    owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
) -> Result<bool, Tet10CapacityError<E>> {
    let scale = next.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return Ok(false);
    }
    ensure_capacity(owner, OperationCount::from_operations(1 + 900)).await?;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(category, OperationCount::from_operations(1 + 900))
        })
        .transpose()?;
    let tolerance = STIFFNESS_MOMENT_RELATIVE_TOLERANCE * scale;
    let mut count = 1_u64;
    let mut matches = true;
    for (left, right) in previous.iter().zip(next) {
        count += 1;
        if (left - right).abs() > tolerance {
            matches = false;
            break;
        }
    }
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(count))?;
    }
    Ok(matches)
}

/// Corners, edge midpoints, face centroids and the centroid.
const SAMPLE_POINTS: [[f64; 4]; 15] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0, 0.5],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
    [0.25, 0.25, 0.25, 0.25],
];

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_simulation_usage::UsageLedger;
    use std::sync::Arc;

    fn straight() -> [[f64; 3]; 10] {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut nodes = [[0.0; 3]; 10];
        nodes[..4].copy_from_slice(&corners);
        for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
            nodes[4 + edge] =
                std::array::from_fn(|axis| 0.5 * (corners[left][axis] + corners[right][axis]));
        }
        nodes
    }

    #[test]
    fn admitted_affine_tet10_validity_and_moments_conserve_work() {
        // Margin: 15 sampled Jacobians, one target scale, four cell-vertex
        // Jacobians, and 20 six-permutation coefficients. Then one ratio
        // scale, 18 midpoint checks, and four full integration points.
        let limit =
            OperationCount::from_operations(15 * 277 + 1 + 4 * 263 + 20 * 90 + 1 + 36 + 4 * 2_386);
        let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
        let context = UsageContext::new(Arc::clone(&ledger));
        context.run_sync(|| {
            tetrahedron10_stiffness_moments(&straight()).unwrap();
        });
        let snapshot = ledger.snapshot().unwrap();
        assert_eq!(snapshot.completed.meshing, limit);
        assert_eq!(snapshot.reserved, OperationCount::ZERO);
    }

    #[test]
    fn straight_element_is_affine_and_its_moments_sum_to_the_volume_identity() {
        let nodes = straight();
        let moments = tetrahedron10_stiffness_moments(&nodes).unwrap();
        assert_eq!(moments.rule, Tet10StiffnessRule::Affine);
        // Σⱼ ∂ᵦNⱼ = 0 (partition of unity), so every moment row sums to zero over nodes.
        for i in 0..10 {
            for a in 0..3 {
                for b in 0..3 {
                    let row_sum = (0..10).map(|j| moments.moment(i, a, j, b)).sum::<f64>();
                    assert!(row_sum.abs() <= 1e-14, "{row_sum}");
                }
            }
        }
        // Σᵢⱼ xᵢ_a xⱼ_b-weighted moments reproduce ∫ δ dV: ∂ₐx_c = δ, so Σᵢ x_iᶜ ∂ₐNᵢ = δₐ꜀.
        let trace = (0..10)
            .flat_map(|i| (0..10).map(move |j| (i, j)))
            .map(|(i, j)| nodes[i][0] * nodes[j][0] * moments.moment(i, 0, j, 0))
            .sum::<f64>();
        assert!((trace - 1.0 / 6.0).abs() <= 1e-14, "{trace}");
    }

    #[test]
    fn mildly_curved_element_converges_and_a_strongly_curved_one_is_rejected() {
        let mut mild = straight();
        mild[4][1] -= 0.05;
        let moments = tetrahedron10_stiffness_moments(&mild).unwrap();
        assert_ne!(moments.rule, Tet10StiffnessRule::Affine);
        let mut strong = straight();
        strong[4][0] += 0.24;
        assert!(!is_valid_tetrahedron10_element(&strong).unwrap());
        let mut inverted = straight();
        inverted[3][2] = -1.0;
        assert_eq!(
            tetrahedron10_stiffness_moments(&inverted),
            Err(Tet10ElementInvalidity::InvertedOrDegenerate)
        );
    }

    /// Element 377 of the published Tet10 bracket mesh (meters): the mesher certified its
    /// Jacobian positive, but analysis could not converge its curved stiffness quadrature.
    const BRACKET_ELEMENT_377: [[f64; 3]; 10] = [
        [
            0.0026731306328256336,
            0.015117805597817715,
            0.028106087506546976,
        ],
        [
            0.0027655490644655236,
            0.015611856897379784,
            0.027771248693692633,
        ],
        [
            0.0026741522806826784,
            0.014865997719849451,
            0.02758560583257708,
        ],
        [
            0.0026731306328256336,
            0.015117805597817715,
            0.027306758031048765,
        ],
        [
            0.002703577318323532,
            0.015366778468109766,
            0.02793932208217616,
        ],
        [
            0.0026845293059057436,
            0.015243441484602419,
            0.027678472586099,
        ],
        [
            0.0026696787751872095,
            0.014990901289704423,
            0.027843813538023086,
        ],
        [
            0.002671569442443801,
            0.015117805597817715,
            0.027707203363988787,
        ],
        [
            0.002703577318323532,
            0.015366778468109766,
            0.027538096154045226,
        ],
        [
            0.0026696675214352866,
            0.014993854118408593,
            0.027444003433363486,
        ],
    ];

    // witness: FEA-MESH-DISCRETIZATION-001/tet10_elevation @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: the bracket element that passed the mesher's positivity certificate but failed
    //   analysis stiffness quadrature is rejected by the one rule both now apply, and its
    //   straightened (affine) repair is admitted; the repair search itself is not covered.
    #[test]
    fn bracket_element_377_fails_the_shared_rule_and_its_straightened_repair_passes() {
        let failure = tetrahedron10_stiffness_moments(&BRACKET_ELEMENT_377).unwrap_err();
        assert!(matches!(
            failure,
            Tet10ElementInvalidity::InsufficientJacobianMargin
                | Tet10ElementInvalidity::UnconvergedStiffnessQuadrature
        ));
        let mut straightened = BRACKET_ELEMENT_377;
        for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
            straightened[4 + edge] = std::array::from_fn(|axis| {
                0.5 * (BRACKET_ELEMENT_377[left][axis] + BRACKET_ELEMENT_377[right][axis])
            });
        }
        assert_eq!(
            tetrahedron10_stiffness_moments(&straightened).unwrap().rule,
            Tet10StiffnessRule::Affine
        );
    }
}
