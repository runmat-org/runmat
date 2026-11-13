mod energy;
pub(in crate::assembly) use energy::assemble_preparing_with_owner;
pub(crate) use energy::SolidEnergyPreparation;
use std::{
    cell::RefCell, collections::BTreeMap, future::Future, marker::PhantomData, pin::Pin, rc::Rc,
};

use runmat_analysis_core::MechanicalDampingModel;
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact, SolverVolumeElement};
use runmat_simulation_usage::AsyncCapacityOwner;

use crate::algebra::{OperatorRole, RealSparseMatrix};
use crate::{
    physics::structural::material::StructuralElementMaterial,
    planning::resource::LocalExecutionPlan, properties::CompiledPropertyMap,
};

use super::elements::solid::material::SolidMaterialError;
use super::elements::solid::{
    global_stiffness_matrix as tetrahedron4_stiffness, tetrahedron10_consistent_mass_matrix,
    tetrahedron10_consistent_mass_matrix_with_capacity, tetrahedron10_global_stiffness_matrix,
    tetrahedron10_global_stiffness_matrix_with_capacity, tetrahedron4_consistent_mass_matrix,
    SolidMaterial, Tetrahedron10CapacityError, Tetrahedron10ElementError,
    Tetrahedron10ElementGeometry, Tetrahedron4ElementError, Tetrahedron4ElementGeometry,
};
use crate::algebra::{
    element_assembly::{
        assemble_elements_with_async_owner, assemble_elements_with_owner, within_local_plan,
        AssembledElements, AsyncElementKernel, ElementAssemblyError, ElementDofMap,
        ElementFillFuture, ElementOutputShape, ElementOutputs, ElementView,
    },
    MatrixProperties, RealSparseAssembler,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverSolidTopology {
    pub dof_count: usize,
    pub node_count: usize,
    pub volume_element_count: usize,
    pub order: ElementOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SolverSolidAssemblyError {
    #[error(transparent)]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
    #[error("awaited assembly for {0:?} requires a staged element kernel")]
    AwaitedKernelRequiresStaging(ElementOrder),
    #[error("invalid solver mesh artifact: {0}")]
    InvalidArtifact(String),
    #[error("invalid solid material coverage: {0}")]
    InvalidMaterialCoverage(String),
    #[error("element {element_id} has {actual} nodes; expected {expected}")]
    InvalidElementNodeCount {
        element_id: u64,
        expected: usize,
        actual: usize,
    },
    #[error("element {element_id} references unknown node {node_id}")]
    UnknownElementNode { element_id: u64, node_id: u64 },
    #[error("element {element_id} region {region_id} has no material assignment")]
    UnassignedRegion { element_id: u64, region_id: String },
    #[error("element {element_id} region {region_id} has {assignment_count} material assignments")]
    AmbiguousRegion {
        element_id: u64,
        region_id: String,
        assignment_count: usize,
    },
    #[error("element {element_id} stiffness failed: {message}")]
    ElementStiffness { element_id: u64, message: String },
    #[error("element {element_id} mass failed: {message}")]
    ElementMass { element_id: u64, message: String },
    #[error("invalid solid mass operator")]
    InvalidMassOperator,
    #[error("solid sparse operator failed: {0}")]
    SparseOperator(String),
    #[error("solid local execution failed: {0}")]
    LocalExecution(String),
    #[error("solid assembly was cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolverSolidOperators {
    pub stiffness: RealSparseMatrix,
    pub consistent_mass: RealSparseMatrix,
    pub damping: RealSparseMatrix,
    pub lumped_mass_diag: Vec<f64>,
    pub total_mass_kg: f64,
}

/// Exact successful Tet4 reference work for one immutable solid assembly block.
/// A failed element reports its executed primitive prefixes through the existing ledger
/// reservations; the unused grant remains available to the child.
pub struct Tet4SolidGrantSchedule {
    element_count: usize,
    primitive_per_element: runmat_simulation_usage::OperationCount,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SolidCapacityPhase {
    Block { elements: std::ops::Range<usize> },
    Tet4Element { element: usize },
    Tet10Element { element: usize },
}

pub(in crate::assembly) struct SolidAssemblyExecution<'a> {
    pub local_execution: LocalExecutionPlan,
    pub capture: Option<&'a mut SolidEnergyPreparation>,
    pub maximum_block_elements: usize,
    pub awaited_elements: bool,
}

struct SolidElementGrant<'a, O, F, A> {
    before: &'a RefCell<O>,
    element: usize,
    order: ElementOrder,
    _type: PhantomData<fn() -> (F, A)>,
}

impl<O, F, A> runmat_simulation_usage::AsyncCapacityOwner<A> for SolidElementGrant<'_, O, F, A>
where
    O: FnMut(SolidCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
{
    fn ensure_capacity(
        &self,
        additional: runmat_simulation_usage::OperationCount,
    ) -> Pin<Box<dyn Future<Output = Result<(), A>> + '_>> {
        let phase = match self.order {
            ElementOrder::Tet4 => SolidCapacityPhase::Tet4Element {
                element: self.element,
            },
            ElementOrder::Tet10 => SolidCapacityPhase::Tet10Element {
                element: self.element,
            },
        };
        let future = (self.before.borrow_mut())(phase, additional);
        Box::pin(future)
    }
}

struct SolidAwaitedKernel<'a, O, F, A> {
    elements: &'a [SolverVolumeElement],
    properties: &'a CompiledPropertyMap,
    nodes: &'a NodeIndex,
    damped: bool,
    before: Rc<RefCell<O>>,
    _type: PhantomData<fn() -> (F, A)>,
}

impl<O, F, A> AsyncElementKernel<f64, SolverSolidAssemblyError, A>
    for SolidAwaitedKernel<'_, O, F, A>
where
    O: FnMut(SolidCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: Future<Output = Result<(), A>>,
    A: From<SolverSolidAssemblyError>,
{
    fn fill<'a>(
        &'a mut self,
        index: usize,
        dofs: &'a [u32],
        values: &'a mut [f64],
        shape: ElementOutputShape,
    ) -> ElementFillFuture<'a, SolverSolidAssemblyError, A> {
        Box::pin(async move {
            let element = &self.elements[index];
            let mut out = ElementOutputs::from_buffer(dofs.len(), shape, values);
            let grant = SolidElementGrant::<O, F, A> {
                before: &self.before,
                element: index,
                order: element.order,
                _type: PhantomData,
            };
            if element.order == ElementOrder::Tet4 {
                let required = runmat_simulation_usage::OperationCount::from_operations(
                    1_951 + if self.damped { 432 } else { 0 },
                );
                grant
                    .ensure_capacity(required)
                    .await
                    .map_err(ElementAssemblyError::Admission)?;
                return element_contribution(
                    element,
                    self.properties,
                    self.nodes,
                    self.damped,
                    &mut out,
                )
                .map_err(|error| ElementAssemblyError::Element {
                    element: index,
                    error,
                });
            }
            let material = resolve_element_material(element, self.properties).map_err(|error| {
                ElementAssemblyError::Element {
                    element: index,
                    error,
                }
            })?;
            let elasticity = solid_material(material);
            let (coordinates, _) =
                element_geometry::<10>(element.element_id, &element.node_ids, self.nodes).map_err(
                    |error| ElementAssemblyError::Element {
                        element: index,
                        error,
                    },
                )?;
            let geometry = Tetrahedron10ElementGeometry {
                nodes_m: coordinates,
            };
            let stiffness =
                tetrahedron10_global_stiffness_matrix_with_capacity(elasticity, geometry, &grant)
                    .await
                    .map_err(|error| match error {
                        Tetrahedron10CapacityError::Numerical(error) => {
                            ElementAssemblyError::Element {
                                element: index,
                                error: tetrahedron10_failure(element.element_id, error, false),
                            }
                        }
                        Tetrahedron10CapacityError::Admission(error) => {
                            ElementAssemblyError::Admission(error)
                        }
                    })?;
            let mass =
                tetrahedron10_consistent_mass_matrix_with_capacity(elasticity, geometry, &grant)
                    .await
                    .map_err(|error| match error {
                        Tetrahedron10CapacityError::Numerical(error) => {
                            ElementAssemblyError::Element {
                                element: index,
                                error: tetrahedron10_failure(element.element_id, error, true),
                            }
                        }
                        Tetrahedron10CapacityError::Admission(error) => {
                            ElementAssemblyError::Admission(error)
                        }
                    })?;
            grant
                .ensure_capacity(runmat_simulation_usage::OperationCount::from_operations(
                    128 + if self.damped { 2_700 } else { 0 },
                ))
                .await
                .map_err(ElementAssemblyError::Admission)?;
            write_element(&mut out, &stiffness, &mass, material.damping, self.damped).map_err(
                |error| ElementAssemblyError::Element {
                    element: index,
                    error,
                },
            )
        })
    }
}

impl Tet4SolidGrantSchedule {
    pub fn new(
        artifact: &SolverMeshArtifact,
        properties: &CompiledPropertyMap,
    ) -> Result<Self, SolverSolidAssemblyError> {
        if properties.identity.mesh_digest != Some(artifact.canonical_digest) {
            return Err(SolverSolidAssemblyError::InvalidMaterialCoverage(
                "compiled property map belongs to a different solver mesh".into(),
            ));
        }
        let mut damped = false;
        for element in &artifact.topology.volume_elements {
            if element.order != ElementOrder::Tet4 {
                return Err(SolverSolidAssemblyError::AwaitedKernelRequiresStaging(
                    element.order,
                ));
            }
            damped |= !matches!(
                resolve_element_material(element, properties)?.damping,
                MechanicalDampingModel::Undamped
            );
        }
        // Stiffness: 24 volume + 120 gradients + 15 elasticity + 1728 contraction.
        // Mass: 24 volume + 14 matrix. Write: 26 row-sum/scale, plus 3*12² if
        // this operator carries a damping matrix for any element.
        let primitive_per_element = runmat_simulation_usage::OperationCount::from_operations(
            1_951 + if damped { 432 } else { 0 },
        );
        Ok(Self {
            element_count: artifact.topology.volume_elements.len(),
            primitive_per_element,
        })
    }

    pub fn block_required(
        &self,
        block: std::ops::Range<usize>,
        scatter: runmat_simulation_usage::OperationCount,
    ) -> Result<runmat_simulation_usage::OperationCount, SolverSolidAssemblyError> {
        use runmat_simulation_usage::LedgerError;
        if block.start > block.end || block.end > self.element_count {
            return Err(SolverSolidAssemblyError::Accounting(
                LedgerError::InvalidReservation,
            ));
        }
        let count = u64::try_from(block.len())
            .map_err(|_| SolverSolidAssemblyError::Accounting(LedgerError::CountOverflow))?;
        self.primitive_per_element
            .checked_mul(count)
            .and_then(|work| work.checked_add(scatter))
            .map_err(|_| SolverSolidAssemblyError::Accounting(LedgerError::CountOverflow))
    }
}

pub fn solver_solid_topology(
    artifact: &SolverMeshArtifact,
) -> Result<SolverSolidTopology, SolverSolidAssemblyError> {
    artifact
        .validate()
        .map_err(|failure| SolverSolidAssemblyError::InvalidArtifact(failure.to_string()))?;
    Ok(SolverSolidTopology {
        dof_count: artifact.topology.nodes.len().saturating_mul(3),
        node_count: artifact.topology.nodes.len(),
        volume_element_count: artifact.topology.volume_elements.len(),
        order: artifact.resolved_request.element_order,
    })
}

pub fn assemble_solver_solid_operators(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    assemble_solver_solid_operators_with_execution(
        artifact,
        properties,
        LocalExecutionPlan::serial(),
    )
}

/// Computes bounded waves of element contributions under the admitted local CPU plan, then
/// scatters them in canonical mesh order. No worker owns a replica of the global sparse rows.
pub fn assemble_solver_solid_operators_with_execution(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    local_execution: LocalExecutionPlan,
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    assemble_solver_solid_operators_capturing(artifact, properties, local_execution, None)
}

pub(in crate::assembly) fn assemble_solver_solid_operators_capturing(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    local_execution: LocalExecutionPlan,
    capture: Option<&mut SolidEnergyPreparation>,
) -> Result<SolverSolidOperators, SolverSolidAssemblyError> {
    within_local_plan(local_execution, || {
        let future = assemble_solver_solid_operators_capturing_with_owner(
            artifact,
            properties,
            SolidAssemblyExecution {
                local_execution,
                capture,
                maximum_block_elements: usize::MAX,
                awaited_elements: false,
            },
            |_, _| std::future::ready(Ok(())),
        );
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result,
            std::task::Poll::Pending => unreachable!("immediate solid owner cannot suspend"),
        }
    })
    .map_err(|error| SolverSolidAssemblyError::LocalExecution(error.to_string()))?
}

pub(in crate::assembly) async fn assemble_solver_solid_operators_capturing_with_owner<O, F, A>(
    artifact: &SolverMeshArtifact,
    properties: &CompiledPropertyMap,
    execution: SolidAssemblyExecution<'_>,
    before_block: O,
) -> Result<SolverSolidOperators, A>
where
    O: FnMut(SolidCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<SolverSolidAssemblyError> + Send,
{
    let topology = solver_solid_topology(artifact)?;
    if properties.identity.mesh_digest != Some(artifact.canonical_digest) {
        return Err(SolverSolidAssemblyError::InvalidMaterialCoverage(
            "compiled property map belongs to a different solver mesh".into(),
        )
        .into());
    }
    let nodes = artifact
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            (
                node.node_id,
                (node.coordinates_m.canonical_value(), index * 3),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assemble_solid_operators_with_owner(
        &artifact.topology.volume_elements,
        properties,
        &nodes,
        topology.dof_count,
        execution,
        before_block,
    )
    .await
}

const STIFFNESS: usize = 0;
const MASS: usize = 1;
const DAMPING: usize = 2;
const LUMPED_MASS: usize = 0;
const ELEMENT_MASS: usize = 0;

/// Element kernels run on the admitted CPU pool and are scattered into one shared symbolic
/// pattern in canonical element order, so operators are bitwise independent of thread count.
async fn assemble_solid_operators_with_owner<O, F, A>(
    elements: &[SolverVolumeElement],
    properties: &CompiledPropertyMap,
    nodes: &NodeIndex,
    dof_count: usize,
    execution: SolidAssemblyExecution<'_>,
    before_block: O,
) -> Result<SolverSolidOperators, A>
where
    O: FnMut(SolidCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<SolverSolidAssemblyError> + Send,
{
    let SolidAssemblyExecution {
        local_execution,
        mut capture,
        maximum_block_elements,
        awaited_elements,
    } = execution;
    let map =
        ElementDofMap::from_elements(dof_count, elements.len(), local_execution, |index, out| {
            let element = &elements[index];
            for node_id in &element.node_ids {
                let (_, offset) = nodes.get(node_id).copied().ok_or(
                    SolverSolidAssemblyError::UnknownElementNode {
                        element_id: element.element_id,
                        node_id: *node_id,
                    },
                )?;
                out.extend((0..3).map(|axis| Some(offset + axis)));
            }
            Ok(())
        })
        .map_err(element_assembly_error)?;
    let pattern = map.symbolic_pattern(local_execution).map_err(|error| {
        element_assembly_error(
            error
                .map_element(|never| match never {})
                .map_admission(|never| match never {}),
        )
    })?;
    let damped = elements.iter().try_fold(false, |damped, element| {
        Ok::<_, SolverSolidAssemblyError>(
            damped
                || !matches!(
                    resolve_element_material(element, properties)?.damping,
                    MechanicalDampingModel::Undamped
                ),
        )
    })?;
    let shape = ElementOutputShape {
        matrices: if damped { 3 } else { 2 },
        vectors: 1,
        scalars: 1,
    };
    let before = Rc::new(RefCell::new(before_block));
    let kernel = |index: usize, _: &[u32], out: &mut ElementOutputs<'_, f64>| {
        element_contribution(&elements[index], properties, nodes, damped, out)
    };
    let mut visit = |view: &ElementView<'_, f64>| match capture.as_deref_mut() {
        Some(capture) => capture.capture(view.dofs, view.matrix(STIFFNESS)),
        None => Ok(()),
    };
    let grant = |block: std::ops::Range<usize>, scatter_work| {
        let future =
            (before.borrow_mut())(SolidCapacityPhase::Block { elements: block }, scatter_work);
        async move { future.await.map_err(ElementAssemblyError::Admission) }
    };
    let assembled = if awaited_elements {
        let mut async_kernel = SolidAwaitedKernel::<O, F, A> {
            elements,
            properties,
            nodes,
            damped,
            before: Rc::clone(&before),
            _type: PhantomData,
        };
        assemble_elements_with_async_owner(
            &map,
            &pattern,
            shape,
            local_execution,
            maximum_block_elements,
            kernel,
            &mut async_kernel,
            &mut visit,
            grant,
        )
        .await
    } else {
        assemble_elements_with_owner(
            &map,
            &pattern,
            shape,
            local_execution,
            maximum_block_elements,
            kernel,
            &mut visit,
            grant,
        )
        .await
    }
    .map_err(element_assembly_error)?;
    let AssembledElements {
        matrices,
        vectors,
        scalars,
    } = assembled;
    let mut matrices = matrices.into_iter();
    let stiffness = matrices.next().unwrap_or_default();
    let mass = matrices.next().unwrap_or_default();
    let damping = matrices.next();
    Ok(SolverSolidOperators {
        stiffness: sparse(&pattern, stiffness, OperatorRole::Stiffness)?,
        consistent_mass: sparse(&pattern, mass, OperatorRole::Mass)?,
        damping: match damping {
            Some(values) => sparse(&pattern, values, OperatorRole::Damping)?,
            None => finish_operator(
                RealSparseAssembler::square(
                    dof_count,
                    OperatorRole::Damping,
                    MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                ),
                OperatorRole::Damping,
            )?,
        },
        lumped_mass_diag: vectors.into_iter().nth(LUMPED_MASS).unwrap_or_default(),
        total_mass_kg: scalars[ELEMENT_MASS],
    })
}

fn element_assembly_error<A: From<SolverSolidAssemblyError>>(
    error: ElementAssemblyError<SolverSolidAssemblyError, A>,
) -> A {
    match error {
        ElementAssemblyError::Element { error, .. } => error.into(),
        ElementAssemblyError::Admission(error) => error,
        ElementAssemblyError::Accounting(error) => {
            SolverSolidAssemblyError::Accounting(error).into()
        }
        ElementAssemblyError::Cancelled => SolverSolidAssemblyError::Cancelled.into(),
        ElementAssemblyError::Execution(reason) => {
            SolverSolidAssemblyError::LocalExecution(reason).into()
        }
        ElementAssemblyError::DofOutOfRange {
            element,
            dof,
            dof_count,
        } => SolverSolidAssemblyError::SparseOperator(format!(
            "element {element} references degree of freedom {dof} outside 0..{dof_count}"
        ))
        .into(),
        ElementAssemblyError::DofIndexOverflow(count) => SolverSolidAssemblyError::SparseOperator(
            format!("degree-of-freedom count {count} exceeds the 32-bit assembly index"),
        )
        .into(),
        ElementAssemblyError::OutsidePattern { row, column } => {
            SolverSolidAssemblyError::SparseOperator(format!(
                "assembly entry ({row}, {column}) is outside the symbolic pattern"
            ))
            .into()
        }
        ElementAssemblyError::PatternMismatch(reason) => SolverSolidAssemblyError::SparseOperator(
            format!("symbolic pattern differs from the element map: {reason}"),
        )
        .into(),
        ElementAssemblyError::Pattern(error) => {
            SolverSolidAssemblyError::SparseOperator(error.to_string()).into()
        }
        ElementAssemblyError::Allocation(count) => SolverSolidAssemblyError::SparseOperator(
            format!("assembly storage allocation failed for {count} entries"),
        )
        .into(),
    }
}

fn element_contribution(
    element: &SolverVolumeElement,
    properties: &CompiledPropertyMap,
    nodes: &NodeIndex,
    damped: bool,
    out: &mut ElementOutputs<'_, f64>,
) -> Result<(), SolverSolidAssemblyError> {
    let material = resolve_element_material(element, properties)?;
    let elasticity = solid_material(material);
    match element.order {
        ElementOrder::Tet4 => {
            let (coordinates, _) =
                element_geometry::<4>(element.element_id, &element.node_ids, nodes)?;
            let geometry = Tetrahedron4ElementGeometry {
                nodes_m: coordinates,
            };
            let stiffness = tetrahedron4_stiffness(elasticity, geometry)
                .map_err(|error| tetrahedron4_failure(element.element_id, error, false))?;
            let mass = tetrahedron4_consistent_mass_matrix(elasticity, geometry)
                .map_err(|error| tetrahedron4_failure(element.element_id, error, true))?;
            write_element(out, &stiffness, &mass, material.damping, damped)
        }
        ElementOrder::Tet10 => {
            let (coordinates, _) =
                element_geometry::<10>(element.element_id, &element.node_ids, nodes)?;
            let geometry = Tetrahedron10ElementGeometry {
                nodes_m: coordinates,
            };
            let stiffness = tetrahedron10_global_stiffness_matrix(elasticity, geometry)
                .map_err(|error| tetrahedron10_failure(element.element_id, error, false))?;
            let mass = tetrahedron10_consistent_mass_matrix(elasticity, geometry)
                .map_err(|error| tetrahedron10_failure(element.element_id, error, true))?;
            write_element(out, &stiffness, &mass, material.damping, damped)
        }
    }
}

fn write_element<const D: usize>(
    out: &mut ElementOutputs<'_, f64>,
    stiffness: &[[f64; D]; D],
    mass: &[[f64; D]; D],
    damping: MechanicalDampingModel,
    damped: bool,
) -> Result<(), SolverSolidAssemblyError> {
    // This shared output path handles Tet4 (D=12) and Tet10 (D=30).
    // Reserve before writing any output. The row-sum and diagonal reductions
    // are charged even when mass validation subsequently fails.
    let nodes = u64::try_from(D / 3).map_err(|_| {
        SolverSolidAssemblyError::Accounting(runmat_simulation_usage::LedgerError::CountOverflow)
    })?;
    let dimension = u64::try_from(D).map_err(|_| {
        SolverSolidAssemblyError::Accounting(runmat_simulation_usage::LedgerError::CountOverflow)
    })?;
    let reduction = nodes
        .checked_mul(nodes)
        .and_then(|value| value.checked_add(nodes))
        .and_then(|value| value.checked_sub(2))
        .ok_or(SolverSolidAssemblyError::Accounting(
            runmat_simulation_usage::LedgerError::CountOverflow,
        ))?;
    let damping_count = if damped {
        dimension
            .checked_mul(dimension)
            .and_then(|value| value.checked_mul(3))
            .ok_or(SolverSolidAssemblyError::Accounting(
                runmat_simulation_usage::LedgerError::CountOverflow,
            ))?
    } else {
        0
    };
    let prefix =
        reduction
            .checked_add(damping_count)
            .ok_or(SolverSolidAssemblyError::Accounting(
                runmat_simulation_usage::LedgerError::CountOverflow,
            ))?;
    let full = prefix
        .checked_add(
            nodes
                .checked_mul(2)
                .ok_or(SolverSolidAssemblyError::Accounting(
                    runmat_simulation_usage::LedgerError::CountOverflow,
                ))?,
        )
        .ok_or(SolverSolidAssemblyError::Accounting(
            runmat_simulation_usage::LedgerError::CountOverflow,
        ))?;
    let usage = runmat_simulation_usage::UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                runmat_simulation_usage::UsageCategory::Assembly,
                runmat_simulation_usage::OperationCount::from_operations(full),
            )
        })
        .transpose()
        .map_err(SolverSolidAssemblyError::Accounting)?;
    for (target, row) in out.matrix(STIFFNESS).chunks_exact_mut(D).zip(stiffness) {
        target.copy_from_slice(row);
    }
    for (target, row) in out.matrix(MASS).chunks_exact_mut(D).zip(mass) {
        target.copy_from_slice(row);
    }
    if damped {
        let (mass_coefficient, stiffness_coefficient) = match damping {
            MechanicalDampingModel::Undamped => (0.0, 0.0),
            MechanicalDampingModel::Rayleigh {
                mass_coefficient,
                stiffness_coefficient,
            } => (
                mass_coefficient.canonical_value(),
                stiffness_coefficient.canonical_value(),
            ),
        };
        let target = out.matrix(DAMPING);
        for row in 0..D {
            for column in 0..D {
                target[row * D + column] = mass_coefficient * mass[row][column]
                    + stiffness_coefficient * stiffness[row][column];
            }
        }
    }
    // Row-sum-scaled nodal lumping: each node's three DOFs receive its share of element mass.
    let element_mass_kg = mass
        .iter()
        .step_by(3)
        .map(|row| row.iter().step_by(3).sum::<f64>())
        .sum::<f64>();
    let diagonal_sum = (0..D / 3).map(|node| mass[node * 3][node * 3]).sum::<f64>();
    if !element_mass_kg.is_finite()
        || element_mass_kg <= 0.0
        || !diagonal_sum.is_finite()
        || diagonal_sum <= 0.0
    {
        if let Some(batch) = &mut batch {
            batch
                .complete(runmat_simulation_usage::OperationCount::from_operations(
                    prefix,
                ))
                .map_err(SolverSolidAssemblyError::Accounting)?;
        }
        return Err(SolverSolidAssemblyError::InvalidMassOperator);
    }
    let lumped = out.vector(LUMPED_MASS);
    for node in 0..D / 3 {
        let nodal_mass = element_mass_kg * mass[node * 3][node * 3] / diagonal_sum;
        lumped[node * 3..node * 3 + 3].fill(nodal_mass);
    }
    *out.scalar(ELEMENT_MASS) = element_mass_kg;
    if let Some(batch) = &mut batch {
        batch
            .complete(runmat_simulation_usage::OperationCount::from_operations(
                full,
            ))
            .map_err(SolverSolidAssemblyError::Accounting)?;
    }
    Ok(())
}

pub(super) fn resolve_element_material(
    element: &SolverVolumeElement,
    properties: &CompiledPropertyMap,
) -> Result<StructuralElementMaterial, SolverSolidAssemblyError> {
    properties
        .solid_material(element.stable_identity)
        .ok_or_else(|| SolverSolidAssemblyError::UnassignedRegion {
            element_id: element.element_id,
            region_id: element.region_id.source_topology_id.clone(),
        })
        .and_then(|entry| {
            StructuralElementMaterial::from_model(&entry.material).map_err(|error| {
                SolverSolidAssemblyError::InvalidMaterialCoverage(error.to_string())
            })
        })
}

fn solid_material(material: StructuralElementMaterial) -> SolidMaterial {
    SolidMaterial {
        youngs_modulus_pa: material.youngs_modulus_pa,
        poisson_ratio: runmat_types::physical::PoissonRatio::new(
            material.poisson_ratio.canonical_value(),
        ),
        density_kg_per_m3: runmat_types::physical::MassDensity::new(
            material.density_kg_per_m3.canonical_value(),
        ),
    }
}

fn sparse(
    pattern: &crate::algebra::SparsePattern,
    values: Vec<f64>,
    role: OperatorRole,
) -> Result<RealSparseMatrix, SolverSolidAssemblyError> {
    let assembler = RealSparseAssembler::from_planned_values(
        pattern.clone(),
        values,
        role,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| SolverSolidAssemblyError::SparseOperator(format!("{error:?}")))?;
    finish_operator(assembler, role)
}

fn finish_operator(
    mut assembler: RealSparseAssembler,
    role: OperatorRole,
) -> Result<RealSparseMatrix, SolverSolidAssemblyError> {
    assembler.add_contribution(super::solid_matrix::solid_operator_provenance(role));
    assembler
        .finish()
        .map_err(|error| SolverSolidAssemblyError::SparseOperator(format!("{error:?}")))
}

pub(super) type NodeIndex = BTreeMap<u64, ([f64; 3], usize)>;

pub(super) fn element_geometry<const N: usize>(
    element_id: u64,
    node_ids: &[u64],
    nodes: &NodeIndex,
) -> Result<([[f64; 3]; N], [usize; N]), SolverSolidAssemblyError> {
    if node_ids.len() != N {
        return Err(SolverSolidAssemblyError::InvalidElementNodeCount {
            element_id,
            expected: N,
            actual: node_ids.len(),
        });
    }
    let mut coordinates = [[0.0; 3]; N];
    let mut offsets = [0; N];
    for (local, node_id) in node_ids.iter().copied().enumerate() {
        let (point, offset) =
            nodes
                .get(&node_id)
                .copied()
                .ok_or(SolverSolidAssemblyError::UnknownElementNode {
                    element_id,
                    node_id,
                })?;
        coordinates[local] = point;
        offsets[local] = offset;
    }
    Ok((coordinates, offsets))
}

fn element_failure(element_id: u64, failure: impl std::fmt::Display) -> SolverSolidAssemblyError {
    SolverSolidAssemblyError::ElementStiffness {
        element_id,
        message: failure.to_string(),
    }
}

fn tetrahedron4_failure(
    element_id: u64,
    failure: Tetrahedron4ElementError,
    mass: bool,
) -> SolverSolidAssemblyError {
    match failure {
        Tetrahedron4ElementError::Accounting(error)
        | Tetrahedron4ElementError::InvalidMaterial(SolidMaterialError::Accounting(error)) => {
            SolverSolidAssemblyError::Accounting(error)
        }
        other if mass => mass_failure(element_id, other),
        other => element_failure(element_id, other),
    }
}

fn tetrahedron10_failure(
    element_id: u64,
    failure: Tetrahedron10ElementError,
    mass: bool,
) -> SolverSolidAssemblyError {
    match failure {
        Tetrahedron10ElementError::Accounting(error)
        | Tetrahedron10ElementError::InvalidMaterial(SolidMaterialError::Accounting(error)) => {
            SolverSolidAssemblyError::Accounting(error)
        }
        other if mass => mass_failure(element_id, other),
        other => element_failure(element_id, other),
    }
}

fn mass_failure(element_id: u64, failure: impl std::fmt::Display) -> SolverSolidAssemblyError {
    SolverSolidAssemblyError::ElementMass {
        element_id,
        message: failure.to_string(),
    }
}

#[cfg(test)]
#[path = "solver_solid/tests.rs"]
pub(crate) mod tests;
