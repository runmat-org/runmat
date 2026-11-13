# Admitted GENERAL sparse solve

`solve::general` provides a shared real serial CPU entry point for a square original
CSR operator. It retains GENERAL properties: no symmetry, coercivity, positive
pivot, pivot repair or fallback is inferred. Explicit controls govern shared
GMRES, unpivoted fixed-support right ILU and the shared source-independent
right-inverse proof. An unavailable candidate is indeterminate, not a diagnosis
that the operator is singular.

Count-only planning and actual operator rederivation use the same envelope.
Admission checks resident payload and arithmetic work before numerical allocation.
The envelope includes proof construction, original matrix actions, GMRES vector
and projected-system arithmetic, retained factor and physical RHS application.
The proof's original candidate factor is reused for the physical solve. Resource
model revision 99 reflects corrected shared uniqueness work accounting. Borrowed
CSR/RHS, caller state, placement, allocator overhead and measured RSS remain
additional whole-task obligations; this is a component envelope.

Even zero load must pass source-independent operator verification. A successful
physical result carries the construction-only original-operator certificate and
a freshly recomputed residual of the recovered solution against the original
CSR/RHS. Typed cancellation, insufficient admission, indeterminate uniqueness,
nonfinite arithmetic and nonconvergence return no successful solution. Serialized
controls/budget are not proof, and local tangent nonsingularity does not establish
nonlinear equilibrium uniqueness, stability or contact qualification.

Independent closed linear references cover nonsymmetric and indefinite operators
at scales 1e-80, 1 and 1e80. Nullspace/zero load, nonsingular unavailable ILU,
limited RHS work, both admission axes, policy/inventory/overflow and cancellation
checks accompany the shared proof/factor equivalence tests.

The prepared body/contact Newton route now uses this component with complete
immutable material/contact accepted history and the existing load-block controller.
[Governed GENERAL resource demand](general-sparse-resources.md) binds original
controls and actual reduced support. Whole task admission, equality/contact
interaction checks, independent deformable benchmarks, convergence, cold native
contact continuation and mandatory accuracy/publication remain open.
All original scientific and release gates remain unchanged and open.
