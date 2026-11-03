# q-mesh-fix-6 notes

Branch q-mesh-fix-6 from t1-transplant (f92b8cf0a1, then merged f9b884d862). Work dir: $J/work/q-mesh-fix-6
(scripts sync.sh, batch.sh, wait.sh, meshq.sh, replay_dump.rs debug harness, dump-hook.patch — neither committed).
Build Mac: ~/src/q-mesh-fix-6, ~/targets/q-mesh-fix-6, runs ~/runs/qmf6/{in,base,base-corpus,base-blade4,dbg*}, logs ~/logs/qmf6.

## Baseline (c13-96a1d055ea release CLI = base mesher; `runmat mesh`)
- cabin 100/2 mm tet10: InvalidGeometry CurveMesh vertex pcurve (1 s).
- pin fitting v1 8/0.4 mm tet10: InvalidGeometry OrderElevation exact midpoint leaves trim domain (101 s).
- cantc (cantilever 0.3 m/1 mm tet10 seed 3): QualityTargetUnreachable sliver (43 s). Replay qmf-cc03.json same.
- induction replay qmf-ih.json: FacetRecovery UnsatisfiableConstraint at constraint 8461 (42 s).
- plate.json replay: ok, 420 tets, min SJ 0.01006.
- tetrahedron lib 586/0 (7 ignored).

## Root causes found
- Cantilever: the volume cell metric is the intersection of contributions on the cell's incident entities, then
  graded over the cell graph by scaling the whole tensor to the neighbour's finest length. The cantilever's
  proximity sizing is anisotropic (0.3 m along the beam, 0.05 m across); grading turned it into 0.065 m
  isotropic, so volume edges along the beam that the surface (graded by distance, directional) made 0.0625–0.25 m
  long counted as 1.93x too long next to the boundary, where no point can be admitted.
- Fix (coordinator decision: integrate the metric along edges): size is measured along each edge as the
  integral of the metric between its endpoint metrics (log mean). Boundary nodes carry the field their faces
  were meshed with; interior nodes carry region sizing ∩ every edge/vertex-supported source ∩ every face's
  constant sizing graded from the face triangles (new SurfaceMetricSource). Shape still uses the cell metric.

## Hand-back 2026-11-03 (coordinator asked to wrap up; WIP round)
Commits on q-mesh-fix-6 (base f9b884d862 = t1-transplant):
- 9aec9ae3c1 Diagnostics only: vertex pcurve disagreement and rejected exact midpoints now name their evidence.
- 53f4c8b336 Edge-integrated volume size (size crate point_field.rs: SurfaceMetricSource, integrated_edge_length,
  intersect_metrics; tetrahedron volume_quality/node_size.rs wired into working, stream, mutation, validation).
  NOT yet merge-ready: the tetrahedron lib and curve lib results for this commit are still running (below).

### Results so far (debug build of 53f4c8b336 minus the size-test fix)
- cantilever replay qmf-cc03.json: was QualityTargetUnreachable (65 s) -> ok, 2012 tets, 201 refined, min metric SJ
  0.00286 (target 0.001), 0.6 s. Receipt ~/logs/qmf6/b2-replay.log.
- plate.json replay: unchanged (420 tets, SJ 0.01006).
- induction qmf-ih.json: still FacetRecovery constraint 8461; detail: support triangle at
  (0.01383,0.02662,0)/(0.01383,0.02662,-0.003)/(0.01345,0.02612,-0.00187) "candidate reported an unknown recovered
  segment conflict" (try_insert_support_node fallback after try_insert_on_protected_facets rejected). Not started.
- size lib: 55/56; the one failure was a wrong expected value in my new test (used linear instead of geometric
  size interpolation); fixed in 53f4c8b336, rerun queued (b3).

### Baselines (c13 release CLI, ~/runs/qmf6/base*/summary.txt and *.json receipts)
- cabin D3: InvalidGeometry vertex pcurve, 1 s. Cause: curve_kernel.rs hard-codes pcurve_absolute_error = 1e-10
  (UV units) for the vertex UV agreement check; real STEP vertices carry tolerance_m. Planned fix: admit by 3D
  separation under the source vertex tolerance (needs a surface evaluator in the curve stage, or derive a UV
  tolerance from the vertex tolerance). The new diagnostic (9aec9ae3c1) prints the UV difference: see dbg1/cabin.out.
- pin v1 8/0.4 D4: OrderElevation "exact midpoint leaves its source trim domain", 101 s. Same pattern:
  DelaunaySolverTopologyOptions.trim_boundary_tolerance_uv = 1e-10. dbg1/pin-v1.out will show which path
  (curve image classified Outside vs surface UV midpoint outside a concave trim). Planned fix: curve midpoints
  skip trim classification of their own pcurve images (they lie on the trim by construction) and keep the 3D
  agreement check; surface midpoints outside the trim fall back to the closest surface point of the chord midpoint.
- muffler D2: this time TimeBudgetExceeded in Tetrahedralize after 3665 s (2026-10-29 run: sliver at 1707 s).
  Rerun with the 53f4c8b336 CLI next (same sliver class as the cantilever: 3 boundary nodes, refinement objective).
- blade 3 mm/0.05 D5: TimeBudgetExceeded in OrderElevation after 4068 s. Blade 4 mm/0.15: ok, 145,673 elements,
  1878 s, of which OrderElevation 1633 s, tetrahedralization 89 s, surface ~128 s. D5 = Tet10 elevation speed.
- corpus (1 h limit): twobox 2156/16 s, pcm 64753/24 s, cantm 1497/60 s, cants 1510/34 s, cantc FAIL, cantilever
  317798/96 s, cavity 200304/91 s, pipe 270088/290 s, heatsink 10/0.1 mm TIMEOUT 1 h, bracket 126089/104 s,
  bracket-t10 126089/1148 s; tube/cht/hv still running. square-plate is not a solid (dropped).

### Detached jobs on runmat-build-mac (nohup; my PIDs)
- 75739 chain1: batch b2 (PID 34394: b2-tet, b2-curve -> ~/logs/qmf6/b2.out, b2-*.log), then meshq dbg1 with the
  debug CLI ~/targets/q-mesh-fix-6/debug/runmat (= 53f4c8b336 + dump hook) on cabin, pin-v1, cantc ->
  ~/runs/qmf6/dbg1/summary.txt (+ volume dumps in ~/runs/qmf6/dumps). Expected ~1 h after the gate admits it.
- 19780 chain2: after dbg1, batch b3 (size lib rerun) -> ~/logs/qmf6/b3.out.
- 95844 corpus baseline meshq (tube, cht, hv left) -> ~/runs/qmf6/base-corpus/summary.txt.
- ~/targets/q-mesh-fix-6 (17 GB) is kept for the successor (incremental debug CLI); delete it when done.

### Next steps for the successor
1. Read b2.out (tetrahedron lib must stay 586/0; curve lib), b3.out, dbg1/summary.txt.
2. If tetrahedron lib regresses, check goldens that depend on non-uniform size (volume_refinement tests,
   quality_replay compatibility digests) and over_resolved_boundary fixtures.
3. D3 and D4 fixes as planned above (both are 1e-10 UV absolute tolerances); regression tests in
   curve discretize/vertex_uv/tests.rs and order_elevation geometry tests.
4. Induction: make the on-facet Steiner path (try_insert_on_protected_facets) admit the support node at the
   constraint-8461 triangle; the replay qmf-ih.json reproduces in 40 s.
5. D5: profile OrderElevation on blade4 (1633 s for 146k Tet10) with `sample`; bracket-t10 shows the same 10x.
6. Corpus + muffler + blade with one fea-release CLI once the code is final.
