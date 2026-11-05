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

## q-mesh-fix-7 hand-back (2026-11-03, early: asked to hand back)
Uncommitted working-tree changes (not yet compiled green; h3 chain verifies them):
- size point_field.rs: integrated_edge_length orders endpoint lengths (bit-identical both directions) -> size lib 56/0 (~/logs/qmf6/h1.out).
- D3 (curve): SharedCurveEvaluationContext + discretize_shared_curves / _partition / validate_shared_curve_geometry take
  `surfaces: &dyn ExactSurfaceEvaluator`; vertex_uv.rs admits incident pcurve ends by model-space separation <= max(vertex
  tolerance_m, geometry error) when UV differs > 1e-10; closed-edge endpoint check same rule; validation re-derives face
  uses with face_uses_for_parameters. New test vertex_uv::tests::incident_pcurves_agree_within_the_source_vertex_tolerance_in_model_space.
- D4 (tetrahedron order_elevation/geometry/evaluation.rs): curve midpoints no longer trim-classify their pcurve images
  (3D agreement kept); surface midpoints outside the trim fall back to the closest surface point to the chord point
  (closest_trimmed_chord_point). No regression test yet: needs pin-v1 evidence first.
- Induction diagnostic (facet_recovery/support_refinement.rs): segment-conflict rejection names node, edge, split
  rejection and on-facet rejection.
Debug harness (uncommitted, applied by sync.sh): work/q-mesh-fix-6/qmf_mesh.rs -> crates/runmat-cli/examples/qmf-mesh.rs
(calls mesh_exact_geometry below CLI admission) + qmesh.sh runner, cases-h1.txt (cabin, pin-v1, cantc).
Running: build-Mac PID 65960 chain h3 (build example, curve lib, induction replay -> ~/logs/qmf6/h3.out; then harness
cases -> ~/runs/qmf6/h3/summary.txt).

## q-mesh-fix-8 (2026-11-03, in progress)
Commits: aab5264ec5 D3 (vertex pcurve ends agree in model space; size symmetric fix), abde500c6b D4 (curve midpoints skip
own-pcurve trim classification; closest trimmed chord point), 11a1e9a09d induction diagnostic, f1f20f9f69 merge t1-transplant.
Lib tests h4 (aab5264ec5 tree): curve 43/0, size 56/0, surface 103/0/1 ign, execution 131/0, tetrahedron 586/0/7 ign.
Release CLI ~/fea-cli/runmat-qmf8-f1f20f9f69 (signed in): cantc ok 2012 el 10 s (was fail 43 s); pin-v1 still OrderElevation
trim (surface edge on planar face, chord midpoint outside trim); cabin passes curve stage, lost to server outage at 340 s.
Uncommitted (testing h6/h7): completion.rs boundary pcurve images agreement-only; surface midpoints outside trim within the
chordal deviation of the chord point are admitted; conforming split falls back to complete legalization; new test
tet10_curve_midpoints_do_not_trim_classify_their_own_pcurve_images (BOUNDARY_OUTSIDE_EVALUATOR fixture).
Induction: topology before the failed insertion WAS constrained-Delaunay (h5) -> insertion procedures leave non-CD faces.
Harness note: qmf-mesh debug build is too slow for site CAD; use release CLI runs (meshq.sh) now that sign-in works.
- b562a3f8e8 D4 follow-up committed (completion agreement-only; near-chord surface midpoints; regression test). tet lib 587/0.
- 723318a06a merged integration/0.7.1 (f81788a4a8).
- Induction (uncommitted, h11): conforming segment split now uses a constrained Bowyer-Watson cavity with correction
  (barrier faces, split-child edges, cone visibility incl. near-flat cones), falling back to the edge star.
  Complete legalization as fallback was tried: same failure, 10x slower -> reverted.
- D5 (uncommitted, h11): profile of bracket-t10 worker (release CLI, ~/runs/qmf6/prof/w1.txt): single core, ~85% of busy
  samples in meshing_core tet10_element::integrate (shared rule stiffness quadrature 8^3+12^3(+16^3) points x 900
  moments). Shared rule now runs via runmat_cpu_execution::ordered_map over elements (bit-identical; first error in order).
- Muffler with qmf8-f1f20f9f69 CLI: reaches OrderElevation at 1585 s (was TimeBudget in Tetrahedralize at 3665 s), then
  "curve and surface evaluations disagree" -> uncommitted: curve/pcurve agreement uses max(request tolerance, ExactEdge
  tolerance_m), error names the edge, points, distance.
- Run retry wrapper ~/runs/qmf6/meshq-retry.sh (server flaps; re-runs only ServerUnreachable cases).
- P0 hero (m01-module, 22 boxes shared, Tet4 8 mm): 830c112746 hull-edge on-hull detection fix + lattice regression test
  (fails before, passes after). Sent to q-hero-pack (aa13b1c95980ad3ee). 9b3647d5cd D5 parallel shared rule;
  9d9ed859b1 edge tolerance for curve/pcurve agreement (muffler); f68e28c024 diagnostics. tet lib 588/0 (p2).
- Induction cavity work parked: work/q-mesh-fix-6/induction-cavity.patch (blocked by a near-flat sliver lying in the
  facet plane at the split segment; complete-legalization fallback did not help).
- bracket-t10 with qmf8-f1f20f9f69: 475,767 elements / 2066 s vs c13 126,089 / 1148 s -> check sizing change (53f4c8b336).

## q-mesh-fix-9 (2026-11-04, handed back early on request)
- Merged integration/0.7.1 (7507f51f46) -> afdc02c877.
- Hero P0 "neighbor strictly inside its circumsphere" = point-set hull bug, already fixed by 830c112746 (hero CLI
  0a97eae0da carries it as ec4a5a07fa). With it, the m01 replay (dumps/hero/volume-input-14618-0.json) passes point set
  and segment recovery, then fails in facet recovery.
- Fixed: facet_recovery/cavity.rs facet_intersection_seed scanned EVERY tetrahedron with exact predicates per cavity
  attempt (charging search steps) -> "facet topology-search limit exceeded" at constraint 8566 (p6, 888 s). Now a
  face-adjacency walk from the facet vertices bounded by an open box around the facet (same result set, proof in doc
  comment). Lazy PlaneSigns (q-mesh-fix-8 WIP) fixed for lifetimes. p7: budget error gone.
- Real failure now (p8, 847 s): constraint 8566, facet on z=0.026 (TIM bottom / lid top, 3 mm TIM layer): temp support
  segment breaks a recovered segment edge; splitting that edge leaves an unflippable illegal link face in the TIM;
  complete legalization leaves it too. Same class as the 4-cell study (constraint 711, z=0.026) and the induction case.
- Uncommitted->committed in this round: local conforming split (legalize_inserted_node_and_check_star: link flips +
  exact star check in a cavity_trial, fallback to complete path); better conflict diagnostic; parked
  induction-cavity.patch applied as a FIRST strategy (SplitStrategy::ConstrainedCavity) with edge-star fallback.
  The cavity strategy is NOT yet verified (p9 tet lib + p10 hero replay were running at hand-back).
- Single-threaded: shared interfaces join all 22 solids into one constraint component; one CDT per component is serial.
  Time split (p6-p8 samples, debug replay): point set ~6 min, segment recovery ~4 min (full CD validation after each
  temporary segment), facet recovery O(N) per conforming split (from_canonical/freeze/snapshot per canonical_call).
- Repro subsets (from q-hero-pack-2): ~/runs/qmf6/in/m01-sub-a.step (TIM+lid), m01-sub-b.step (8 solids),
  m01-module-lid.step (lid10, memory runaway 8.7 GB), m01-4cell.step (c4). Harness cases ~/runs/qmf6/cases-p9.txt.
- Detached on build Mac: chain-p9.sh (build qmf-mesh example, tet lib -> ~/logs/qmf6/p9.out; harness subsets ->
  ~/runs/qmf6/p9/summary.txt, volume dumps ~/runs/qmf6/dumps/p9), then chain-p10.sh (hero replay -> ~/logs/qmf6/p10.out,
  samples prof/p10-*.txt).

## q-mesh-fix-10 (2026-11-04, handed back)
Commit: see `git log -1` on q-mesh-fix-6 ("Facet recovery: in-place support-edge, segment-split and facet-patch paths ...").
- p9 (46a1d228f5): tet lib 588/0/7 ign. p9 subset harness used a STALE qmf-mesh binary (example build used
  `-p runmat-cli`; the package is `runmat`), so p9 has no subset evidence. Fixed in p11.steps (`-p runmat --example qmf-mesh`).
- p10 hero replay (46a1d228f5): fails at constraint 8566 after 867 s, peak RSS 14.0 GB (~/logs/qmf6/p10-rss.txt).
  The constrained cavity of the midpoint swallows the whole star of another recovered segment edge.
- Profile p8-4/p11-3: facet recovery time is dominated by canonical_call snapshots (snapshot_parts, about 80-96% of
  samples), first from the per-constraint facet-patch cavity, then from conversions.
- This round (committed, tet lib 588/0/7 ign at ~/logs/qmf6/p11.out):
  1. FacetRecoveryWorking keeps the canonical form until an edit touches the mutation; topology is private behind
     topology()/edit(). canonical_call takes providers' topology directly (no second adopt/from_canonical).
  2. Exact local admission: insertion::observed_change_violation checks every face of every created cell (the
     validator's insphere/region rules) and the protected faces of removed cells, using the new
     mutation::EdgeChangeObserver::changed_cells.
  3. In-place O(change) paths run in cavity trials with rollback, in this order: the facet-patch cavity
     (try_recover_facet_patch_in_place; the analysis in cavity.rs/sides/star/footprint is now generic over
     TetrahedralTopology, and PlaneSigns caches lazily in a map instead of a node-count vector); support-edge local
     flips; the temporary support segment (one pass, link flips, admission); and the conflicting segment-edge split.
     A declined in-place step falls back to the old canonical provider. Final errors append "in-place path: ...".
  4. In-place split cascade: if the midpoint's constrained cavity holds a whole recovered-segment-edge star
     (encroachment), split that edge first (depth <= 8). If the two edges share a vertex (acute corner, here 46
     degrees), split both atomically at the same distance from the vertex (recovered_segment_point_on: nearest dyadic
     parameter). The other edge uses its edge star; the conflicting edge uses the constrained cavity.
  5. local_flip_can_connect_endpoints no longer allocates a node-count vector per call. Removed the stray
     conforming_split.rs.orig.
- p11 hero replay (this commit): still fails at 8566 after 985 s, peak RSS 15.5 GB (~/logs/qmf6/p11-rss.txt; debug).
  The in-place pair split now gets to admission, then fails: the cell [(-1.00725,.28743,.029), (-1.00597,.28508,.026),
  (-1.00725,.28853,.029), P=(-1.00820,.28721,.026)] has the neighbor apex Q=(-1.00597,.28817,.026) (the shell point on
  the other edge) strictly inside its circumsphere. Link flips (2-3 only) cannot remove it.
  Wall time did not improve: the remaining canonical calls (try_recover_facet_with_cavity, refine_support_triangle_interior,
  and canonical fallbacks after in-place declines) still snapshot, and the first in-place declines re-run on the
  canonical path. Profile samples are in ~/runs/qmf6/prof/p11-*.txt (the last run is the newest set).
- Next steps:
  a. In the pair split, insert Q with the constrained Bowyer-Watson cavity as well (not edge star). Or add 3-2 and 4-4
     flips to legalize_inserted_node: the current Lawson loop only does 2-3, so non-convex link faces stall.
  b. Move try_recover_facet_with_cavity and refine_support_triangle_interior onto the working mutation. They are the
     remaining snapshot sources. The pattern is plan on the generic topology, then replace_cavity in a trial with
     observed_change_violation.
  c. Fix 8566, then add a regression test from a sub-a or sub-b dump (RUNMAT_DEBUG_VOLUME_DUMP_DIR with the qmf-mesh
     harness), run release CLIs on m01 at 8 mm and on lid10, and notify q-hero-pack-2 (a3654a9036a5ac106).
- Harness: chain-p11.sh, p11.steps, rssmax.sh in ~/runs/qmf6 (hero replay with peak RSS; qmf-mesh example; subsets).

## q-mesh-fix-11 (2026-11-04, handed back; WIP, constraint 8566 still fails)
Base 0d5790e4ef. One commit on q-mesh-fix-6 (see git log -1). Changes:
1. Link flips: `legalize_inserted_node` takes `LinkFlips::{TwoToThree, Complete { kept_edges, facet_flips }}`.
   Complete adds exact 3-2 and 4-4 flips (Joe's set) on the edge of an illegal link face that segment p-d passes;
   2-3 is preferred anywhere in the link. No flip removes a kept edge (recovered segment edges, split halves), a
   protected face or a region boundary, with one exception: with `facet_flips`, a 4-4 flip may replace the two
   coplanar protected triangles x-y-p, x-y-d by p-d-x, p-d-y (2D Delaunay flip inside one facet; x-y is not a
   segment edge, so both triangles are in one facet). Reported as `FacetFlip`; callers turn it into
   ProtectedFacetSplit records (first source -> both new triangles, oriented like it; second source -> none).
   The Bowyer-Watson path keeps 2-3 only. Facet flips are on only in the pair split and split_and_admit_locally.
2. Pair split at a shared vertex legalizes each node before the next; the second node uses the edge star when its
   constrained cavity would swallow the whole star of a kept edge (that is what lost the half in p12).
3. In-place facet cavity recovery (cavity/in_place.rs): plan on the working topology; direct, pair, progressive,
   exact and batch Steiner refills as cavity trials with exact local admission. Wrapped in
   `FacetRecoveryWorking::atomic` (detached nested trial: rolls back topology, segments and Steiner ids), so the
   canonical fallback starts from the original state (no orphan Steiner nodes or identity collisions).
4. In-place facet-interior refinement (refine_support_triangle_interior_in_place); canonical only on decline.
5. canonical_node_inventory duplicate-position check was O(N^2) per Steiner refill; now one ordered set.
6. execution.rs: protected-facet splits now also update the current constraint's recovered `support` triangles
   (apply_splits_with_current); before, only earlier facets and protection were updated.
7. Unit tests (insertion/mutable/tests.rs): Complete flips restore global Delaunay after 40 edge-star splits of a
   48-point set where 2-3 alone stalls; Complete flips never remove kept edges.

### Results
- Tet lib: 590 passed, 0 failed, 7 ignored (588 + 2 new) on the final code (~/logs/qmf6/q4.out, q4-tet.log).
- p12 (code = items 1,3,4,5 without facet flips / edge-star fallback):
  - hero replay: still fails at 8566 after 755 s (was 985 s); peak RSS 20.0 GB (was 15.5 GB)
    (~/logs/qmf6/p12h.out, p12-hero.log, p12-rss.txt; samples ~/runs/qmf6/prof/p12-*.txt). New reason: the pair
    split lost the half V'-Q (P's constrained cavity swallowed its star) -> fixed by item 2, not yet rerun.
    The RSS increase is unexplained (look at the p12 samples first).
  - sub-a (qmf-mesh, 0.008/0.0002 tet4): FacetRecovery UnsatisfiableConstraint at constraint 1869 after 284 s
    (~/runs/qmf6/p12/suba.out). Pre-existing: the sub-a volume dump (~/runs/qmf6/dumps/p12/volume-input-54192-0.json)
    replays to the same failure on 0d5790e4ef (46 s) and on p12 (36 s) (~/logs/qmf6/suba-base.log, suba-new.log).
    sub-a replay is the fast reproducer (~35 s debug).
  - sub-b: killed by the 8 GB physical guard after 1315 s (~/runs/qmf6/p12/subb.json).
- suba-r2 (item 2 added): still 1869. Cause: in the facet plane z=.029 a segment split leaves protected triangle ABC
  as a sliver (B 3.9e-5 m off line AC) and the coplanar apex E of protected ACE lies in ABC's circumcircle; the facet
  triangulation is not 2D Delaunay, so no CDT holds both. Item 1's facet flips target exactly this; the sub-a replay
  with them (q4) never ran: the gate held it for over an hour at 87-98 GB free (floor 100 GB; others' targets).

### Next
a. Rerun `~/runs/qmf6/suba-new.sh <tag>` (sub-a replay, ~35 s) on this commit. If 1869 passes, run the hero replay
   (`batch.sh p12h` with rssmax/sampler as in chain-p12.sh). If FacetFlip records break final facet validation,
   check the coverage rule there (a flip child is not inside its single source triangle).
b. If the hero passes 8566: regression test from the sub-a dump, release CLIs on m01 8 mm and lid10, notify
   q-hero-pack-2 (a3654a9036a5ac106).
c. Explain the 15.5 -> 20 GB RSS rise (p12 samples).
- Disk: removed my stale builtins/runtime/fea artifacts (Nov 3) and ~/targets/q-mesh-fix-6-base; my target is ~27 GB.

## q-mesh-fix-12 (2026-11-04, handed back; WIP, sub-a 1869 and hero 8566 still fail)
Commit 6f6abddf59 on q-mesh-fix-6 (base 2312ef4919). Sub-a replay: `~/runs/qmf6/suba-new.sh <tag>`.
Debug prints used for the traces below: $J/tmp/work/q-mesh-fix-6/qmf12-debug-r15.patch (full diff vs 2312ef4919 at r15 incl. QMF_DEBUG eprintlns; not committed).

### What 1869 really was (sub-a, z=.029 = TIM top, on the convex hull)
- Each surface triangle is its own constraint, and every surface triangle edge is a segment.
- Cascade at 1869 splits side-face segment D-F (D=(-1.073,.3234375,.029), 4.9 mm). Its midpoint
  encroaches in-plane segment D-C (2.5 mm, 27 degrees at D) -> shared-vertex pair split.
- Old pair split put the D-C point at the long edge's midpoint distance: B at 96.5% of D-C, 8.9e-5 m
  from C. Child triangle A-B-C is a sliver; E (vertex of the coplanar surface triangle A-C-E across
  segment A-C) lies in its circumcircle. A-C is a kept segment and A-B-C, A-C-E are different
  constraints, so no facet flip applies (the q-mesh-fix-11 4-4 facet flips never fire here: hull,
  two-cell star, different facets).
### Changes in 6f6abddf59
1. Pair split shell distance = half the shorter edge (shared_vertex_split_points; unit test
   shared_vertex_pair_split_stays_inside_the_shorter_edge).
2. Blocked-segment splits (encroached_corner_segments): a segment edge at a split endpoint whose
   opposite coplanar vertex lies in the new child triangle's circumcircle is split at its midpoint in
   the same trial; bounded cascade (64). Diametral-ball encroachment was tried first: it spreads over
   the whole top face (>64 splits) because the surface mesh has obtuse triangles; the circumcircle
   test needs 5 splits here. Extras are split FIRST (deepest first) by constrained cavity: split last,
   the pair nodes' cavities swallowed the still-kept extras and fell back to edge stars, whose link
   flips stall.
3. Link flips: 2-2 hull flip (both boundary triangles protected = facet flip, or both unprotected);
   4-4 facet flip may span two regions across the facet plane; star scan for planar illegal faces
   through the inserted node (no link flip reaches them). Unit test
   boundary_facet_flip_repairs_an_illegal_face_through_the_inserted_node.
4. Encroachment cascade stops (tries edge star) when a new swallowed edge shares an endpoint with an
   edge it already split: r15 showed a Zeno loop (halves shrinking toward one vertex until "retries
   exhausted").
5. Memory: canonical_call releases the working mutation while the provider runs (rebuilt from the
   result anyway; all callers end on provider Err); finish_conforming_insertion no longer clones the
   whole input topology.
### Results
- Tet lib 592/0/7 ign (~/logs/qmf6/r16.out) before item 4; rerun in chain-r6 (r6.out).
- Sub-a progression (~/logs/qmf6/rN-suba.log): r1 (2312ef4919) 1869 pair admission; r5-r14 pair
  admission at successive cells; r15 pair split accepted, then support-edge cascade Zeno loop; r17
  (6f6abddf59): runs 332 s, then FacetRecovery InvalidTopology inside a canonical segment recovery:
  a cell on the side face x=-1.073 has a coplanar Steiner apex in its circumsphere
  (P=(-1.073,.3194572,.0287141), R=(-1.073,.315625,.029), S=(-1.073,.3165970,.0288839), W=(-1.08,..,.026)).
  So some earlier in-place edit left a planar non-Delaunay pair that local admission did not reject
  (or the canonical validator applies different protection). Next: find the edit that created those
  cells (trace replace_cavity of cells containing P and S), and compare observed_change_violation with
  validate_constrained_delaunay_volume_topology on coplanar apexes.
- Tet lib on 6f6abddf59: 592/0/7 ign (~/logs/qmf6/r6.out).
- Hero replay on 6f6abddf59 (~/logs/qmf6/r6h.out, r6-hero.log, r6-rss.txt, prof/r6-*.txt): still fails at
  8566 after 761 s (p12: 755 s); peak RSS 17.95 GB (p12: 20.0 GB). New reason: the support-edge cascade on
  z=.026 reaches a shared-vertex pair split whose admission finds cell [(-1.0048516,.2916515,.026),
  (-1.0059688,.2909439,.026), (-1.0061875,.2921875,.029), (-1.005125,.2918823,.029)] with apex
  (-1.0056562,.2913177,.029) in its circumsphere (spans the 3 mm TIM; apex coplanar with the two z=.029
  corners). The canonical fallback fails validation with the same pattern (two z=.026 + two z=.029
  corners, coplanar apex). Both sub-a (side face) and hero now fail on cells whose apex is coplanar with a
  face plane of the cell's other corners: check whether the in-place admission and the canonical
  validator treat coplanar (insphere-degenerate) apexes the same, and whether a pending (not yet
  recovered) facet triangle in that plane needs a planar hull/facet flip that link flips miss.
### RSS 15.5 -> 20 GB (p12)
- Peak is at the very end (14:24:52; sample 25 s earlier showed 7 GB footprint): the failing
  constraint's canonical fallback (try_split_recovered_segment_edge -> insert_conformingly_with:
  from_canonical(clone) + freeze + legalize + validate) runs while the working mutation and its
  retained canonical copy are alive, i.e. 2 mutations + ~4 canonical topologies at once. A
  TopologyMutation (face_uses/tetrahedron_indices/vertex_stars B-trees) is several times a canonical
  topology. Item 5 removes one mutation and one canonical copy from that peak; ConstrainedCavity then
  EdgeStar run the whole sequence twice (sequential, not additive).

## q-mesh-fix-13 (2026-11-04, handed back on request; WIP)
Commit 2d79f1744d. Debug variant: $J/tmp/work/q-mesh-fix-6/qmf13-debug.patch (QMF_DEBUG_C=<constraint> prints;
harness writes full error to QMF_ERROR_FILE). Scripts: ~/runs/qmf6/suba-dbg2.sh <tag> (QMFC=<constraint>).
- 1869 root cause: the support refinement of facet X=[D,F,R] placed temporary-segment chain node S inside the
  circumcircle of an already recovered (protected) sliver support triangle PQR of the same facet (fan splits make
  slivers). With PQR a barrier, no CDT holds S; the canonical tseg legalization then fails validation. Input to the
  tseg was valid, so admission and validator agree; the producer was the support bookkeeping.
- Fix: suspend encroachable current-facet recovered triangles during a refining step, re-protect survivors, and
  rebuild pending support by 2D Bowyer-Watson insertion of the new nodes (facet_delaunay.rs, 3 unit tests).
  Sub-a now passes 1869 in 16 steps (~40 s to the next failure).
- Next sub-a failure: constraint 11867 (z=.026 facet, TIM layer). Split of bottom segment C-E ping-pongs with its
  twin top edge C_top-E_top (3 mm apart, 6 mm long: each midpoint swallows the other's star). Added: twin pair
  split (both midpoints in one edit) and blocking-segment split; the twin pair split still declines: cell
  [C, M_top, M, A] has apex X(z=.024) in its sphere across face C-M-A (z=.026 plane, pending). The added 2-3 flip
  on faces through the inserted node did not repair it; next: diagnose why (2-3 geometry vs 3-2/4-4 on edge M-A,
  kept halves) — likely the same family as the hero 8566 pattern (cells spanning the TIM).
- Tet lib + hero chain on this commit started (~/runs/qmf6/chain-q13.sh; ~/logs/qmf6/q13.out, q13h.out, q13h.err,
  q13-rss.txt, q13-chain.done).
