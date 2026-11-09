# Qualification evidence and dispositions

Use [CI and release qualification](CI.md) for the required checks and manual
qualification commands. The records in this directory bind test results to
source inputs, executed artifacts and declared exclusions.

## Reading the evidence

Each product receipt identifies the source revision, artifact hashes, commands
and acceptance results it covers. A passing result applies to that recorded
scope. Source conservation can retain a result when a reviewed change leaves
its relevant inputs unchanged; affected behavior needs successor evidence.

The [cross-platform product receipt](products/cross-platform-r1322.json)
collects the earlier native, browser and host results. Its component receipts
preserve the original measurements, including the 31-case native product
cohorts. Later CI runs record their own ordinary-test and staged-product
results against their exact commits. Counts from overlapping cohorts must be
reported separately.

The [scientific policy](scientific/policy.json) declares the required offline
cohorts and their ordinary-CI routing. The scientific verifier checks receipt
identities, evidence hashes and exact source-conservation records. Renew
affected receipts when their inputs change.

## Platform coverage

The product receipts state the hardware and execution route used. Intel macOS
qualification in the cross-platform receipt uses Rosetta on Apple Silicon;
it provides no physical Intel hardware result. GPU receipts identify their
backend and supported precision. Browser acceptance covers the recorded SDK,
WASM and OCCT artifacts. Desktop caller checks cover the recorded integration
boundary; signed installer acceptance requires its own product evidence.

## Recorded limitations

These dispositions belong to the source and scope recorded in their receipts.
Review them when integrating changes to the affected mechanisms.

| Area | Disposition and evidence |
| --- | --- |
| FEA architecture and style | [Reviewed debt](quality/fea-style-debt-0.7.md) records the accepted diagnostic identities and source anchors. CI rejects additions and mismatches. RM-1287 tracks the structural work. |
| Governed meshing | The scientific policy records the approved unrun 12-command recipe. Catalog and metadata validation provide fixture-coverage checks; numerical and performance qualification remains in the manual lane. |
| Extended transient rods | RM-1297 tracks the deferred scientific envelope. Retain the exclusions and measurements stated in the associated receipts. |
| Simulation work measurement | The accounting specification excludes the declared opaque solver and OCCT kernels from reference-work measurement. Their exclusion concerns allowance measurement, not numerical correctness. |
| Plot property memory | Temporary host allocations made by properties of existing plots have a documented finite-memory budgeting gap. Shared property-allocation budgeting remains deferred. |
| Native distribution size | Embedded AOT support and selective runtime linking remain hardening work. Product receipts qualify the distributed layout they actually exercised. |
| External memory | Dependency shader compilers, GPU drivers and browser allocations remain outside RunMat's managed allocation cap. |

Keep release checklists, transfer logs and working coordination notes outside
the repository. Commit durable policies, receipts, conservation reviews and
their required evidence here. Release publication and deployment are separate
operations from qualification.
