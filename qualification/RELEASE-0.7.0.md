# RunMat 0.7.0 qualification index

Status: qualification remains open under RM-1293. The current candidate includes
repairs made after the historical product checkpoint documented below. Its
completed and pending gates are listed here; historical receipts retain their
original source and artifact identities. No push, tag, publication or deployment
has been performed.

## Current candidate qualification

Branch: `0-7-0-rc`, committed base
`527f4e5f32e712d502d16b26e98ea9f91f507e22`, with the frozen R1260 repair packet.
The source repairs and final qualification receipts still require their closing
commits. The evidence locations below are relative to the workspace parent.

| Scope | Verified result | Remaining |
| --- | --- | --- |
| macOS ARM64 | Fresh optimized CLI: 31/31 product cases and 3/3 dataset cases. `settings-r1287-main-products-verified.json` under `.tmp-rm1064`. | Final receipt consolidation. |
| macOS Intel/Rosetta | Fresh optimized CLI: 31/31 product cases and 3/3 dataset cases. `.tmp-rm1064/settings-r1292-intel-acceptance-collected/collection.json`. | Final receipt consolidation; physical Intel hardware was not tested. |
| Linux | Fresh optimized CLI: 31/31 product cases and 3/3 dataset cases. Renewed Runtime: 18,269 passed, zero failed; affected strict lint passes. `.tmp-rm1064/settings-r1306-linux-qualified-collected/collection.json`. | Shipping artifact collection and final receipt consolidation. |
| Windows | Renewed Runtime: 18,237 passed, zero failed. Optimized runtime archive completed. | Targeted output conformance, remaining lint/benchmark, fresh CLI and dataset acceptance. |
| Optimized WASM/SDK | Node 2 checks; pinned dataset/publication 1 scenario; Chrome products 10 cases; boundaries 12 checks; input 5 checks; exact CAD 32 assertions. All pass. `.tmp-rm1064/settings-r1302-sdk-browser-qualified.report.json`. | Final receipt consolidation. |

Mac ordinary outcome reconciliation retains 35,437 passes; Linux retains 35,438.
Both have zero unresolved failures, 397 existing ignores and 45 routed cases
across 813 targets. These totals conserve unaffected executions and include
reviewed successors; they are not fresh whole-workspace runs against one binary.
Windows reconciliation remains pending its final successor.

The local dev server serves the qualified SDK with manifest `3af31d7d06523523`.
Seven served runtime, OCCT and LSP files match candidate hashes over HTTP:
`.tmp-rm1064/settings-r1303-served-http.json`. Runtime WASM SHA-256:
`b02b8c33fbf0eba3cbc8976c1455304d2ccd58d2905efc4e132e17d3052c9239`.

## Historical product checkpoint

The following sections describe the earlier checkpoint and its retained evidence.
Use the current-candidate table above for release readiness.

## Source and distribution

Branch: `0-7-0-rc`. Conserved product Rust/SDK source is anchored at
`c7a12b7b9f050f6789dda56a1024636243b7a63e`. Three AOT linker files have a
Windows-only execution-stack successor at
`0b36a6a1da1e3de2d6f8bb95ddac86d2922e6cdf`; its independent Windows build,
product and physical-hardware results are in the [Windows receipt](products/windows-r1059.json).
Release-routing source is
`0d09250a4c9a8394de9ed906953e72dc22c8fb12`; its two validation-only Python
changes and CI changes do not alter product code. Private Server/Desktop
checkpoint: `82e070c074c583a7cc477bc1adc85da66de7a357`.
Original measured sources and artifact identities remain in their receipts.

`node scripts/verify-release-version.mjs 0.7.0` verifies 83 workspace crates
and the TypeScript package. The four native release targets in
`.github/workflows/release.yml` match the matrix below. Intel macOS execution
uses the owner-approved Rosetta route on Apple Silicon.

## Product matrix

| Target/product | Result and scope | Evidence |
| --- | --- | --- |
| macOS ARM64 CLI/native | Corrected optimized CLI builds; real remote controls pass. Retained 31-case Metal product cohort and native pure/callback application/test pairs pass, including seven semantic controls, installation/loader/notices admission and separate-process reopening. | [Native pairs](products/macos-native-r1055.json), [remote CLI](products/remote-cli-r1057.json) |
| macOS x86_64/Rosetta | Corrected optimized CLI: 31 passed, 0 failed, 0 omitted; eight assessment commands pass. Three 25,871-file guards and 15 collected receipt hashes match. Physical Intel hardware was not tested. | [Intel receipt](products/macos-intel-r1057.json) |
| Linux x86_64/Vulkan | Corrected optimized CLI: 31 passed, 0 failed, 0 omitted. Six 25,871-file guards and 20 collected receipt hashes match. Actual GPU workers, callbacks, cancellation and public native compile/link execute. | [Linux receipt](products/linux-r1057.json) |
| Windows x86_64 | Repaired optimized CLI: 31 passed, 0 failed, 0 omitted. Four 25,871-file guards pass. Eight earlier assessment commands and three focused Win32 tests retain their passing evidence. Seven unique physical Victus controls pass against unchanged candidate bytes: six first-run passes plus a console-probe successor that explicitly enables Ctrl-C inheritance, retaining the original assertions and timeout. DXC loading, RTX 3050/D3D12, offline CPU execution, cancellation and all three native images execute. | [Windows receipt](products/windows-r1059.json); initial failures and their dispositions are preserved there |
| Optimized WASM/public SDK | Two Node execution/reset checks, 10 Chrome product cases, 12 boundary checks, five input checks and 32 exact-CAD assertions pass. Supported local named FEA uses real OCCT and supervised numerical workers. | External browser receipt below |
| Desktop caller | Conserved native caller/settings/provider acceptance: 18 passed, 0 failed; one GUI Keychain pass is user-attested. Remote frontend upload/cluster/cipher tests: 23 passed, 0 failed; focused ESLint passes. This scope does not include a fresh signed Desktop installer. | External Desktop conservation below |
| Real remote CLI/Server | Positive and failing jobs pass through production HTTP routes and a registered node, preserving encrypted terminal output, original errors, pre-error console and JSON reattachment. Server domain tests: 21 passed, 0 failed; affected HTTP test-target checks pass. Synthetic auth/in-memory stores do not establish production identity-provider, durability or deployment qualification. | [Remote receipt](products/remote-cli-r1057.json) |

Each count belongs to its named cohort; some cohorts overlap.

## Retained host, GPU and scientific evidence

| Cohort | Measured result and conservation |
| --- | --- |
| Host execution | Six R1042 cohorts retain 1,389 passes, 0 failures, 4 ignored. Numerical-owner changes require their separately measured successor evidence. The R1055 review preserves this scope. |
| Supplementary Metal readback | Four selected hardware-only cases pass, 0 failed/ignored; 18,320 filtered. Their original harness and the R1055 source-conservation record remain attached. |
| Linux GPU | R992 hardware inventory: 242 passes plus one Metal-only exclusion; ordinary inventory: 1,310 passes/243 ignored. R1012 scatter2/scatter3 execute F32/F64 and 72 shader variants validate. Original harnesses are retained at their measured identities. |
| Windows GPU | R929's original normalization failure remains recorded. R936's 12-case normalization family passes; combined with 230 unaffected original cases, 242 supported unique cases pass, with one Metal-only exclusion. F32 uses D3D12; F64 uses Vulkan. |
| Linux thermoelastic | One selected callback/typed-denial case passes, 0 failed/ignored, 2,150 filtered; it includes Tet4/Tet10 variants. |
| Authored CLI physics | 12 passed, 0 failed/ignored/filtered, 2083.94s. Exact command, cases, measured source, harness/CLI identities and log are retained in the [receipt](scientific/receipts/authored-cli-physics.json). |
| Driver recovery physics | 18 passed, 0 failed/ignored, 464 filtered, 3343.49s. Exact command, cases, measured source, harness and log are retained in the [receipt](scientific/receipts/driver-recovery-physics.json). |
| Server accounting | Eight copy/seed cases and one actual PostgreSQL initializer pass. Migration 0205 conserves 18 usage rows and three professional plans. Server terminal-output changes have separate 21-case domain and real CLI successor evidence. |

The scientific verifier reconstructs the measured source from the exact
[conservation review](scientific/reviews/windows-stack-r1059.json), checks its
supporting evidence hashes and retains the original measurements. It does not
relabel historical executions as fresh product tests.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Corrected ARM64 macOS CLI | `e0252f6fcc8fcd93364079352acfe43c84ad13d9c0b03c1bc64448d6d3594268` |
| Corrected x86_64 macOS CLI | `2acf88b9139b9e51b8afb4321900265eb5e07f3fe6c9e597c622c80252ce5cc6` |
| Corrected Linux CLI | `b45cf823f46b98f7b3c15f3084c878de2f801f53885d4cb232bde8022838e2e5` |
| Windows CLI | `c47fdf64fd10d43ec202486cf170f02eceebed7bf7485372b3a3944a71482696` |
| Windows package/native-image transfer archive | `e105e59a49f6fa601e6e0240aa1cbbea7478f4aa78b1dfcb3c725f62ae0a8dab` |
| Runtime WASM | `ea55c974dd17ced111157c2c97279afce79fb9ad7c0775bcc8158faa71845fa8` |
| SDK package inventory | `5973f23d2dc9fc55df72c7fab00fea7922c1fda772232fcb210332d514caed24` |
| OCCT WASM | `c4fbd1c5f9c7e0b180486694897b530b60b0109d15efdfc8a08d76e75ae8965b` |
| LSP WASM | `ce3871e33e7bf78f0f242476916a0a356f11915b0c05062817865615d0d78cdb` |

Native receipts preserve runtime archive/payload/manifest/notices identities.
The Windows archive contains 76 files, all rehashed on Main and Victus. Its
1,007-file physical candidate includes the separately pinned execution tools.
The source repair reserves a 64 MiB Windows execution stack; the preserved
runtime archive is unchanged. The modified-header diagnostic copy is excluded
from the shipped package.
The approved LICENSE.md and FREE-USE.md retain SHA-256 values
`f923b86930cc30ee506959d2e026ae4cef0b29f74c00947540dd80f28e4eae51`
and `0ce48a3efdb6408518cafb1749bcf7a04ff8b07439c254ec7f04aa46768012f6`.
The completed 6,999-commit license-history verification and independent backups
are preserved separately; qualification does not repeat that rewrite.

## External evidence locations

Locations below are relative to Main's workspace parent of this repository.
They contain review records and retained log/artifact locations. Raw credentials,
node state, binaries and customer data are not committed with this index.

| Record | Main location | SHA-256 |
| --- | --- | --- |
| Host conservation | `.tmp-rm1064/settings-r1055-host-reuse.json` | `db723e969924936db35c892da9b41900e6b7f70667edc45d009ac2872c584f7a` |
| Browser acceptance | `.tmp-rm1064/settings-r1055-browser-acceptance.verified.json` | `45d8f7918ddd2f91f2aca989d85a30c60ff91926c40f3f0b236cc5975dca5ede` |
| Linux GPU conservation | `.tmp-rm1064/settings-r1055-gpu-conserved/CONSERVATION.md` | `f96012e191b50e9790a1846e6c3586b0c6ef44d8256e257be9b98fd36cb56019` |
| Windows GPU conservation | `.tmp-rm1064/settings-r1055-windows-gpu-conserved/CONSERVATION.md` | `53bf4e36b49988de6f42fde8e68066b32d68e9b9d4b58d60acec2cba5618060c` |
| Desktop conservation | `.tmp-rm1064/settings-r1055-desktop-component-conservation.md` | `96d5986e626b977630643cab2d41cc50cf6fd316354d2ea22ce3dd4e461cc727` |
| Server accounting | `.tmp-rm1064/settings-server-wording-r1055.receipt.json` | `fb5169312b3e049993bf98a014c44feb1d3e01b05a88ced679cd052a6235cd27` |

## Quality policy and approved dispositions

Architecture accepts exactly 89 approved FEA findings; the language ledger
covers 134 members and its debt matcher passes 1/0. FEA Clippy conserves 183
identities/273 emissions across 109 files; every other first-party crate retains
strict linting. The [quality record](quality/fea-style-debt-0.7.md) binds the
reviewed source/toolchain and original evidence. No new waiver was added.

The release workflow verifies every required scientific receipt and the policy's
explicit exceptions. Five catalog/workflow validators and 34 tooling tests pass.
These are metadata checks, not executions of the deferred meshing recipe.

Approved dispositions: the 12-command governed meshing recipe is unrun under the November 4 owner
decision; extended transient rods under RM-1297; opaque solver/OCCT work-measurement
exclusions; existing-plot temporary property-allocation budgeting; FEA architecture/
style debt; embedded AOT/support-pack/selective-linking continuation; dependency
compiler/driver/browser memory outside the managed cap. Those limitations remain
separate from the passing numerical and product assertions above.

## Build-machine disposition

Linux and Windows qualification VMs are running the current qualification work.
Their persistent disks, warm caches and T4 configurations remain intact. They
will be stopped after qualification and artifact collection. Main holds the
collected Mac artifacts and receipts; the Victus holds its earlier tested
package. Publication remains a separate authorized step.
