# RunMat 0.7.0 qualification index

Status: Windows qualification is pending. This index does not certify a
release-ready candidate. No push, tag, publication or deployment is included.

## Source and distribution

Branch: `0-7-0-rc`. Product Rust/SDK source is anchored at
`c7a12b7b9f050f6789dda56a1024636243b7a63e`. Release-routing source is
`0d09250a4c9a8394de9ed906953e72dc22c8fb12`; its two validation-only Python
changes and CI changes do not alter product code. Private Server/Desktop
checkpoint: `82e070c074c583a7cc477bc1adc85da66de7a357`.
Original measured sources and artifact identities remain in their receipts.

`node scripts/verify-release-version.mjs 0.7.0` verifies83 workspace crates
and the TypeScript package. The four native release targets in
`.github/workflows/release.yml` match the matrix below. Intel macOS execution
uses the owner-approved Rosetta route on Apple Silicon.

## Product matrix

| Target/product | Result and scope | Evidence |
| --- | --- | --- |
| macOS ARM64 CLI/native | Corrected optimized CLI builds; real remote controls pass. Retained31-case Metal product cohort and native pure/callback application/test pairs pass, including seven semantic controls, installation/loader/notices admission and separate-process reopening. | [Native pairs](products/macos-native-r1055.json), [remote CLI](products/remote-cli-r1057.json) |
| macOS x86_64/Rosetta | Corrected optimized CLI:31 passed,0 failed,0 omitted; eight assessment commands pass. Three25871-file guards and15 collected receipt hashes match. Physical Intel hardware was not tested. | [Intel receipt](products/macos-intel-r1057.json) |
| Linux x86_64/Vulkan | Corrected optimized CLI:31 passed,0 failed,0 omitted. Six25871-file guards and20 collected receipt hashes match. Actual GPU workers, callbacks, cancellation and public native compile/link execute. | [Linux receipt](products/linux-r1057.json) |
| Windows x86_64 | Current optimized build,31-product/eight-assessment cohort, final package and physical Victus7 are pending. Three focused private Win32 publication tests already pass. | Final product receipt pending |
| Optimized WASM/public SDK | Node execution/reset2, Chrome product cases10, boundaries12, input checks5 and exact-CAD assertions32 pass. Supported local named FEA uses real OCCT and supervised numerical workers. | External browser receipt below |
| Desktop caller | Conserved native caller/settings/provider acceptance18/0; GUI Keychain1/0 is user-attested. Remote frontend upload/cluster/cipher tests23/0 and focused ESLint pass. This scope does not include a fresh signed Desktop installer. | External Desktop conservation below |
| Real remote CLI/Server | Positive and failing jobs pass through production HTTP routes and a registered node, preserving encrypted terminal output, original errors, pre-error console and JSON reattachment. Server domain21/0 and affected HTTP test-target checks pass. Synthetic auth/in-memory stores do not establish production identity-provider, durability or deployment qualification. | [Remote receipt](products/remote-cli-r1057.json) |

Each count belongs to its named cohort; some cohorts overlap.

## Retained host, GPU and scientific evidence

| Cohort | Measured result and conservation |
| --- | --- |
| Host execution | Six R1042 cohorts retain1389 passes,0 failures,4 ignored. Numerical-owner changes require their separately measured successor evidence. The R1055 review preserves this scope. |
| Supplementary Metal readback | Four selected hardware-only cases pass,0 failed/ignored;18320 filtered. Their original harness and the R1055 source-conservation record remain attached. |
| Linux GPU | R992 hardware inventory242 passes plus one Metal-only exclusion; ordinary inventory1310 passes/243 ignored. R1012 scatter2/scatter3 execute F32/F64 and72 shader variants validate. Original harnesses are retained at their measured identities. |
| Windows GPU | R929's original normalization failure remains recorded. R936's12-case normalization family passes; combined with230 unaffected original cases,242 supported unique cases pass, with one Metal-only exclusion. F32 uses D3D12; F64 uses Vulkan. |
| Linux thermoelastic | One selected callback/typed-denial case passes,0 failed/ignored,2150 filtered; it includes Tet4/Tet10 variants. |
| Authored CLI physics |12 passed,0 failed/ignored/filtered,2083.94s. Exact command, cases, measured source, harness/CLI identities and log are retained in the [receipt](scientific/receipts/authored-cli-physics.json). |
| Driver recovery physics |18 passed,0 failed/ignored,464 filtered,3343.49s. Exact command, cases, measured source, harness and log are retained in the [receipt](scientific/receipts/driver-recovery-physics.json). |
| Server accounting | Eight copy/seed cases and one actual PostgreSQL initializer pass. Migration0205 conserves18 usage rows and three professional plans. Server terminal-output changes have separate21-case domain and real CLI successor evidence. |

The scientific verifier reconstructs the measured source from the exact
[conservation review](scientific/reviews/release-routing-r1058.json), checks its
supporting evidence hashes and retains the original measurements. It does not
relabel historical executions as fresh product tests.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Corrected ARM64 macOS CLI | `e0252f6fcc8fcd93364079352acfe43c84ad13d9c0b03c1bc64448d6d3594268` |
| Corrected x86_64 macOS CLI | `2acf88b9139b9e51b8afb4321900265eb5e07f3fe6c9e597c622c80252ce5cc6` |
| Corrected Linux CLI | `b45cf823f46b98f7b3c15f3084c878de2f801f53885d4cb232bde8022838e2e5` |
| Final Windows CLI/package | Pending |
| Runtime WASM | `ea55c974dd17ced111157c2c97279afce79fb9ad7c0775bcc8158faa71845fa8` |
| SDK package inventory | `5973f23d2dc9fc55df72c7fab00fea7922c1fda772232fcb210332d514caed24` |
| OCCT WASM | `c4fbd1c5f9c7e0b180486694897b530b60b0109d15efdfc8a08d76e75ae8965b` |
| LSP WASM | `ce3871e33e7bf78f0f242476916a0a356f11915b0c05062817865615d0d78cdb` |

Native receipts preserve runtime archive/payload/manifest/notices identities.
The approved LICENSE.md and FREE-USE.md retain SHA-256 values
`f923b86930cc30ee506959d2e026ae4cef0b29f74c00947540dd80f28e4eae51`
and `0ce48a3efdb6408518cafb1749bcf7a04ff8b07439c254ec7f04aa46768012f6`.
The completed6999-commit license-history verification and independent backups
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

Architecture accepts exactly89 approved FEA findings; the language ledger
covers134 members and its debt matcher passes1/0. FEA Clippy conserves183
identities/273 emissions across109 files; every other first-party crate retains
strict linting. The [quality record](quality/fea-style-debt-0.7.md) binds the
reviewed source/toolchain and original evidence. No new waiver was added.

The release workflow verifies every required scientific receipt and the policy's
explicit exceptions. Five catalog/workflow validators and34 tooling tests pass.
These are metadata checks, not executions of the deferred meshing recipe.

Approved dispositions: governed meshing recipe12 unrun under the November4 owner
decision; extended transient rods under RM-1297; opaque solver/OCCT work-measurement
exclusions; existing-plot temporary property-allocation budgeting; FEA architecture/
style debt; embedded AOT/support-pack/selective-linking continuation; dependency
compiler/driver/browser memory outside the managed cap. Those limitations remain
separate from the passing numerical and product assertions above.
