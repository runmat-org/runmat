# Metadata audit cutover

`metadata_view.rs` projects immutable canonical catalog entries and every legacy
registration. It does not invoke a builtin or create an alternative inventory.
Canonical entries come first for settled-name lookup; legacy entries remain
subject to all per-definition checks. A separate test rejects conflicting
duplicate metadata. This avoids the linked-runtime/TLS ambiguity of unit tests
that dispatch through another runtime library.

The exhaustive expected sets are fixed reviewed literals: the original inline
baseline plus the sorted cutover receipts in `reviewed/`. Tests never compute
their expectations from the live catalog or filter unexpected names out.
Differences require source-owner review, not automatic acceptance of failed
output. The original additions below were checked against source at parent
revision `4383723333`.

## Extension expectation provenance

All paths below are relative to `crates/runmat-builtins/src/catalog/entries/`.
Each listed identifier is the literal declared extension ID.

| Owner path | Reviewed identifiers |
| --- | --- |
| `math/reduction/bounds/contract.rs` | `bounds-extra-outputs` |
| `cells/core/cellstr/contract.rs` | `cellstr-cell-input`, `cellstr-symbolic-input` |
| `math/linalg/factor/decomposition/integer_inputs.rs` | `decomposition.mrdivide`: `decomposition-integer-scale-factor`, `decomposition-complex-integer-scale-factor` |
| `io/repl_fs/directory_listing/dir/contract.rs` | `dir-folder-pattern` |
| `math/discrete/factorial/mod.rs` | `factorial-like-residency`, `factorial-logical-input` |
| `io/repl_fs/path_syntax/fullfile/contract.rs` | `fullfile-numeric-character-codes` |
| `io/repl_fs/genpath/contract.rs` | `genpath-excludes`, `genpath-numeric-character-codes` |
| `io/repl_fs/directory_listing/ls/contract.rs` | `ls-portable-row-layout` |
| `io/repl_fs/path/mod.rs` | `path-numeric-character-codes` |
| `structs/core/rmfield/contract/extension.rs` | `rmfield-variadic-field-arguments` |
| `math/rounding/round/mod.rs` | `round-decimal-mode-alias` |
| `io/repl_fs/savepath/contract.rs` | `savepath-diagnostic-outputs`, `savepath-directory-target`, `savepath-numeric-character-codes` |
| `io/repl_fs/environment/setenv/contract.rs` | `setenv-status-outputs` |
| `structs/core/setfield/contract/extension.rs` | `setfield-indexed-resident-field`, `setfield-object-family`, `setfield-textual-index` |
| `strings/transform/strjoin/contract.rs` | `strjoin-rowwise` |
| `io/repl_fs/environment/unsetenv/contract.rs` | `unsetenv-status-output` |

Bounds, the two mrdivide declarations, and strjoin account for the four newly
explicit L03 expectation differences. The other20 were existing baseline
expectation omissions. Runtime acceptance and strict-profile gate tests remain
independent of these metadata checks.

## Integer audit expectation provenance

The29 added names already declare audit metadata in the following owners:

| Authority | Names |
| --- | --- |
| Legacy `crates/runmat-runtime/src/builtins/interop/native_ffi.rs` | calllib, libfunctions, libisloaded, libpointer, libstruct, loadlibrary, setdatatype, unloadlibrary |
| Legacy `crates/runmat-runtime/src/builtins/interop/java.rs` | isjava, javaArray, javaMethod, javaMethodEDT, javaObject, javaObjectEDT, javaaddpath, javaclasspath, javarmpath, jenv, usejava |
| Legacy `crates/runmat-runtime/src/builtins/interop/mex.rs` | mex, mexcuda, mexext |
| Legacy `crates/runmat-runtime/src/builtins/interop/python.rs` | pyargs, pyenv, pyrun, pyrunfile, terminate |
| Canonical `io/repl_fs/directory_listing/{dir,ls}/mod.rs` under the catalog entries root | dir, ls |

`strings/core/genvarname` and `introspection/superclasses` retain canonical
integer audit declarations, so their existing expectations remain.
`strings/transform/strjoin` now declares integer capabilities for its exact
scalar-integer delimiter extension and `integer_audit: None`; only its obsolete
NotApplicable expectation is removed. The settled-API test now sees canonical
`stats/summary/corr/integer.rs` and all other canonical capabilities instead of
failing an obsolete legacy-registration lookup.

No production behavior changes. Source formatting and diff checks do not qualify
these tests: execute the metadata projection, settled capabilities, integer
audit dispositions and exhaustive extension tests on the integrated revision.
Unrelated generated-document parity tests are not repaired by this packet.

## RM-1290 integrated inventory review

The 0.7.0 integration contains declarations added after the inline baseline.
The fixed receipts contain 576 audit names and 205 builtin/extension pairs
(71 distinct extension IDs). They supplement the baseline; they do not replace
the raw registration, duplicate-consistency, audit well-formedness, extension
error-contract, compatibility-gate or generated-product tests. No production
contract is changed to make these expectations pass.

Audit additions were reconciled to literal catalog identities and their shared
declaration owners: 534 typed FEA members, 22 geometry members, nine GPUDevice
members, three FEA/geometry method bindings, and eight meshing accessors. FEA
closed-vocabulary/no-argument and structural-object contracts live in
`src/fea_contract/{integer,policy_types/integer,policy_choices/integer,study_types/integer}.rs`;
numeric physical values and exact counts retain their separate positive
integer capabilities rather than being classified inapplicable. Geometry and
meshing entries use their catalog-owned method/accessor contracts. GPUDevice
members take an issued handle, not an integer resource ID; their audit has no
alias target. The incoming FEA surface replaces the old `fea.authorStudy`,
`fea.plot` and `fea.step` audit spellings with the typed study/result and method
surface; those three obsolete baseline audit names are removed, not filtered
from live discovery. This review does not requalify FEA algorithms.

Extension receipts were checked against the declarations under
`src/catalog/entries/math/optim/` and `math/ode/odeset/`: GA's shared descriptor
and creator/crossover/mutation/scaling/selection/plot entry owners; fmincon,
fminimax, fgoalattain, fseminf, fminunc, linprog, intlinprog, lsqlin, lsqnonneg,
quadprog, patternsearch, particleswarm, simulannealbnd and surrogateopt; and
odeset's exact/resident option payload declarations. OptimizationProblem.solve
and EquationProblem.solve retain the extension unions declared by their actual
selected-solver entry owners. Integer structural controls, exact numeric
crossings, single-to-double forms, resident gathers, sparse solver forms and
the numerical-failure exit status remain distinct IDs. This is an inventory
receipt, not a new claim of MATLAB parity or numerical acceptance.

The retired `gpudevice-provider-info` facade gate is not retained as a live
extension expectation. The device contract exposes issued physical lifecycle
handles and retains only its separately declared logical-selector extension;
the device catalog tests check that retirement explicitly.

The fixtures are fixed embedded test inputs and require no filesystem service.
Keep them sorted and duplicate-free. Tests never rewrite them; updates require
source-owner review. Their
first matching execution is recorded in the RM-1290 qualification ledger; a
source review or passing formatter alone is not acceptance.
