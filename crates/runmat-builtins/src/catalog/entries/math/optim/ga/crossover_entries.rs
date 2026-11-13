use super::{
    crossover_inference as infer,
    helper_documentation::{doc, example},
    helper_entries::entry,
    *,
};
use crate::*;
const SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Calling forms",paragraphs:&[
        "children=crossoverFcn(parents,options,nvars,problem,scores,population) pairs consecutive one-based parent indices and returns one host-double child row per pair. population is a full finite real member-by-nvars matrix. nvars is a positive exact structural scalar. problem is the scalar operator context; these predefined crossovers do not call problem.FitnessFcn or nonlinear constraints. options is a scalar context or [] and has no active fields in these standalone numerical calls.",
        "crossoverheuristic reads one finite score per member to choose the better parent; the other crossover functions do not inspect scores. Intermediate and heuristic crossover accept an optional final ratio. Intermediate ratio is scalar or one value per variable, default one; heuristic ratio is scalar, default1.2. Ratios must be finite and nonnegative. Other extra arguments are errors.",
    ]},
    BuiltinDocumentationSection{heading:"Numerical operators",paragraphs:&[
        "Scattered independently chooses each gene from either parent. Single-point uses one random cut; two-point uses two random cuts and exchanges the intervening segment. Arithmetic uses one common uniform convex weight. Intermediate draws a uniform coefficient multiplied by ratio, using one shared draw for a scalar ratio and per-variable draws for a ratio vector. Heuristic extrapolates from the worse parent toward the better parent by ratio; ties favor the first parent.",
        "Laplace uses a centered unit-scale Laplace displacement times the absolute parental difference with a common randomly selected base parent. This is RunMat's distribution policy, not a promise of another implementation's random trajectory. Random draws come from the selected runtime stream. Unrepresentable proposals raise an error instead of silently clipping coordinates.",
        "Arithmetic convex combinations preserve bounds and linear constraints satisfied by both parents. Other proposal kinds can violate constraints; ga applies its independent original-domain admission before installing generated rows. Calling a crossover directly does not perform that solver repair. Integer coordinates are not inferred from problem properties or rounded by these numerical proposal functions.",
    ]},
    BuiltinDocumentationSection{heading:"Classes and services",paragraphs:&[
        "All active numeric roles pass class and residency checks before any provider transfer or random draw. RunMat's declared GA extensions admit exact native numeric/integer data and promote single data to double; integer nvars uses the integer-option gate. Authenticated automatic device placement can materialize on the host. Explicit or unproven device intent requires caller gather. Unused problem fields and inactive scores are not traversed.",
        "Finite host-memory policies refuse before numerical work because the full helper workspace is not finitely certified. Original cancellation, allocation and provider errors remain errors. No fitness or constraint callback is dispatched, and failures after random work do not restart the operation.",
    ]},
];
const fn crossover(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    summary: &'static str,
    examples: &'static [BuiltinExample],
    inference: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    random: bool,
) -> BuiltinCatalogEntry {
    let mut result = entry(
        name,
        descriptor,
        doc(name, summary, SECTIONS, examples),
        inference,
        random,
    );
    result.integer_capabilities = super::crossover_descriptors::INTEGERS;
    result
}
pub const CROSSOVERSCATTERED_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossoverscattered",
    &GA_CROSSOVER_HELPER_DESCRIPTOR,
    "Choose parental genes independently.",
    &[example(
        "P=[0 0;1 1]; C=crossoverscattered([1 2],[],2,struct(),[],P);",
        "assert(isequal(size(C),[1 2])); assert(all(C==0 | C==1));",
    )],
    infer::scattered,
    true,
);
pub const CROSSOVERSINGLEPOINT_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossoversinglepoint",
    &GA_CROSSOVER_HELPER_DESCRIPTOR,
    "Join parents at one random cut.",
    &[example(
        "C=crossoversinglepoint([1 2],[],2,struct(),[],[1 2;3 4]);",
        "assert(isequal(size(C),[1 2])); assert(C(1)==1); assert(C(2)==2 | C(2)==4);",
    )],
    infer::single,
    true,
);
pub const CROSSOVERTWOPOINT_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossovertwopoint",
    &GA_CROSSOVER_HELPER_DESCRIPTOR,
    "Exchange the segment between two random cuts.",
    &[example(
        "C=crossovertwopoint([1 2],[],3,struct(),[],[1 2 3;4 5 6]);",
        "assert(isequal(size(C),[1 3])); assert(C(1)==1); assert(all(C==[1 2 3] | C==[4 5 6]));",
    )],
    infer::two,
    true,
);
pub const CROSSOVERINTERMEDIATE_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossoverintermediate",
    &GA_PARAMETRIC_CROSSOVER_HELPER_DESCRIPTOR,
    "Interpolate or extrapolate with a selected ratio.",
    &[example(
        "C=crossoverintermediate([1 2],[],2,struct(),[],[0 2;2 4],0);",
        "assert(isequal(C,[0 2]));",
    )],
    infer::intermediate,
    true,
);
pub const CROSSOVERARITHMETIC_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossoverarithmetic",
    &GA_CROSSOVER_HELPER_DESCRIPTOR,
    "Form a shared-weight convex parental combination.",
    &[example(
        "C=crossoverarithmetic([1 2],[],2,struct(),[],[0 2;2 4]);",
        "assert(all(C>=[0 2] & C<=[2 4])); assert(abs(C(2)-C(1)-2)<1e-14);",
    )],
    infer::arithmetic,
    true,
);
pub const CROSSOVERHEURISTIC_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossoverheuristic",
    &GA_PARAMETRIC_CROSSOVER_HELPER_DESCRIPTOR,
    "Extrapolate toward the parent with better score.",
    &[example(
        "C=crossoverheuristic([1 2],[],1,struct(),[4;0],[0;2],1.2);",
        "assert(abs(C-2.4)<1e-14);",
    )],
    infer::heuristic,
    false,
);
pub const CROSSOVERLAPLACE_CATALOG_ENTRY: BuiltinCatalogEntry = crossover(
    "crossoverlaplace",
    &GA_CROSSOVER_HELPER_DESCRIPTOR,
    "Propose centered Laplace parental displacements.",
    &[example(
        "C=crossoverlaplace([1 2],[],2,struct(),[],[2 3;2 3]);",
        "assert(isequal(C,[2 3]));",
    )],
    infer::laplace,
    true,
);
