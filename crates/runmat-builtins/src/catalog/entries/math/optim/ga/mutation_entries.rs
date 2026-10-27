use super::{
    helper_documentation::{doc, example},
    helper_entries::creator,
    mutation_inference as infer, *,
};
use crate::*;
const SECTIONS:&[BuiltinDocumentationSection]=&[
    BuiltinDocumentationSection{heading:"Calling forms",paragraphs:&[
        "children=mutationFcn(parents,options,nvars,problem,state,scores,population) returns one host-double row per selected one-based parent index. population has nvars columns, options is a scalar genetic options context or [], and problem is the R2026a scalar operator context including FitnessFcn. No fitness or nonlinear constraint function is called. scores is inactive for these predefined kernels. State must be a scalar structure.",
        "Gaussian mutation accepts optional scale and shrink arguments, default one and one. Uniform mutation accepts optional rate, default0.01. Other mutation kinds accept no extra arguments. Scale is finite nonnegative, shrink finite, and rate is in [0,1]. PopulationType is doubleVector, except uniform also supports bitstring with exact zero/one parents. Custom populations use custom mutation callbacks instead.",
    ]},
    BuiltinDocumentationSection{heading:"Schedules and proposals",paragraphs:&[
        "Gaussian mutation uses InitialPopulationRange and the selected bounds to initialize deviation. It reads nonnegative state.Generation and MaxGenerations, then reconstructs the recursive shrink schedule through the next generation. Each coordinate receives a normal displacement. An unrepresentable proposal retains its parent; a deviation that cannot be represented is an error. Uniform mutation independently replaces selected coordinates with uniform draws in their working intervals, or flips selected bits for bitstring populations.",
        "Power mutation uses displacement fraction U^2 toward a selected interval endpoint. Adaptive-feasible and positive-basis mutations use the original linear/bound/integer geometry and the same exact feasible-cone owner as ga. They require an originally feasible parent. Their standalone adaptation reconstructs successful and unsuccessful transitions from consecutive state.Best values; finite lower values are improvements. Adaptive-feasible samples maximal-cone directions; positive-basis cycles the minimal basis. A blocked direction retains its parent.",
        "Working intervals use InitialPopulationRange and finite bounds; absent ranges use [-10,10] for continuous variables and [-9999,10001] for mixed-integer problems, shifted/intersected with the admitted domain. Only power and feasible mutations consume original linear/integer rows; Gaussian and uniform do not enforce those rows. The outer ga solver independently admits its offspring. A standalone power proposal does not round coordinates to integers.",
    ]},
    BuiltinDocumentationSection{heading:"Services and representation",paragraphs:&[
        "Only active data and controls pass class and authenticated residency admission. The declared GA extensions admit exact native numeric data, promote single data, and independently gate exact integer controls. Authenticated automatic placement can materialize on the host; explicit or unproven device intent requires caller gather. Inactive scores, nonlinear callbacks and unrelated option payloads are not traversed.",
        "These standalone mutations capture their selected Clock service, use the original runtime random stream and cancellation checkpoints, and refuse finite host-memory policies before workspace construction. They do not reset the enclosing solver's clock or count their numerical work as fitness evaluation. Errors after random draws do not replay the operation. Their public state-based schedule does not retain a hidden global cache.",
    ]},
];
const fn mutation(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    summary: &'static str,
    examples: &'static [BuiltinExample],
    inference: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
) -> BuiltinCatalogEntry {
    let mut result = creator(
        name,
        descriptor,
        doc(name, summary, SECTIONS, examples),
        inference,
        true,
        false,
    );
    result.integer_capabilities = super::mutation_descriptors::INTEGERS;
    result
}
pub const MUTATIONGAUSSIAN_CATALOG_ENTRY:BuiltinCatalogEntry=mutation("mutationgaussian",&GA_PARAMETRIC_MUTATION_HELPER_DESCRIPTOR,"Apply the recursive Gaussian mutation schedule.",&[example(
    "o=optimoptions('ga','MaxGenerations',5); p=struct('FitnessFcn',@(x)x(1)^2); s=struct('Generation',0); C=mutationgaussian([2 1],o,1,p,s,[],[1;3],0,1);",
    "assert(isequal(C,[3;1]));")],infer::gaussian);
pub const MUTATIONUNIFORM_CATALOG_ENTRY:BuiltinCatalogEntry=mutation("mutationuniform",&GA_PARAMETRIC_MUTATION_HELPER_DESCRIPTOR,"Replace selected coordinates or flip selected bits.",&[example(
    "p=struct('FitnessFcn',@(x)x(1)^2); C=mutationuniform([1 2],[],2,p,struct(),[],[1 2;3 4],0);",
    "assert(isequal(C,[1 2;3 4]));")],infer::uniform);
pub const MUTATIONPOWER_CATALOG_ENTRY:BuiltinCatalogEntry=mutation("mutationpower",&GA_MUTATION_HELPER_DESCRIPTOR,"Move toward interval endpoints with a squared uniform fraction.",&[example(
    "p=struct('FitnessFcn',@(x)x(1)^2,'lb',2,'ub',2,'intcon',1); C=mutationpower(1,[],1,p,struct(),[],2);",
    "assert(isequal(C,2));")],infer::power);
pub const MUTATIONADAPTFEASIBLE_CATALOG_ENTRY:BuiltinCatalogEntry=mutation("mutationadaptfeasible",&GA_MUTATION_HELPER_DESCRIPTOR,"Sample adaptive original-domain feasible directions.",&[example(
    "p=struct('FitnessFcn',@(x)x(1)^2,'Aeq',1,'Beq',2,'lb',0,'ub',3); s=struct('Best',[4 3]); C=mutationadaptfeasible(1,[],1,p,s,[],2);",
    "assert(abs(C-2)<1e-8);")],infer::adaptive);
pub const MUTATIONPOSITIVEBASIS_CATALOG_ENTRY:BuiltinCatalogEntry=mutation("mutationpositivebasis",&GA_MUTATION_HELPER_DESCRIPTOR,"Cycle the minimal feasible positive basis.",&[example(
    "p=struct('FitnessFcn',@(x)x(1)^2,'lb',2,'ub',2,'intcon',1); C=mutationpositivebasis(1,[],1,p,struct('Best',4),[],2);",
    "assert(isequal(C,2));")],infer::positive);
