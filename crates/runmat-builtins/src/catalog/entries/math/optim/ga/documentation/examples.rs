use crate::*;
const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    example("initial-observations", "Return the initial population and its raw fitness",
        "o=optimoptions('ga','Display','off','PopulationSize',3,'InitialPopulationMatrix',[-1;0;2],'MaxGenerations',0);\n[x,f,flag,out,pop,scores]=ga(@(x)x(1)^2,1,[],[],[],[],[],[],[],o);",
        "assert(x==0 && f==0 && flag==0); assert(out.generations==0 && out.funccount==3); assert(isequal(pop,[-1;0;2])); assert(isequal(scores,[1;0;4]));"),
    example("evolve-real-generations", "Evolve offspring while preserving a feasible elite",
        "rng(19); o=optimoptions('ga','Display','off','PopulationSize',4,'EliteCount',1,'CrossoverFraction',0,'InitialPopulationMatrix',[-1;0;0.5;1],'MaxGenerations',2);\n[x,f,flag,out,pop,scores]=ga(@(x)x(1)^2,1,[],[],[],[],[],[],[],o);",
        "assert(x==0 && f==0 && flag==0); assert(out.generations==2 && out.funccount>=4); assert(isequal(size(pop),[4 1])); assert(all(abs(scores-pop.^2)<1e-12*max(1,abs(scores))));"),
    example("mixed-integer-population", "Evaluate an admitted integer population",
        "o=optimoptions('ga','Display','off','PopulationSize',4,'InitialPopulationMatrix',[0;1;2;3],'MaxGenerations',0);\n[x,f,flag,out,pop,scores]=ga(@(x)(x(1)-2)^2,1,[],[],[],[],0,3,[],1,o);",
        "assert(x==2 && f==0 && flag==0); assert(out.funccount==4); assert(all(pop==floor(pop))); assert(isequal(scores,[4;1;0;1]));"),
    example("vectorized-fitness", "Measure one complete vectorized population",
        "o=optimoptions('ga','Display','off','PopulationSize',3,'InitialPopulationMatrix',[-1 1;0 0;1 -1],'UseVectorized',true,'MaxGenerations',0);\n[x,f,flag,out,pop,scores]=ga(@(x)sum(x.^2,2),2,[],[],[],[],[],[],[],o);",
        "assert(isequal(x,[0 0]) && f==0 && flag==0); assert(out.funccount==3); assert(isequal(scores,[2;0;2]));"),
];
