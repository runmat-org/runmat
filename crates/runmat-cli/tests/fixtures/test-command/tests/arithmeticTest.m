function tests = arithmeticTest()
  tests = functiontests(localfunctions);
end

function testPasses(testCase)
  testCase.verifyEqual(helper() + testDependencyValue(), 42);
end

function testFails(testCase)
  testCase.verifyEqual(1 + 1, 3);
end

function testHangs(testCase)
  while true
  end
  testCase.verifyTrue(false);
end
