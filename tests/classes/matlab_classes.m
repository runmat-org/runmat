% MATLAB Class System Edge Cases
% This file tests MATLAB's object-oriented programming features

%% 1. Simple Class Definition (would be in separate .m file)
% classdef SimpleClass < handle
%     properties
%         Value
%         Name
%     end
%     
%     properties (Access = private)
%         PrivateData
%     end
%     
%     methods
%         function obj = SimpleClass(val, name)
%             obj.Value = val;
%             obj.Name = name;
%             obj.PrivateData = rand();
%         end
%         
%         function result = getValue(obj)
%             result = obj.Value;
%         end
%         
%         function obj = setValue(obj, newVal)
%             obj.Value = newVal;
%         end
%     end
%     
%     methods (Static)
%         function obj = createDefault()
%             obj = SimpleClass(0, 'default');
%         end
%     end
% end

%% 2. Class Usage Tests (assuming class exists)
% obj1 = SimpleClass(42, 'test');
% val = obj1.getValue();
% obj1.setValue(100);
% 
% % Static method call
% default_obj = SimpleClass.createDefault();

%% 3. Handle vs Value Class Behavior
% Handle classes: objects are passed by reference
% Value classes: objects are copied

%% 4. Inheritance Testing
% classdef DerivedClass < SimpleClass
%     properties
%         ExtraProperty
%     end
%     
%     methods
%         function obj = DerivedClass(val, name, extra)
%             obj@SimpleClass(val, name);
%             obj.ExtraProperty = extra;
%         end
%         
%         function result = getValue(obj)  % Override parent method
%             result = obj.Value * 2;
%         end
%     end
% end

%% 5. Property Attributes Testing
% properties (SetAccess = private, GetAccess = public)
% properties (Dependent)
% properties (Constant)
% properties (Abstract)

%% 6. Method Attributes Testing
% methods (Access = protected)
% methods (Static)
% methods (Abstract)
% methods (Sealed)

%% 7. Events and Listeners (Advanced OOP)
% events
%     ValueChanged
% end
% 
% methods
%     function obj = setValue(obj, newVal)
%         oldVal = obj.Value;
%         obj.Value = newVal;
%         notify(obj, 'ValueChanged', ...
%                matlab.event.PropertyEventData('Value', oldVal, newVal));
%     end
% end

%% 8. Class Validation and Input Parsing
% methods
%     function obj = setValidatedValue(obj, val)
%         arguments
%             obj SimpleClass
%             val (1,1) double {mustBePositive}
%         end
%         obj.Value = val;
%     end
% end

%% 9. Operator Overloading
% methods
%     function result = plus(obj1, obj2)
%         result = SimpleClass(obj1.Value + obj2.Value, ...
%                             [obj1.Name, '+', obj2.Name]);
%     end
%     
%     function result = eq(obj1, obj2)
%         result = obj1.Value == obj2.Value;
%     end
% end

%% 10. Enumeration Classes
% classdef Color < uint8
%     enumeration
%         Red (1)
%         Green (2)
%         Blue (3)
%     end
% end
% 
% color1 = Color.Red;
% color_val = uint8(color1);

%% 11. Abstract Classes and Interfaces
% classdef (Abstract) AbstractShape
%     methods (Abstract)
%         area = calculateArea(obj)
%         perimeter = calculatePerimeter(obj)
%     end
% end

%% 12. Sealed Classes
% classdef (Sealed) FinalClass
%     % Cannot be inherited from
% end

%% 13. Class Introspection
% obj = SimpleClass(1, 'test');
% class_name = class(obj);
% is_handle = isa(obj, 'handle');
% methods_list = methods(obj);
% properties_list = properties(obj);

%% 14. Dynamic Property Access
% obj = SimpleClass(1, 'test');
% prop_name = 'Value';
% val = obj.(prop_name);  % Dynamic property access

%% 15. Class Arrays and Indexing
% obj_array(1) = SimpleClass(1, 'first');
% obj_array(2) = SimpleClass(2, 'second');
% obj_array(3) = SimpleClass(3, 'third');
% 
% first_obj = obj_array(1);
% values = [obj_array.Value];  % Comma-separated list expansion

%% 16. Copy Semantics
% original = SimpleClass(42, 'original');
% shallow_copy = original;      % Handle class: same object
% deep_copy = copy(original);   % Explicit copy

%% 17. Destructor and Cleanup
% methods
%     function delete(obj)
%         fprintf('Destroying object: %s\n', obj.Name);
%     end
% end

%% 18. Package System (+package directories)
% import mypackage.MyClass;
% obj = MyClass();
% 
% % Or fully qualified:
% obj = mypackage.MyClass();

%% For now, test basic object-like behavior with structures
student = struct('name', 'Alice', 'age', 20, 'grades', [85, 90, 78]);
student_name = student.name;
student.age = 21;  % Property assignment

% Array of structures
students(1) = struct('name', 'Alice', 'age', 20);
students(2) = struct('name', 'Bob', 'age', 21);
students(3) = struct('name', 'Charlie', 'age', 19);

all_names = {students.name};  % Comma-separated list to cell array
all_ages = [students.age];    % To numeric array