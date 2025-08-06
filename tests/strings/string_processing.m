% MATLAB String and Character Processing Edge Cases
% This file tests string manipulation, character arrays, and text processing

%% 1. Character Array Basics
% Basic character array operations
char_str = 'Hello World';
char_length = length(char_str);
char_size = size(char_str);

% Character indexing
first_char = char_str(1);             % 'H'
last_char = char_str(end);            % 'd'
substring = char_str(1:5);            % 'Hello'
reverse_str = char_str(end:-1:1);     % 'dlroW olleH'

% Character array concatenation
greeting = 'Hello';
name = 'MATLAB';
full_greeting = [greeting, ' ', name]; % 'Hello MATLAB'

%% 2. Multi-row Character Arrays
% Creating multi-row character arrays (padding with spaces)
char_matrix = char('Hello', 'World', 'MATLAB');
char_matrix_size = size(char_matrix);  % Should be [3 x 6]

% Accessing rows
first_row = char_matrix(1, :);        % 'Hello '
second_row = char_matrix(2, :);       % 'World '

% Deblank to remove trailing spaces
clean_first = deblank(first_row);     % 'Hello'

%% 3. String Arrays (R2016b+)
% String scalar and arrays
str_scalar = "Hello World";
str_array = ["Hello", "World", "MATLAB"];
str_matrix = ["A", "B"; "C", "D"];

% String operations
str_length = strlength(str_scalar);   % Length of string
str_concat = str_scalar + " Again";   % String concatenation
str_join = join(str_array, "-");      % Join with delimiter

% String indexing and extraction
str_char = extractAfter(str_scalar, 6); % "World"
str_before = extractBefore(str_scalar, 6); % "Hello"

%% 4. String vs Character Conversion
% Converting between types
char_to_str = string(char_str);       % Convert char to string
str_to_char = char(str_scalar);       % Convert string to char

% Mixed operations
mixed_concat = char_str + string(" Test"); % If supported

%% 5. String Search and Pattern Matching
% Basic string searching
search_str = 'The quick brown fox jumps over the lazy dog';
contains_result = contains(search_str, 'fox');     % Logical result
starts_result = startsWith(search_str, 'The');     % Logical result
ends_result = endsWith(search_str, 'dog');         % Logical result

% Find positions
strfind_result = strfind(search_str, 'the');       % Find 'the'
strcmpi_result = strcmpi('Hello', 'HELLO');        % Case-insensitive compare

% Regular expressions (if supported)
% regex_match = regexp(search_str, '\w{5}', 'match'); % 5-letter words

%% 6. String Replacement and Modification
% Basic replacement
original = 'Hello World';
replaced = strrep(original, 'World', 'MATLAB');    % 'Hello MATLAB'

% Case conversion
upper_str = upper('hello world');                  % 'HELLO WORLD'
lower_str = lower('HELLO WORLD');                  % 'hello world'
title_str = 'hello world';                         % Would need custom titlecase

% Trimming whitespace
padded_str = '  Hello World  ';
trimmed_str = strtrim(padded_str);                 % 'Hello World'

%% 7. String Splitting and Parsing
% String splitting
csv_data = 'apple,banana,cherry,date';
split_result = strsplit(csv_data, ',');            % Cell array of strings

% Tokenization
sentence = 'The quick brown fox';
tokens = strsplit(sentence, ' ');                  % Split on spaces

% Number extraction from strings
number_str = 'Value: 123.45';
% num_value = str2double(extractAfter(number_str, 'Value: '));

%% 8. String Formatting and Conversion
% Number to string conversion
num_val = 123.456;
str_from_num = num2str(num_val);                   % '123.456'
formatted_str = sprintf('Value: %.2f', num_val);   % 'Value: 123.46'

% String to number conversion
str_number = '123.456';
num_from_str = str2double(str_number);             % 123.456

% Invalid conversions
invalid_str = 'not_a_number';
invalid_num = str2double(invalid_str);             % NaN

%% 9. Character Code Operations
% ASCII/Unicode operations
char_codes = double(char_str);                     % ASCII codes
chars_from_codes = char([72, 101, 108, 108, 111]); % 'Hello'

% Special characters
newline_char = char(10);                           % Newline
tab_char = char(9);                                % Tab
null_char = char(0);                               % Null character

%% 10. String Comparison Edge Cases
% Exact comparison
exact_match = strcmp('Hello', 'Hello');            % true
exact_mismatch = strcmp('Hello', 'hello');         % false

% Case-insensitive comparison
case_insensitive = strcmpi('Hello', 'HELLO');      % true

% Partial comparison
partial_match = strncmp('Hello World', 'Hello', 5); % true

% Natural sorting (if available)
str_list = {'item1', 'item10', 'item2'};
% sorted_natural = natsort(str_list);  % Would give {'item1', 'item2', 'item10'}

%% 11. String Interpolation and Templates
% sprintf formatting
template_str = sprintf('Name: %s, Age: %d, Score: %.1f', 'John', 25, 95.7);

% Multiple format specifiers
format_test = sprintf('%d + %d = %d', 2, 3, 5);

% Special format cases
scientific = sprintf('%.2e', 1234.5);              % '1.23e+03'
hexadecimal = sprintf('%X', 255);                  % 'FF'

%% 12. Cell Array of Strings
% Cell array operations
cell_strings = {'apple', 'banana', 'cherry'};
cell_length = length(cell_strings);
cell_element = cell_strings{2};                    % 'banana'

% Converting cell array to string array
% if function exists
% str_array_from_cell = string(cell_strings);

% Cell array string operations
cell_upper = cellfun(@upper, cell_strings, 'UniformOutput', false);
cell_lengths = cellfun(@length, cell_strings);

%% 13. String Encoding and Special Characters
% Special characters in strings
special_chars = 'Line 1\nLine 2\tTabbed';         % Escape sequences
quote_str = 'He said, "Hello there!"';            % Quotes in string
apostrophe_str = 'It''s a beautiful day';         % Apostrophe escape

% Unicode characters (if supported)
% unicode_str = 'Café résumé naïve';              % Accented characters
% chinese_str = '你好世界';                        % Chinese characters

%% 14. String Performance Considerations
% Efficient string building
n = 1000;
% Inefficient: repeated concatenation
% slow_build = '';
% for i = 1:n
%     slow_build = [slow_build, num2str(i), ' '];
% end

% Efficient: cell array then join
fast_build = cell(1, n);
for i = 1:n
    fast_build{i} = num2str(i);
end
efficient_result = strjoin(fast_build, ' ');

%% 15. String Validation and Sanitization
% Input validation
% function result = validate_email(email)
%     pattern = '\w+@\w+\.\w+';  % Simple email pattern
%     result = ~isempty(regexp(email, pattern, 'once'));
% end

% String sanitization
unsafe_str = 'user<script>alert("xss")</script>input';
% safe_str = regexprep(unsafe_str, '<[^>]*>', '');  % Remove HTML tags

%% 16. Locale-Specific String Operations
% Case conversion with locale considerations
turkish_str = 'İstanbul';                         % Turkish capital İ
% locale_lower = lower(turkish_str, 'tr');        % Locale-specific if supported

% Sorting with locale
mixed_case = {'apple', 'Banana', 'cherry', 'Date'};
% locale_sort = sort(mixed_case);                  % Default sort
% collation_sort = sort(mixed_case, 'ComparisonMethod', 'locale');

%% 17. String Memory and Efficiency
% String vs character array memory usage
large_char = repmat('A', 1, 10000);               % Large character array
large_str = string(large_char);                   % Convert to string

% String sharing (strings may share memory)
str1 = "repeated text";
str2 = "repeated text";                           % May share memory

%% 18. Edge Cases and Error Conditions
% Empty strings
empty_char = '';
empty_str = "";
empty_check_char = isempty(empty_char);           % true
empty_check_str = strlength(empty_str);           % 0

% Very long strings
long_str = repmat('X', 1, 1000000);               % 1 million characters

% Invalid operations
try
    invalid_index = char_str(100);                % Out of bounds
catch
    fprintf('Caught indexing error\n');
end

%% 19. String Interoperability
% String with numeric operations
str_with_num = "Value: " + string(42);           % If supported

% Mixed data types in cell arrays
mixed_cell = {'text', 123, true, "string"};
types_in_cell = cellfun(@class, mixed_cell, 'UniformOutput', false);

%% 20. Advanced String Functions
% String distance/similarity (if available)
% edit_dist = editDistance('kitten', 'sitting');  % Levenshtein distance

% Sound-based matching (if available)
% soundex_code = soundex('Smith');                % Phonetic algorithm

% String tokenization with delimiters
complex_delim = 'word1;word2,word3:word4';
% multi_split = strsplit(complex_delim, {';', ',', ':'});

fprintf('String processing tests completed.\n');
fprintf('Character array length: %d\n', char_length);
fprintf('String scalar: %s\n', str_scalar);
fprintf('Search contains fox: %d\n', contains_result);
fprintf('Formatted string: %s\n', template_str);