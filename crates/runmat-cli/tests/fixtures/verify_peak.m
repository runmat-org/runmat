function result = verify_peak(value, limit)
arguments
 value (1,1) double
 limit (1,1) double
end
result = struct('evaluated', true, 'accepted', value <= limit);
end
