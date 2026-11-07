function rlnm_oracle(input_file, output_file, upstream_root)
% Executes pinned upstream MATLAB calculations on shared numerical inputs.
% S3 and S2 calculation blocks are evaluated directly from upstream sources;
% their file I/O and graphics are deliberately excluded. Apache-2.0; see NOTICE.
load(input_file);
algorithm = fullfile(upstream_root, 'C2_specificity_test', 'code_algorithm');
addpath(fullfile(algorithm, 'utils'));

% Run the original S3 LOLO/OLS block without rewriting its equations.
lesion_maps = maps;
lesion_symptom = labels(:);
num_symptoms = numel(unique(labels));
num_lesions = size(maps, 1);
num_domain_voxels = size(maps, 2);
source = fileread(fullfile(algorithm, 'S3_lesion_specific.m'));
start_at = strfind(source, '%% ===== Group sums');
end_at = strfind(source, '%% ===== Write the per-lesion maps');
assert(numel(start_at) == 1 && numel(end_at) == 1);
eval(source(start_at:end_at-1));
regressors = lolo_regressor;
residuals = lesion_specific_maps;

t = zeros(num_symptoms, num_domain_voxels);
effect = t; p_one = t; p_two = t; q_one = t; q_two = t;
for symptom = 1:num_symptoms
    selected = labels(:) == symptom;
    n_target = sum(selected);
    t(symptom,:) = welch_t(residuals(selected,:), residuals(~selected,:));
    effect(symptom,:) = specific_effect_score(residuals(selected,:), residuals(~selected,:));
    count_one = zeros(1, num_domain_voxels);
    count_two = count_one;
    for draw = 1:size(orders, 1)
        order = orders(draw,:);
        null_t = welch_t(residuals(order(1:n_target),:), residuals(order(n_target+1:end),:));
        count_one = count_one + (null_t >= t(symptom,:));
        count_two = count_two + (abs(null_t) >= abs(t(symptom,:)));
    end
    p_one(symptom,:) = (count_one + 1) / (size(orders, 1) + 1);
    p_two(symptom,:) = (count_two + 1) / (size(orders, 1) + 1);
    q_one(symptom,:) = bh_fdr(p_one(symptom,:));
    q_two(symptom,:) = bh_fdr(p_two(symptom,:));
end

% Projection and the original S2 signed-overlap block (including tie precedence).
projected = zeros(size(lesion_rows, 1), size(fc, 2));
for lesion = 1:size(lesion_rows, 1)
    projected(lesion,:) = mean(fc(lesion_rows(lesion,:),:), 1);
end
R = projected';
num_voxels = size(R, 1); num_files = size(R, 2);
UPPER_PERCENTILE = 75; LOWER_PERCENTILE = 25;
source = fileread(fullfile(upstream_root, 'C1_sensitivity_test', 'code_algorithm', ...
                         'S2_true_lesion_LNM_generate.m'));
start_at = strfind(source, '%% ===== Select the top and bottom');
end_at = strfind(source, '%% ===== Save the result');
assert(numel(start_at) == 1 && numel(end_at) == 1);
eval(source(start_at:end_at-1));
overlap = overlap(:)';

save(output_file, 'regressors', 'residuals', 't', 'effect', 'p_one', 'p_two', ...
     'q_one', 'q_two', 'projected', 'overlap', '-v7');
end
