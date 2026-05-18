%% RLS_parameter_estimation.m
% RLS-based internal parameter estimation for lithium-ion battery cells
%
% This script estimates ohmic resistance (Ri), diffusion resistance (Rdiff),
% diffusion capacitance (Cdiff), and reconstructed terminal voltage using a
% first-order RC equivalent circuit model.
%
% Required input:
%   - CSV file containing cell voltage columns and current column
%   - LUT files: OCV.mat, Ri.mat, Rdiff.mat, Cdiff.mat
%
% Note:
%   The experimental dataset used in the paper is not included in this
%   repository due to institutional review. Users should replace the example
%   file paths with their own data paths.

clear; clc;

%% User settings
data_file = fullfile('data', 'example_cycle.csv');
lut_dir   = fullfile('data', 'lut');

voltage_columns = 1:6;
current_column  = 8;

sampling_time = 1;       % [s]
capacity_Ah   = 3.6;     % [Ah]
lambda         = 0.99;    % forgetting factor
P0             = eye(3);
P_clip         = 1e-2;

%% Load LUT
load(fullfile(lut_dir, 'OCV.mat'), 'OCV');
load(fullfile(lut_dir, 'Ri.mat'), 'Ri');
load(fullfile(lut_dir, 'Rdiff.mat'), 'Rdiff');
load(fullfile(lut_dir, 'Cdiff.mat'), 'Cdiff');

OCV   = OCV(:);
Ri    = Ri(:);
Rdiff = Rdiff(:);
Cdiff = Cdiff(:);

SOC = (1:-0.05:0).';

if any([numel(OCV), numel(Ri), numel(Rdiff), numel(Cdiff)] ~= numel(SOC))
    error('The LUT lengths must match the SOC vector length.');
end

OCV_lookup   = @(soc) interp1(SOC, OCV,   soc, 'linear', 'extrap');
Ri_lookup    = @(soc) interp1(SOC, Ri,    soc, 'linear', 'extrap');
Rdiff_lookup = @(soc) interp1(SOC, Rdiff, soc, 'linear', 'extrap');
Cdiff_lookup = @(soc) interp1(SOC, Cdiff, soc, 'linear', 'extrap');

SOC_from_voltage = @(V) max(0, min(1, interp1(OCV, SOC, V, 'linear', 'extrap')));

%% Load data
raw_data = readmatrix(data_file);

cell_voltage = raw_data(:, voltage_columns);
pack_current = raw_data(:, current_column);

% For a 2P module, pack current is divided by two to obtain cell-branch current.
cell_current = pack_current / 2;

num_samples = size(cell_voltage, 1);
num_cells   = size(cell_voltage, 2);

Ri_est    = nan(num_samples, num_cells);
Rdiff_est = nan(num_samples, num_cells);
Cdiff_est = nan(num_samples, num_cells);
Vhat      = nan(num_samples, num_cells);

%% RLS estimation for each cell
for cell_idx = 1:num_cells

    V = cell_voltage(:, cell_idx);
    I = cell_current;

    soc0 = SOC_from_voltage(V(1));

    SOC_ref = zeros(num_samples, 1);
    SOC_ref(1) = soc0;

    for k = 2:num_samples
        SOC_ref(k) = SOC_ref(k-1) + I(k) * sampling_time / (3600 * capacity_Ah);
        SOC_ref(k) = max(0, min(1, SOC_ref(k)));
    end

    OCV_ref = arrayfun(OCV_lookup, SOC_ref);

    Ri_0    = Ri_lookup(soc0);
    Rdiff_0 = Rdiff_lookup(soc0);
    Cdiff_0 = Cdiff_lookup(soc0);

    theta = zeros(3, num_samples);
    theta(:, 1) = [
        Ri_0;
        -Ri_0 + sampling_time / Cdiff_0 + sampling_time * Ri_0 / (Rdiff_0 * Cdiff_0);
        1 - sampling_time / (Rdiff_0 * Cdiff_0)
    ];

    P = zeros(3, 3, num_samples);
    P(:, :, 1) = P0;

    Ri_cell    = zeros(num_samples, 1);
    Rdiff_cell = zeros(num_samples, 1);
    Cdiff_cell = zeros(num_samples, 1);

    Ri_cell(1)    = Ri_0;
    Rdiff_cell(1) = Rdiff_0;
    Cdiff_cell(1) = Cdiff_0;

    for k = 2:num_samples

        phi = [
            I(k);
            I(k-1);
            V(k-1) - OCV_ref(k-1)
        ];

        P_prev = P(:, :, k-1);
        P_prev = max(0, min(P_clip, P_prev));

        gain = (P_prev * phi) / (lambda + phi.' * P_prev * phi);

        P_new = (P_prev - gain * phi.' * P_prev) / lambda;
        P_new = max(0, min(P_clip, P_new));
        P(:, :, k) = P_new;

        overpotential = V(k) - OCV_ref(k);
        y_hat = theta(:, k-1).' * phi;
        error_k = overpotential - y_hat;

        theta(:, k) = abs(theta(:, k-1) + gain * error_k);

        b0 = theta(1, k);
        b1 = theta(2, k);
        a1 = theta(3, k);

        Ri_cell(k)    = abs(b0);
        Rdiff_cell(k) = abs((b1 - a1 * b0) / (1 + a1));
        Cdiff_cell(k) = abs(sampling_time / (b1 - a1 * b0));
    end

    Ri_est(:, cell_idx)    = Ri_cell;
    Rdiff_est(:, cell_idx) = Rdiff_cell;
    Cdiff_est(:, cell_idx) = Cdiff_cell;

    %% Voltage reconstruction
    Vdiff_state = zeros(num_samples, 1);
    V_recon = zeros(num_samples, 1);

    V_recon(1) = OCV_ref(1) + I(1) * Ri_cell(1) + Vdiff_state(1);

    for k = 2:num_samples
        tau = max(Rdiff_cell(k) * Cdiff_cell(k), 1e-4);
        alpha = exp(-sampling_time / tau);

        Vdiff_state(k) = alpha * Vdiff_state(k-1) ...
            + (1 - alpha) * I(k) * Rdiff_cell(k);

        V_recon(k) = OCV_ref(k) + I(k) * Ri_cell(k) + Vdiff_state(k);
    end

    Vhat(:, cell_idx) = V_recon;
end

%% Reconstruction metrics
e = cell_voltage - Vhat;

RMSE = sqrt(mean(e.^2, 1));
MAE  = mean(abs(e), 1);

R2 = zeros(1, num_cells);
RAE = zeros(1, num_cells);

for cell_idx = 1:num_cells
    V = cell_voltage(:, cell_idx);
    V_recon = Vhat(:, cell_idx);

    R2(cell_idx) = 1 - sum((V - V_recon).^2) / max(sum((V - mean(V)).^2), eps);
    RAE(cell_idx) = sum(abs(V - V_recon)) / max(sum(abs(V - mean(V))), eps);
end

results = table((1:num_cells).', RMSE.', MAE.', R2.', RAE.', ...
    'VariableNames', {'Cell', 'RMSE', 'MAE', 'R2', 'RAE'});

disp(results);

%% Save estimated parameters
output_dir = fullfile('results');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

save(fullfile(output_dir, 'RLS_estimated_parameters.mat'), ...
    'Ri_est', 'Rdiff_est', 'Cdiff_est', 'Vhat', 'results');