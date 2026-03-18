clear;
close all;
clc;

%% 0. Add QuaDRiGa path
quadriga_path = 'C:\WCML\Exercise_2.4\QuaDRiGa-main\QuaDRiGa-main\quadriga_src';
addpath(genpath(quadriga_path));
rehash;

assert(~isempty(which('qd_simulation_parameters')), ...
    'QuaDRiGa toolbox not found. Check quadriga_path.');

rng(42);   % Reproducibility

%% 1. Define Simulation Parameters
fc = 3.5e9;                 % Center frequency (Hz)
num_tx = 1;                 % Number of transmit antennas (SISO)
num_rx = 1;                 % Number of receive antennas (SISO)
ue_speed_kmh = 3;           % User equipment speed (km/h)
num_snapshots = 20000;      % Desired number of snapshots

ue_speed_mps = ue_speed_kmh * 1000 / 3600;

%% 2. Create QuaDRiGa Simulation Parameters
s = qd_simulation_parameters;
s.center_frequency = fc;
s.sample_density = 2.5;         % Reasonable default for single mobility
s.use_absolute_delays = 1;
s.show_progress_bars = 1;

samples_per_meter = s.samples_per_meter;

% Choose track length so that interpolation gives about num_snapshots points
track_length = (num_snapshots - 1) / samples_per_meter;
track_direction = 0;   % 0 rad = moving toward +x direction

fprintf('Center frequency       : %.2f GHz\n', fc/1e9);
fprintf('UE speed               : %.2f km/h (%.4f m/s)\n', ue_speed_kmh, ue_speed_mps);
fprintf('Sample density         : %.2f samples / half-wavelength\n', s.sample_density);
fprintf('Samples per meter      : %.4f\n', samples_per_meter);
fprintf('Target snapshots       : %d\n', num_snapshots);
fprintf('Track length           : %.4f m\n', track_length);

%% 3. Create Scenario and Layout
l = qd_layout(s);

% Tx configuration
l.tx_array = qd_arrayant('omni');
l.tx_position = [0; 0; 25];     % BS at 25 m height

% Rx configuration
l.rx_array = qd_arrayant('omni');
l.no_rx = num_rx;

% Create a proper linear track with physical length
track = qd_track('linear', track_length, track_direction);
track.initial_position = [100; 0; 1.5];    % Start at x=100 m, y=0, z=1.5 m

% Interpolate positions using QuaDRiGa sampling rule
track.interpolate_positions(samples_per_meter);

% Align UE orientation with movement direction
track.calc_orientation();

% Assign track and scenario
l.rx_track = track;
l.set_scenario('3GPP_38.901_UMi_NLOS');

fprintf('Scenario               : 3GPP_38.901_UMi_NLOS\n');
fprintf('Actual track snapshots : %d\n', l.rx_track.no_snapshots);

%% 4. Generate Channel Coefficients
b = l.init_builder;
gen_parameters(b);

% get_channels may return segment-wise channels; merge ensures a continuous snapshot sequence
c = get_channels(b);
h = merge(c);

% Extract coefficients
% h.coeff dimensions: [Rx_Ant, Tx_Ant, Num_Paths, Num_Snapshots]
h_coeff = h.coeff;

fprintf('Size of h.coeff        : [%s]\n', num2str(size(h_coeff)));

% Sum over multipath dimension to get effective flat-fading SISO coefficient
h_flat = sum(h_coeff, 3);

% Convert to 1-D complex vector
h_siso = squeeze(h_flat);

fprintf('Processed h_siso size  : [%s]\n', num2str(size(h_siso)));

%% 5. Force exact snapshot count if needed
% Due to interpolation rounding, the actual number may differ by 1 sample.
if numel(h_siso) > num_snapshots
    h_siso = h_siso(1:num_snapshots);
elseif numel(h_siso) < num_snapshots
    error('Generated snapshots (%d) are fewer than requested (%d).', numel(h_siso), num_snapshots);
end

fprintf('Final h_siso length    : %d\n', numel(h_siso));

%% 6. Sanity Checks
num_unique = numel(unique(h_siso));
fprintf('Unique channel values  : %d\n', num_unique);

if num_unique < 100
    warning('Channel variation is suspiciously low. Please inspect the generated track and coefficients.');
end

%% 7. Visualization
figure;
plot(real(h_siso), 'LineWidth', 1);
grid on;
title('Real Part of h\_siso');
xlabel('Snapshot Index');
ylabel('Real(h)');

figure;
plot(imag(h_siso), 'LineWidth', 1);
grid on;
title('Imaginary Part of h\_siso');
xlabel('Snapshot Index');
ylabel('Imag(h)');

figure;
plot(abs(h_siso), 'LineWidth', 1);
grid on;
title('Magnitude of h\_siso');
xlabel('Snapshot Index');
ylabel('|h|');

figure;
plot(real(h_siso), imag(h_siso), '.');
grid on;
axis equal;
title('Complex Plane of h\_siso');
xlabel('Real(h)');
ylabel('Imag(h)');

%% 8. Save Dataset
dataset_filename = 'rayleigh_channel_dataset.mat';
save(dataset_filename, 'h_siso');

fprintf('\nDataset saved to ''%s''\n', dataset_filename);