%% Tunable parameters
n = 86; % number of ROIs

%% Directories
base_dir = '/path/to/CPM_outputs/output_folder';
results_dir = fullfile(base_dir,'canonical_networks');
if ~exist(results_dir, 'dir')
    mkdir(results_dir)
end

%% Read Matrices
% Load in the matrices using the "Import Data" function in the tool bar.
% OR, alternatively, read the existing alread-read matrices:
load('lst_rois.mat');
load(fullfile(base_dir, 'network_pos_consensus.txt'));
load(fullfile(base_dir, 'network_neg_consensus.txt'));

%% Read cannonical networks information
load('nets.mat');
load('nodes.mat');
load('nodes_idx.mat');

%%%%%%%%%%%%
%% pos mat%%
%%%%%%%%%%%%
disp('--------------------------------------------------');
corrMat = network_pos_consensus;
correlation = corrMat;

%% Rearrange correlation matrix
fn_corr_rearr_mat = fullfile(results_dir, 'correlation_canonical_rearr_pos.txt');
correlation_rearr=correlation(nodes_idx',nodes_idx');
writematrix(correlation_rearr, fn_corr_rearr_mat)

%% Rearrange list of rois
lstrois_rearr = lstrois(nodes_idx);
if isequal(string(lstrois_rearr), nodes)
    disp('Equal!')
end

%% Obtain module size
num_module = max(nets);
size=zeros(num_module,1);

for i = 1:num_module
    size(i)=sum(nets==i);
end

%% Module composition
disp('--------------------------------------------------');
disp('Printing module composition ...');

modules = [];
for i = 1:num_module
    tmp_module = lstrois_rearr(find(nets==i));
    modules = [modules, tmp_module'];
    out_file = fullfile(results_dir, strcat('module', num2str(i), '_comp_realvalued.txt'));
    writecell(tmp_module, out_file);
end

writecell(modules', fullfile(results_dir, 'modules_comp_realvalued.txt'));
if isequal(modules',lstrois_rearr)
    disp('Equal!')
end

%%%%%%%%%%%%
%% neg mat%%
%%%%%%%%%%%%
disp('--------------------------------------------------');
corrMat = network_neg_consensus;
correlation = corrMat;

%% Rearrange correlation matrix
fn_corr_rearr_mat = fullfile(results_dir, 'correlation_canonical_rearr_neg.txt');
correlation_rearr=correlation(nodes_idx',nodes_idx');
writematrix(correlation_rearr, fn_corr_rearr_mat)