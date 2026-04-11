function cfg = load_config(name)
%LOAD_CONFIG Read a DysonianLineCNN JSON config file.
%
%   cfg = load_config(name) reads config/<name>.json (relative to this
%   file's parent directory) and returns it as a MATLAB struct via
%   jsondecode.
%
%   Inputs:
%       name   - config basename without extension, e.g. 'paths',
%                'dataset', 'training', 'inference'.
%
%   Example:
%       ds = load_config('dataset');
%       fprintf('N = %d\n', ds.N);
%
%   Notes:
%       - JSON field names that are not valid MATLAB identifiers (e.g.
%         starting with an underscore) are silently renamed by
%         jsondecode; _doc fields become x_doc and can be ignored.
%       - config/paths.json is gitignored and must be copied from
%         config/paths.example.json and edited for the local machine
%         before any script that calls load_config('paths') will work.

    thisFile = mfilename('fullpath');
    thisDir  = fileparts(thisFile);
    repoRoot = fileparts(thisDir);
    cfgDir   = fullfile(repoRoot, 'config');
    cfgPath  = fullfile(cfgDir, [char(name) '.json']);

    if ~isfile(cfgPath)
        if strcmp(char(name), 'paths')
            error('load_config:MissingPaths', ...
                ['config/paths.json not found. Copy ' ...
                 'config/paths.example.json to config/paths.json and ' ...
                 'edit drive_root_mac for your machine. Expected at: %s'], ...
                cfgPath);
        else
            error('load_config:NotFound', ...
                'Config file not found: %s', cfgPath);
        end
    end

    raw = fileread(cfgPath);
    cfg = jsondecode(raw);
end
