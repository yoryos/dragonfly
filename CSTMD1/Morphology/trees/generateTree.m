% generateTree.m
% Author: Chris Snowden - Imperial College London  (cps15@imperial.ac.uk)
% Refactored from code written by Zafeirios Fountas - Imperial College London (zfountas@imperial.ac.uk)
% Last Modified: 22/01/16

function savedPath = generateTree(treesPath, coordinates, scale, offset, maxLength, path)
% uses the trees package to enerate a dendritic tree using the points given as the coordiantes argument. The result is
% saved as a SWC file using the path given as the last argument. Returns the savedPath.

addpath (treesPath)
start_trees;

tree = MST_tree(1, coordinates(:,1), coordinates(:,2), coordinates(:,3), 0.5, maxLength,[],[],'none');
quaddiameter_tree(tree, scale, offset);

swc_tree(tree, path);

savedPath = path

