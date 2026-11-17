import csv
import os
import logging

from .BuildTree import parse_sif_file, load_clustering_results



# Main run method
def run_tool(args):
    logging.debug('Arguments {}'.format(args))

    import data.Patient as Patient
    from .Tree import Tree
    from .CellPopulationEngine import CellPopulationEngine
    from .BuildTreeEngine import BuildTreeEngine

    # init a Patient
    patient_data = Patient.Patient(indiv_name=args.indiv_id, driver_genes_file=args.driver_genes_file)
    # try:  # if sif file is specified
    # Patient load cluster and mut ccf files
    parse_sif_file(args.sif, args.mutation_ccf_file, patient_data)
    load_clustering_results(args.cluster_ccf_file, patient_data, args.blacklist_threshold)
    tree_edges = load_tree_edges_file(args.tree_tsv)
    bt_engine = BuildTreeEngine(patient_data, seed=args.seed)
    tree = Tree()
    tree.init_tree_from_clustering(patient_data.ClusteringResults)
    tree.set_new_edges(tree_edges)
    patient_data.TopTree = tree
    bt_engine.set_top_tree(tree)
    # Computing Cell Population
    cp_engine = CellPopulationEngine(patient_data, seed=args.seed)
    constrained_ccf = cp_engine.compute_constrained_ccf(n_iter=args.n_iter)

    cell_ancestry = bt_engine.get_cell_ancestry()
    cell_abundance = cp_engine.get_cell_abundance_across_samples(constrained_ccf)
    # Output and visualization
    import output.PhylogicOutput
    phylogicoutput = output.PhylogicOutput.PhylogicOutput()
    # TODO write cell population MCMC trace to file
    phylogicoutput.write_all_cell_abundances(cp_engine.get_all_cell_abundances(), args.indiv_id)
    phylogicoutput.write_constrained_ccf_tsv(constrained_ccf, cell_ancestry, args.indiv_id)
    phylogicoutput.write_cell_abundances_tsv(cell_abundance, cell_ancestry, args.indiv_id)
    phylogicoutput.generate_html_from_tree(args.mutation_ccf_file, args.cluster_ccf_file,
                                           args.tree_tsv,
                                           args.indiv_id + '_constrained_ccf.tsv',
                                           sif=args.sif,
                                           drivers=patient_data.driver_genes,
                                           treatment_file=args.treatment_data,
                                           tumor_sizes_file=args.tumor_size,
                                           cnv_file=args.indiv_id + '.cnvs.txt'
                                           if os.path.isfile(args.indiv_id + '.cnvs.txt') else None)


def load_tree_edges_file(tree_tsv):
    from output.PhylogicOutput import PhylogicOutput

    with open(tree_tsv, 'r') as reader:
        top_tree = next(csv.DictReader(reader, delimiter='\t'))
    edges = PhylogicOutput.reformat_edges_for_input(top_tree['edges'])
    # The output format includes a synthetic None -> clonal edge; Tree already
    # holds the clonal root and only needs edges connecting its actual nodes.
    return [(parent, child) for parent, child in edges if parent is not None]
