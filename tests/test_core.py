"""Regression checks for Python 3 data loading and clustering semantics."""
from io import StringIO
from types import SimpleNamespace

import numpy as np

from BuildTree.CellPopulationEngine import CellPopulationEngine
from BuildTree.ClusterObject import Cluster
from BuildTree.Tree import Tree
from Cluster.DpEngine import DpEngine
from data.Patient import NDHistogram, Patient
from data.Sample import TumorSample
from data.SomaticEvents import CN_SegProfile


def mutation_table(rows, *, counts=True):
    header = ["Chromosome", "Start_position", "Reference_Allele", "Tumor_Seq_Allele2"]
    if counts:
        header += ["t_ref_count", "t_alt_count"]
    header += [f"ccf_raw_{i / 100}" for i in range(101)]
    lines = ["\t".join(header)]
    for position, peak, depth in rows:
        values = ["1", str(position), "A", "T"]
        if counts:
            values += [str(depth), str(depth)]
        values += ["1" if i == peak else "0" for i in range(101)]
        lines.append("\t".join(values))
    return StringIO("# mutation input\n\n" + "\n".join(lines) + "\n")


def test_sample_loading_and_patient_concordance():
    samples = [
        TumorSample(mutation_table([(101, 80, 20), (102, 50, 1)]), "tab", sample_name=name)
        for name in ("first", "second")
    ]
    patient = Patient()
    patient.sample_list = samples
    patient.preprocess_samples()
    assert samples[0].mut_varstr == ["1:101:A:T", "1:102:A:T"]
    assert len(samples[0].low_coverage_mutations) == 1
    histogram = patient._make_ND_histogram()
    assert histogram._hist_array.shape == (1, 2, 101)
    np.testing.assert_allclose(np.exp(histogram["1:101:A:T"]).sum(axis=1), 1)
    assert [label for label, _ in histogram.items()] == ["1:101:A:T"]


def test_histograms_without_read_counts_load_and_preprocess():
    sample = TumorSample(mutation_table([(101, 80, 20)], counts=False), "tab", sample_name="sample")
    patient = Patient()
    patient.sample_list = [sample]
    patient.preprocess_samples()
    assert len(sample.concordant_variants) == 1
    assert sample.private_mutations == []


def test_copy_number_segment_rounding(tmp_path):
    segment = tmp_path / "segments.tsv"
    segment.write_text("chromosome\tstart_position\tend_position\ta1_cn\ta2_cn\n"
                       "1\t10\t100\t0.5\t2.5\n")
    profile = CN_SegProfile(str(segment))
    interval = next(iter(profile.seg_tree["1"][50]))
    # Keep the original half-up copy-number calls, rather than Python 3's
    # round-to-even result (0, 2) at these boundary values.
    assert interval.data[2:4] == [3, 1]


def test_seeded_dp_clustering_normalizes_histograms():
    grid = np.arange(101)
    peaks = [(95, 90)] * 5 + [(40, 60)] * 5
    densities = [[np.exp(-0.5 * ((grid - peak) / 3) ** 2) for peak in pair] for pair in peaks]
    histogram = NDHistogram(densities, [str(i) for i in range(len(peaks))])
    engine = DpEngine(histogram, N_iter=8, Pi_k={"r": 3, "mu": 3}, seed=7)
    assert len(engine.results.assign) == len(peaks)
    assert np.isfinite(engine.results.var_loc).all()
    for density in engine.results.clust_CCF_dens:
        assert density.shape == (2, 101)
        np.testing.assert_allclose(density.sum(axis=1), 1, atol=2e-5)


def test_tree_likelihood_and_cell_population_conserve_abundance():
    clusters = {}
    for identifier, peak in [(1, 100), (2, 40), (3, 30)]:
        cluster = Cluster(identifier, ["sample"])
        density = np.zeros(101)
        density[peak] = 1
        cluster.add_sample_density("sample", density)
        cluster.set_blacklist_status()
        clusters[identifier] = cluster
    tree = Tree()
    tree.init_tree_from_clustering(clusters)
    assert np.isfinite(tree.compute_tree_likelihood(["sample"]))
    patient = SimpleNamespace(ClusteringResults=clusters, TopTree=tree)
    engine = CellPopulationEngine(patient, seed=7)
    abundance = engine.get_cell_abundance({1: 100, 2: 40, 3: 30})
    assert abundance == {1: 30, 2: 40, 3: 30}
    assert sum(abundance.values()) == 100


def test_sample_rebins_to_requested_grid():
    sample = TumorSample(mutation_table([(101, 80, 20)]), "tab", ccf_grid_size=51)
    assert sample.mutations[0].ccf_grid_size == 51
    assert np.argmax(sample.mutations[0].ccf_1d) == 40


def test_cell_population_command_uses_supplied_tree_and_iterations(tmp_path, monkeypatch):
    import csv
    from PhylogicNDT import main
    from output.PhylogicOutput import PhylogicOutput

    samples = ["first", "second"]
    sif = tmp_path / "samples.sif"
    sif.write_text("sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint\n"
                   "first\tunused\t\t0.8\t0\nsecond\tunused\t\t0.8\t1\n")
    mutations = tmp_path / "mutations.tsv"
    mutation_header = ["Patient_ID", "Sample_ID", "Cluster_Assignment", "Chromosome", "Start_position",
                       "Reference_Allele", "Tumor_Seq_Allele", "Variant_Type", "t_ref_count", "t_alt_count",
                       "Hugo_Symbol", "Protein_change", "Variant_Classification", "preDP_ccf_mean"]
    mutation_header += [f"preDP_ccf_{i / 100}" for i in range(101)]
    clusters = tmp_path / "clusters.tsv"
    cluster_header = ["Patient_ID", "Sample_ID", "Cluster_ID", "postDP_ccf_mean",
                      "postDP_ccf_CI_low", "postDP_ccf_CI_high"] + [f"postDP_ccf_{i / 100}" for i in range(101)]
    with mutations.open("w") as mutation_file, clusters.open("w") as cluster_file:
        mutation_writer = csv.writer(mutation_file, delimiter="\t")
        cluster_writer = csv.writer(cluster_file, delimiter="\t")
        mutation_writer.writerow(mutation_header)
        cluster_writer.writerow(cluster_header)
        for sample in samples:
            for identifier, peak in [(1, 100), (2, 40)]:
                density = [int(i == peak) for i in range(101)]
                mutation_writer.writerow(["P", sample, identifier, "1", 100 + identifier, "A", "T", "SNP",
                                          30, 20, "TP53", "p.X", "Missense_Mutation", peak / 100] + density)
                cluster_writer.writerow(["P", sample, identifier, peak / 100, peak / 100, peak / 100] + density)
    PhylogicOutput().write_tree_tsv(
        [{"n_iter": 10, "likelihood": [1.0], "edges": [(1, 2)]}], str(tmp_path / "input")
    )
    tree = tmp_path / "input_build_tree_posteriors.tsv"
    observed_tree_files = []
    generate_report = PhylogicOutput.generate_html_from_tree

    def record_report(self, mutations, clusters, tree_file, *args, **kwargs):
        observed_tree_files.append(tree_file)
        return generate_report(self, mutations, clusters, tree_file, *args, **kwargs)

    monkeypatch.setattr(PhylogicOutput, "generate_html_from_tree", record_report)
    monkeypatch.chdir(tmp_path)
    assert main(["CellPopulation", "-i", "P", "-sif", str(sif), "-m", str(mutations),
                 "-c", str(clusters), "-t", str(tree), "-ni", "4", "--seed", "7"]) == 0
    assert observed_tree_files == [str(tree)]
    assert (tmp_path / "P.phylogic_report.html").exists()
    with (tmp_path / "P_cell_population_mcmc_trace.tsv").open() as trace:
        rows = list(csv.DictReader(trace, delimiter="\t"))
    assert len(rows) == 2 * 2 * 4
    assert {row["Iteration"] for row in rows} == {"0", "1", "2", "3"}
    with (tmp_path / "P_cell_population_abundances.tsv").open() as abundances:
        rows = list(csv.DictReader(abundances, delimiter="\t"))
    for sample in samples:
        assert sum(float(row["Cell_abundance"]) for row in rows if row["Sample_ID"] == sample) == 100
