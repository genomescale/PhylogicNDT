"""Exercise serialized reports with Python 3 and current plotting libraries."""

import base64
import json
import re
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from BuildTree.Tree import Tree
from output.PhylogicOutput import PhylogicOutput


def _density(bin_index):
    density = np.zeros(101)
    density[bin_index] = 1.0
    return density


@pytest.mark.parametrize("iterations", [3, 25])
def test_tree_report_round_trip(tmp_path, monkeypatch, iterations):
    """Read integer counts from TSV and render even a short MCMC trace."""
    monkeypatch.chdir(tmp_path)
    densities = {1: [_density(100)], 2: [_density(40)]}
    mutations = [
        SimpleNamespace(
            type="SNV", cluster_assignment=cluster, ccf_1d=densities[cluster][0],
            gene=f"GENE{cluster}", chrN="1", pos=cluster, ref="A", alt="T",
            ref_cnt=20, alt_cnt=10, prot_change="p.V1E", mut_category="Missense_Mutation",
            local_cn_a1=1, local_cn_a2=1,
        )
        for cluster in densities
    ]
    patient = SimpleNamespace(
        indiv_name="report", sample_list=[SimpleNamespace(
            sample_name="sample", mutations=mutations, low_coverage_mutations={},
        )],
    )
    output = PhylogicOutput()
    output.write_patient_cluster_ccfs(patient, densities)
    output.write_patient_mut_ccfs(patient, densities)
    output.write_tree_tsv([{"n_iter": iterations, "likelihood": -1., "edges": [(1, 2)]}], "report")
    output.write_constrained_ccf_tsv({"sample": [(1, 100), (2, 40)]}, {1: [1], 2: [1, 2]}, "report")
    output.generate_html_from_tree(
        "report.mut_ccfs.txt", "report.cluster_ccfs.txt",
        "report_build_tree_posteriors.tsv", "report_constrained_ccf.tsv",
    )

    html = (tmp_path / "report.phylogic_report.html").read_text()
    assert "@@" not in html
    tree_options = json.loads(re.search(r"let tree_options = (.*);", html).group(1))
    assert tree_options["child_dict"] == {"1": [2], "2": [], "null": [1]}
    table = json.loads(re.search(r"\.sample_name_table\((\{.*\})\)", html).group(1))
    assert base64.b64decode(table["pie_plots"][0], validate=True).startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "report_pie_plots/sample.pie_plot.svg").is_file()


def test_pie_plot_png_independent_of_default_save_format():
    tree = Tree()
    root = tree.add_node(1)
    tree.set_root(root)
    with plt.rc_context({"savefig.format": "svg"}):
        image = PhylogicOutput.make_pie_plot(tree, {1: 100})
    assert base64.b64decode(image, validate=True).startswith(b"\x89PNG\r\n\x1a\n")


def test_abundance_sampling_preserves_integer_bins():
    tree = Tree()
    root = tree.add_node(1)
    child = tree.add_node(2)
    grandchild = tree.add_node(3)
    tree.set_root(root)
    tree.add_edge(root, child)
    tree.add_edge(child, grandchild)
    abundances = PhylogicOutput._get_abundances(
        {1: _density(100), 2: _density(40), 3: _density(20)}, tree, n_iter=2,
    )
    assert abundances == {1: 100, 2: 40, 3: 20}


def test_timing_graph_and_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    comps = {("gain_1p", "loss_2q"): (.95, .01, .04)}
    events = [SimpleNamespace(event_name=name, chrN="1", pi_dist=_density(index))
              for name, index in [("gain_1p", 20), ("loss_2q", 80)]]
    timing = SimpleNamespace(WGD=None, all_cn_events={"events": events}, mutations={})
    output = PhylogicOutput()
    output.generate_html_from_timing("timing", timing, comps)
    assert (tmp_path / "timing.comp_graph.png").read_bytes().startswith(b"\x89PNG")
    html = (tmp_path / "timing.phylogic_timing_report.html").read_text()
    assert "@@" not in html
    graph = json.loads(re.search(r"\.comp_graph\((.*)\)", html).group(1))
    assert graph["edges"] == [["gain_1p", "loss_2q"]]
    assert graph["pos"]["gain_1p"][0] < graph["pos"]["loss_2q"][0]


def test_growth_plot_with_current_seaborn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    figures = plt.get_fignums()
    PhylogicOutput().plot_growth_rates({1: np.linspace(-.2, .5, 50)}, "growth")
    assert (tmp_path / "growth.growth_rate.pdf").read_bytes().startswith(b"%PDF")
    assert plt.get_fignums() == figures
