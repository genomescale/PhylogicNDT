"""Scientific workflow regression checks for the Python 3 port."""
import csv
import itertools
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from GrowthKinetics.GrowthKineticsEngine import GrowthKineticsEngine
from LeagueModel.LeagueModelData import League
from PhylogicSim.SimEngine import gen_sample_from_tree_all, get_siblings, run_simulations
from SinglePatientTiming.TimingEngine import TimingMut
from utils.calc_ccf import calc_ccf


def test_simulation_tree_uses_current_networkx_iterators():
    random.seed(31)
    np.random.seed(31)
    tree = nx.DiGraph([(0, 1), (0, 2), (1, 3)])
    assert get_siblings(0, tree) == []
    assert get_siblings(1, tree) == [1, 2]
    ccfs = gen_sample_from_tree_all(tree)
    assert ccfs[0] == 1
    assert all(0 <= value <= 1 for value in ccfs)
    assert ccfs[1] + ccfs[2] <= 1.01
    assert ccfs[3] <= ccfs[1]


def test_simulation_writes_usable_sif_and_normalized_histograms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'clusters.tsv').write_text('1\t0.4\t-1\n1\t0.7\t-1\n')
    (tmp_path / 'purity.tsv').write_text('0.65\n0.85\n')
    (tmp_path / 'coverage.tsv').write_text('100\n120\n')
    (tmp_path / 'segments.tsv').write_text(
        'ID\tChromosome\tStart_Position\tEnd_Position\tMajor_CN\tMinor_CN\n'
        'sample\t1\t1\t100000\t1\t1\n')
    args = SimpleNamespace(clust_file='clusters.tsv', nsamp=2, min_nodes=2,
                           alpha=2, beta=18, nbin=1000, purity=0.7,
                           purity_file='purity.tsv', cov_file='coverage.tsv', artifacts=0.1,
                           cn_dist='segments.tsv', indiv_id='simulation', nmuts=24)
    random.seed(23)
    np.random.seed(23)
    run_simulations(args)
    with open('simulation.sif') as handle:
        samples = list(csv.DictReader(handle, delimiter='\t'))
    assert [float(sample['purity']) for sample in samples] == [0.65, 0.85]
    for sample in samples:
        maf_path = Path(sample['maf_fn'])
        assert maf_path.parent == tmp_path
        table = pd.read_csv(maf_path, sep='\t')
        histograms = table.filter(regex='^ccf_').to_numpy()
        assert histograms.shape[1] == 101
        assert len(table) > 0
        assert np.isfinite(histograms).all()
        np.testing.assert_allclose(histograms.sum(axis=1), 1)
    assert (tmp_path / 'simulation_report.pdf').read_bytes().startswith(b'%PDF')
    assert (tmp_path / 'simulation_plot.png').stat().st_size > 0


def comparison_frame():
    rows = []
    for sample in range(6):
        events = ['TP53_R1Q', 'gain_1q', 'KRAS_G12D' if sample % 2 else 'NRAS_Q61R']
        for first, second in itertools.combinations(events, 2):
            rows.append((f's{sample}', first, second, 0.65, 0.25, 0.1))
    return pd.DataFrame(rows, columns=['sample', 'event1', 'event2', 'p_event1_win', 'p_event2_win', 'unknown'])


@pytest.mark.parametrize('options,expected', [
    ({'keep_only_samples_w_event': 'KRAS'}, {'s1', 's3', 's5'}),
    ({'remove_samps_w_event': 'KRAS'}, {'s0', 's2', 's4'}),
    ({'keep_samps': ['s0', 's1']}, {'s0', 's1'}),
    ({'remove_samps': ['s0', 's1']}, {'s2', 's3', 's4', 's5'}),
])
def test_league_filters_snapshot_dictionary_keys(tmp_path, options, expected):
    league = League(comparison_frame(), cohort=str(tmp_path / 'cohort'), **options)
    assert set(league.events_per_samp) == expected


def test_league_runs_permutations_and_writes_python3_pickle(tmp_path, monkeypatch):
    from PhylogicNDT import main

    monkeypatch.chdir(tmp_path)
    comparison_frame().to_csv('comparisons.tsv', sep='\t', index=False)
    result = main(['LeagueModel', '-comparison_fn', 'comparisons.tsv', '-cohort', 'cohort',
                   '-n_perms', '3', '-n_seasons', '12', '-percent_subset', '0.8', '--seed', '7'])
    assert result == 0
    with open('cohortfig_pkl.pkl', 'rb') as handle:
        league = pickle.load(handle)
    assert league.n_perms == 3
    assert len(league.log_odds_full_run) == 4
    assert all(np.isfinite(values).all() for values in league.log_odds_full_run.values())
    assert Path('cohort.log_odds.pdf').read_bytes().startswith(b'%PDF')
    plt.close('all')


def test_random_tree_simulation_cli(tmp_path, monkeypatch):
    from PhylogicNDT import main

    monkeypatch.chdir(tmp_path)
    command = ['PhylogicSim', '-i', 'random_tree', '-ns', '2', '-nodes', '3', '-nm', '24',
               '--seed', '31']
    result = main(command)
    assert result == 0
    assert len(pd.read_csv('random_tree.sif', sep='\t')) == 2
    assert len(pd.read_csv('random_tree.truth.txt', sep='\t')) > 0
    assert Path('random_tree_report.pdf').read_bytes().startswith(b'%PDF')
    first_truth = Path('random_tree.truth.txt').read_text()
    assert main(command) == 0
    assert Path('random_tree.truth.txt').read_text() == first_truth


def test_league_seed_is_independent_of_python_hash_seed(tmp_path):
    comparison_path = tmp_path / 'comparisons.tsv'
    comparison_frame().to_csv(comparison_path, sep='\t', index=False)
    script = """
import json
import sys
import numpy as np
import pandas as pd
from LeagueModel.LeagueModelData import League
np.random.seed(7)
league = League(pd.read_csv(sys.argv[1], sep='\\t'), cohort='test')
print(json.dumps(league.event_positions, sort_keys=True))
"""
    results = []
    for hash_seed in ('1', '2'):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed,
                   MPLCONFIGDIR=str(tmp_path / 'matplotlib'))
        result = subprocess.run([sys.executable, '-c', script, str(comparison_path)],
                                cwd=tmp_path, env=env,
                                text=True, capture_output=True, check=True)
        results.append(result.stdout)
    assert results[0] == results[1]


def test_timing_multiplicity_and_pi_are_normalized():
    samples = [SimpleNamespace(purity=0.8, sample_name='s1'), SimpleNamespace(purity=0.7, sample_name='s2')]
    ccf = np.zeros((2, 101))
    ccf[:, -1] = 1
    mutation = TimingMut(samples, 'TP53', '1', 100, 'T', 'A', np.array([40, 35]),
                         np.array([60, 65]), np.ones(2), np.full(2, 2), ccf,
                         cluster_assignment=1)
    mutation.get_mult_dist()
    np.testing.assert_allclose(np.exp(mutation.log_mult_dist).sum(axis=0), 1)
    mutation.get_pi_dist(SimpleNamespace(pi_dist=np.ones(101) / 101))
    assert np.isfinite(mutation.pi_dist).all()
    assert mutation.pi_dist.sum() == pytest.approx(1)


def test_growth_uses_sample_times_and_default_stable_burden(monkeypatch):
    monkeypatch.setattr(np.random, 'random', lambda: 0.5)
    trace = {'s1': {1: [20], 3: [30]}, 's2': {1: [40], 3: [50]}}
    engine = GrowthKineticsEngine(None, [])
    engine.estimate_growth_rate(trace, times=np.array([0, 10]), n_iter=1)
    assert engine.growth_rates[1] == pytest.approx([2])
    assert engine.growth_rates[3] == pytest.approx([2])


def test_ccf_histogram_normalization_and_mode():
    histogram = calc_ccf(1, 1, 20, 80, 0.8)
    assert histogram.shape == (101,)
    assert np.isfinite(histogram).all()
    assert histogram.sum() == pytest.approx(1)
    assert np.argmax(histogram) == 50


def test_timing_cli_with_direct_samples(tmp_path, monkeypatch):
    from PhylogicNDT import main

    monkeypatch.chdir(tmp_path)
    columns = ['Patient_ID', 'Sample_ID', 'Cluster_Assignment', 'Chromosome', 'Start_position',
               'Reference_Allele', 'Tumor_Seq_Allele', 'Variant_Type', 't_ref_count', 't_alt_count',
               'Hugo_Symbol', 'Protein_change', 'Variant_Classification']
    columns += [f'preDP_ccf_{i / 100.0}' for i in range(101)]
    with open('mutations.tsv', 'w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(columns)
        for sample in ('s1', 's2'):
            for i, gene in enumerate(('TP53', 'KRAS', 'NRAS', 'BRAF')):
                writer.writerow(['timing', sample, 1, '1', 1000 + i, 'A', 'T', 'SNP', 60, 40,
                                 gene, f'R{i + 1}Q', 'Missense_Mutation'] + [0] * 100 + [1])
    Path('segments.tsv').write_text('Chromosome\tStart\tEnd\tA1.Seg.CN\tA2.Seg.CN\n'
                                    '1\t1\t249250621\t1\t2\n')
    result = main(['Timing', '-i', 'timing', '--seed', '19',
                  '-s', 's1:mutations.tsv:segments.tsv:0.8:0',
                  '-s', 's2:mutations.tsv:segments.tsv:0.8:1'])
    assert result == 0
    timings = pd.read_csv('timing.timing.tsv', sep='\t')
    pi = timings.filter(regex=r'^pi_\d').to_numpy()
    assert len(pi) >= 4
    assert np.isfinite(pi).all()
    np.testing.assert_allclose(pi.sum(axis=1), 1)
    assert Path('timing.comp.tsv').stat().st_size > 100
