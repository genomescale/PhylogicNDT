"""Small real CLI workflows exercise the interfaces between analysis modules."""
import csv
import subprocess
import sys

import numpy as np


def run_tool(directory, *args):
    result = subprocess.run([sys.executable, "-m", "PhylogicNDT", *args],
                            cwd=directory, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def read_rows(path):
    with path.open() as source:
        return list(csv.DictReader(source, delimiter="\t"))


def test_cluster_tree_cell_population_growth_and_timing(tmp_path):
    # Two well-separated mutation populations in two samples.
    grid = np.arange(101)
    header = ["Hugo_Symbol", "Chromosome", "Start_position", "Reference_Allele",
              "Tumor_Seq_Allele2", "t_ref_count", "t_alt_count", "Protein_change",
              "Variant_Classification", "local_cn_a1", "local_cn_a2"]
    header += [f"ccf_raw_{i / 100}" for i in grid]
    sif = ["sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint"]
    for sample_index in range(2):
        sample = f"sample{sample_index}"
        mutation_file = tmp_path / f"{sample}.tsv"
        lines = ["\t".join(header)]
        for mutation in range(20):
            peak = 99 if mutation < 10 else (40 + 20 * sample_index)
            density = np.exp(-0.5 * ((grid - peak) / 3) ** 2)
            density /= density.sum()
            row = ["TP53", "1", str(1000 + mutation), "A", "T", "50", "30",
                   "p.R175H", "Missense_Mutation", "1", "1"]
            lines.append("\t".join(row + [str(value) for value in density]))
        mutation_file.write_text("\n".join(lines) + "\n")
        sif.append(f"{sample}\t{mutation_file}\t\t0.8\t{sample_index}")
    sif_path = tmp_path / "samples.sif"
    sif_path.write_text("\n".join(sif) + "\n")
    run_tool(tmp_path, "Cluster", "-i", "case", "-sif", str(sif_path),
             "-ni", "12", "--seed", "7", "-rb")
    clusters = read_rows(tmp_path / "case.cluster_ccfs.txt")
    assert {row["Sample_ID"] for row in clusters} == {"sample0", "sample1"}
    assert len(read_rows(tmp_path / "case.mut_ccfs.txt")) == 40
    for row in clusters:
        density = [float(row[f"postDP_ccf_{i / 100}"]) for i in range(101)]
        np.testing.assert_allclose(sum(density), 1, atol=1e-5)
    assert (tmp_path / "case.phylogic_report.html").stat().st_size > 1000
    tree_path = tmp_path / "case_build_tree_posteriors.tsv"
    assert read_rows(tree_path)
    run_tool(tmp_path, "CellPopulation", "-i", "cells", "-sif", str(sif_path),
             "-m", "case.mut_ccfs.txt", "-c", "case.cluster_ccfs.txt", "-t", str(tree_path),
             "-ni", "3", "--seed", "7")
    trace = read_rows(tmp_path / "cells_cell_population_mcmc_trace.tsv")
    assert len({row["Iteration"] for row in trace}) == 3
    run_tool(tmp_path, "GrowthKinetics", "-i", "growth", "-ab",
             "cells_cell_population_mcmc_trace.tsv", "-ni", "3", "-t", "0", "10")
    assert read_rows(tmp_path / "growth_growth_rate.tsv")
    assert (tmp_path / "growth.growth_rate.pdf").read_bytes().startswith(b"%PDF")

    segment_file = tmp_path / "segments.tsv"
    segment_file.write_text("Chromosome\tStart\tEnd\tA1.Seg.CN\tA2.Seg.CN\n"
                            "1\t0\t249250621\t1\t1\n")
    timing_sif = tmp_path / "timing.sif"
    timing_sif.write_text("sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint\n" +
                          "\n".join(f"sample{i}\t{tmp_path / 'case.mut_ccfs.txt'}\t{segment_file}\t0.8\t{i}"
                                    for i in range(2)) + "\n")
    run_tool(tmp_path, "Timing", "-i", "timing", "-sif", str(timing_sif))
    assert read_rows(tmp_path / "timing.timing.tsv")
    assert (tmp_path / "timing.comp.tsv").is_file()
