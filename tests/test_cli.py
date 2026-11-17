"""The installed entry points work outside the source directory."""
import importlib
from importlib.resources import files
from pathlib import Path
import subprocess
import sys

import pytest

from PhylogicNDT import build_parser


TOOLS = ["Cluster", "BuildTree", "CellPopulation", "GrowthKinetics",
         "PhylogicSim", "Timing", "SinglePatientTiming", "LeagueModel"]


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_help_and_import(tool, tmp_path):
    result = subprocess.run([sys.executable, "-m", "PhylogicNDT", tool, "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())
    args = build_parser().parse_args([tool])
    module, function = args.func.split(":")
    assert callable(getattr(importlib.import_module(module), function))


def test_import_has_no_cli_or_logging_side_effects(tmp_path):
    code = "import logging; handler = logging.NullHandler(); logging.root.addHandler(handler); import PhylogicNDT; assert handler in logging.root.handlers"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not list(tmp_path.iterdir())


def test_console_entry_point_without_arguments(tmp_path):
    command = Path(sys.executable).parent / "PhylogicNDT"
    result = subprocess.run([str(command)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Choose a tool" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("package,resource", [
    ("data", "supplement_data/Blacklist_SNVs.txt"),
    ("data", "supplement_data/Driver_genes_v1.0.txt"),
    ("data", "supplement_data/cytoBand.txt"),
    ("output", "report_template.html_"),
    ("output", "timing_report_template.html_"),
    ("utils", "rdata_extractor.R"),
])
def test_installed_resources(package, resource):
    assert files(package).joinpath(resource).read_text()
