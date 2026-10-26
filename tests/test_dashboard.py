from __future__ import annotations

import hashlib
import importlib
import unittest
from pathlib import Path

import pandas as pd

from dashboard.config import EXPERIMENTS, FINANCIAL_ASSUMPTIONS, PANEL_PATH
from dashboard.data_loader import (
    load_experiment_artifacts,
    load_model_artifacts,
    load_panel,
    model_artifact_summary,
    validate_panel_schema,
)
from dashboard.simulation_engine import compare_scenarios, derive_baseline_pressure
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_PATHS = [
    PANEL_PATH,
    ROOT / "experiments/exp1_macro_volume",
    ROOT / "experiments/exp2_environmental_rate",
    ROOT / "experiments/exp3_pure_environmental",
    ROOT / "experiments/TRIPLE_BENCHMARK_SYNTHESIS_REPORT.md",
]


def digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


class DashboardSmokeTests(unittest.TestCase):
    def test_each_dashboard_layer_renders_headlessly(self) -> None:
        test_app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(test_app.exception, test_app.exception)
        for layer in [
            "Layer 1 · Urban intelligence",
            "Layer 2 · ML benchmark lab",
            "Layer 3 · What-if simulation",
        ]:
            with self.subTest(layer=layer):
                test_app.radio[0].set_value(layer).run()
                self.assertFalse(test_app.exception, test_app.exception)

    def test_app_and_dashboard_modules_import(self) -> None:
        for module in [
            "app",
            "dashboard.config",
            "dashboard.data_loader",
            "dashboard.simulation_engine",
            "dashboard.views.layer1_urban_bi",
            "dashboard.views.layer2_ml_lab",
            "dashboard.views.layer3_what_if",
        ]:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_loads_panel_and_all_benchmark_artifact_schemas(self) -> None:
        panel = load_panel()
        self.assertEqual(len(panel), 315)
        self.assertTrue({"district_key", "month"}.issubset(panel.columns))
        self.assertFalse(validate_panel_schema(panel))
        self.assertEqual(panel["district_key"].nunique(), 15)
        artifacts = load_experiment_artifacts()
        self.assertEqual(set(artifacts), set(EXPERIMENTS))
        for experiment in artifacts.values():
            self.assertFalse(experiment["metrics"].empty)
            self.assertFalse(experiment["predictions"].empty)
            self.assertFalse(experiment["importance"].empty)
            self.assertTrue(
                {"actual_value", "predicted_value", "district_key", "month"}.issubset(
                    experiment["predictions"].columns
                )
            )

    def test_loads_serialized_models_that_exist(self) -> None:
        models = load_model_artifacts()
        summary = model_artifact_summary(models)
        self.assertEqual(len(models["Experiment 2 — Controlled environmental rate"]), 4)
        self.assertEqual(len(models["Experiment 3 — Pure environmental ablation"]), 4)
        self.assertEqual(len(models["Experiment 1 — Macro volume"]), 0)
        self.assertIn(
            "No serialized model",
            summary.loc[
                summary["Experiment"].eq("Experiment 1 — Macro volume"), "Status"
            ].iloc[0],
        )
        loaded = summary.loc[summary["Status"].eq("Loaded")]
        self.assertEqual(len(loaded), 8)
        self.assertTrue((loaded["Input feature count"].astype(int) > 0).all())
        for experiment_name, directory in [
            (
                "Experiment 2 — Controlled environmental rate",
                "exp2_environmental_rate",
            ),
            (
                "Experiment 3 — Pure environmental ablation",
                "exp3_pure_environmental",
            ),
        ]:
            test_data = pd.read_csv(
                ROOT / "experiments" / directory / "data" / "test.csv"
            )
            for model_name, model in models[experiment_name].items():
                with self.subTest(experiment=experiment_name, model=model_name):
                    input_columns = list(model.feature_names_in_)
                    self.assertEqual(
                        len(model.predict(test_data[input_columns])), len(test_data)
                    )

    def test_calibration_costs_and_scenarios_are_deterministic(self) -> None:
        panel = load_panel()
        calibrated = derive_baseline_pressure(
            panel, "Peak-surge / stress-calibrated capacity"
        )
        target_min = FINANCIAL_ASSUMPTIONS["calibrated_pressure_min"]
        target_max = FINANCIAL_ASSUMPTIONS["calibrated_pressure_max"]
        self.assertAlmostEqual(calibrated["baseline_pressure"].min(), target_min)
        self.assertAlmostEqual(calibrated["baseline_pressure"].max(), target_max)

        settings = {
            "population_growth_pct": 20,
            "capacity_expansion_pct": 15,
            "prevention_reach_pct": 10,
            "access_improvement_minutes": -5,
            "budget_limit_m_egp": 45,
        }
        target = [calibrated["district_key"].iloc[0]]
        first = compare_scenarios(calibrated, settings, target)
        second = compare_scenarios(calibrated, settings, target)
        pd.testing.assert_frame_equal(first, second)
        costs = dict(zip(first["Package"], first["5-yr Cost (M EGP)"]))
        self.assertEqual(costs["center_prevention"], 40.5)
        self.assertEqual(costs["capacity"], 9.0)
        self.assertEqual(costs["access"], 16.0)
        self.assertEqual(costs["prevention"], 7.5)

    def test_protected_inputs_and_experiment_artifacts_are_unchanged(self) -> None:
        before = {path: digest_path(path) for path in PROTECTED_PATHS}
        load_panel.clear()
        load_model_artifacts.clear()
        load_panel()
        load_experiment_artifacts()
        load_model_artifacts()
        after = {path: digest_path(path) for path in PROTECTED_PATHS}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
