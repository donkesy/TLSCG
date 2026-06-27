import unittest
from unittest import mock

import torch

from generate_op_seq.tain.unknown import train_lstmvae


class TwoStagePipelineTests(unittest.TestCase):
    @mock.patch.object(train_lstmvae, "generate_filtered_boundary_dataset")
    @mock.patch.object(train_lstmvae, "load_vulnerability_dataset")
    @mock.patch.object(train_lstmvae, "export_extreme_anomaly_dataset")
    @mock.patch.object(train_lstmvae, "train")
    @mock.patch.object(train_lstmvae, "train_normal_vae")
    def test_two_stage_pipeline_saves_constructed_and_generated_anomalies(
        self,
        train_normal,
        train_vulnerability,
        export_constructed,
        load_vulnerability,
        generate_filtered,
    ):
        normal_model = mock.sentinel.normal_model
        vulnerability_model = mock.sentinel.vulnerability_model
        train_normal.return_value = (normal_model, "normal.pt")
        train_vulnerability.return_value = vulnerability_model
        load_vulnerability.return_value = (
            torch.tensor([[0, 1, 2]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            {"reentrancy": 0},
        )
        config = train_lstmvae.TrainConfig(batch_size=1, device="cpu")

        train_lstmvae.run_two_stage_pipeline(
            normal_dataset_name="normal_all",
            vulnerability_dataset_name="mix_vulnerabilities",
            config=config,
            normal_epochs=2,
            vulnerability_epochs=3,
            anomaly_output_path="constructed.csv",
            generated_output_path="generated.csv",
            num_generated=4,
            perturb_mode="random_swap",
            perturb_intensity=0.5,
            perturb_window_size=12,
        )

        export_constructed.assert_called_once_with(
            normal_dataset_name="normal_all",
            output_path="constructed.csv",
            config=mock.ANY,
            limit=None,
            perturb_mode="random_swap",
            perturb_intensity=0.5,
            perturb_window_size=12,
        )
        generate_filtered.assert_called_once()
        self.assertEqual(generate_filtered.call_args.args[3], "generated.csv")
        self.assertEqual(generate_filtered.call_args.kwargs["num_samples"], 4)

    @mock.patch.object(train_lstmvae, "run_two_stage_pipeline")
    def test_main_forwards_constructed_anomaly_options_to_two_stage_pipeline(self, run_pipeline):
        train_lstmvae.main(
            [
                "--two-stage",
                "--save-extreme-anomaly-csv",
                "constructed.csv",
                "--generate-boundary-csv",
                "generated.csv",
                "--perturb-mode",
                "local_shuffle",
                "--perturb-intensity",
                "0.75",
                "--perturb-window-size",
                "16",
            ]
        )

        kwargs = run_pipeline.call_args.kwargs
        self.assertEqual(kwargs["anomaly_output_path"], "constructed.csv")
        self.assertEqual(kwargs["generated_output_path"], "generated.csv")
        self.assertEqual(kwargs["perturb_mode"], "local_shuffle")
        self.assertEqual(kwargs["perturb_intensity"], 0.75)
        self.assertEqual(kwargs["perturb_window_size"], 16)


if __name__ == "__main__":
    unittest.main()
