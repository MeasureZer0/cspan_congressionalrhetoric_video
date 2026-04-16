import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoder import build_encoder
from .faces_frames_dataset import InferenceDataset

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def _collate_fn(batch: list) -> tuple:
    faces_list = [item[0] for item in batch]
    pose_list = [item[1] for item in batch]
    stems = [item[2] for item in batch]

    lengths = torch.tensor([f.shape[0] for f in faces_list], dtype=torch.long)

    faces_pad = torch.nn.utils.rnn.pad_sequence(faces_list, batch_first=True)
    pose_pad = torch.nn.utils.rnn.pad_sequence(pose_list, batch_first=True)

    return faces_pad, pose_pad, lengths, stems


def run_inference(
    model_path: Path,
    encoder: str,
    img_dir: Path,
    csv_file: Path,
    out_csv: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    ds = InferenceDataset(csv_file=csv_file, img_dir=img_dir)
    print(f"Found {len(ds)} samples to process.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    args_stub = argparse.Namespace(encoder=encoder, freeze_backbone=False)
    model, _ = build_encoder(args_stub, device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {encoder} using weights from {model_path}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["stem", "pred_label", "prob_negative", "prob_neutral", "prob_positive"]
        )

        with torch.no_grad():
            for faces, pose, lengths, stems in tqdm(loader, desc="Inference"):
                faces = faces.to(device, non_blocking=True)
                pose = pose.to(device, non_blocking=True)

                logits = model(faces, pose, lengths)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

                for i, stem in enumerate(stems):
                    writer.writerow(
                        [
                            stem,
                            LABEL_NAMES[int(preds[i])],
                            f"{probs[i, 0]:.6f}",
                            f"{probs[i, 1]:.6f}",
                            f"{probs[i, 2]:.6f}",
                        ]
                    )

    print(f"Results saved to: {out_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference - save results to CSV")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument(
        "--encoder", type=str, required=True, choices=["fast_gru", "dual_stream"]
    )
    p.add_argument("--csv-file", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=Path("inference_results.csv"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Using device: {device}")

    run_inference(
        model_path=args.model_path,
        encoder=args.encoder,
        img_dir=args.img_dir,
        csv_file=args.csv_file,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
