"""FRAME real-time inference loop.

Pipeline per frame:
  keyboard -> encode -> scene state update -> predict -> GRU update -> decode -> display

Run with --stub for profiling on random noise (no trained model needed).
Run with --checkpoint for real model inference.
Target: >=15fps on real model, >=30fps on stubs.
"""

import argparse
import time
from collections import deque

import torch

from inference.display import FrameDisplay
from inference.keyboard import capture_action
from inference.stub import (
    StubDecoder,
    StubEncoder,
    StubGRUState,
    StubPredictor,
    StubSceneState,
)


def _load_models(checkpoint: str, tokenizer_checkpoint: str, device: torch.device) -> tuple:
    """Load trained predictor and tokenizer for inference.

    Returns:
        (tok_encoder, tok_vq, tok_decoder, predictor, scene_state, gru, device)
    """
    from predictor.gru_state import GRUContinuousState
    from predictor.scene_state import PersistentSceneState
    from predictor.transformer import CausalTransformerPredictor
    from tokenizer.decoder import CNNDecoder
    from tokenizer.encoder import CNNEncoder
    from tokenizer.vq import VectorQuantizer

    # Load tokenizer
    tok_ckpt = torch.load(tokenizer_checkpoint, map_location=device, weights_only=False)
    tok_cfg = tok_ckpt["config"]["model"]

    tok_encoder = CNNEncoder(
        channels=tok_cfg["encoder_channels"],
        codebook_dim=tok_cfg["codebook_dim"],
    ).to(device)
    tok_vq = VectorQuantizer(
        n_codes=tok_cfg["codebook_size"],
        code_dim=tok_cfg["codebook_dim"],
        ema_decay=tok_cfg["ema_decay"],
    ).to(device)
    tok_decoder = CNNDecoder(codebook_dim=tok_cfg["codebook_dim"]).to(device)

    tok_encoder.load_state_dict(tok_ckpt["encoder"])
    tok_vq.load_state_dict(tok_ckpt["vq"])
    tok_decoder.load_state_dict(tok_ckpt["decoder"])
    tok_encoder.requires_grad_(False)
    tok_vq.requires_grad_(False)
    tok_decoder.requires_grad_(False)

    # Load predictor
    pred_ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    pred_cfg = pred_ckpt["config"]["model"]

    predictor = CausalTransformerPredictor(
        n_layers=pred_cfg["n_layers"],
        n_heads=pred_cfg["n_heads"],
        d_model=pred_cfg["d_model"],
        d_ffn=pred_cfg["d_ffn"],
        codebook_size=pred_cfg["codebook_size"],
        tokens_per_frame=pred_cfg["tokens_per_frame"],
        action_dim=pred_cfg["action_dim"],
        action_embed_dim=pred_cfg["action_embed_dim"],
        film_layers=pred_cfg["film_layers"],
        action_dropout=0.0,  # no dropout at inference
    ).to(device)
    predictor.load_state_dict(pred_ckpt["predictor"])
    predictor.requires_grad_(False)

    scene_state = PersistentSceneState(
        dim=pred_cfg["scene_state_dim"],
        alpha=pred_cfg["scene_state_alpha"],
    ).to(device)
    if "scene_state" in pred_ckpt:
        scene_state.load_state_dict(pred_ckpt["scene_state"])

    gru = GRUContinuousState(
        input_dim=pred_cfg["d_model"],
        hidden_dim=pred_cfg["gru_dim"],
    ).to(device)
    gru.load_state_dict(pred_ckpt["gru_state"])
    gru.requires_grad_(False)

    context_frames = pred_cfg["context_frames"]

    return tok_encoder, tok_vq, tok_decoder, predictor, scene_state, gru, context_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAME real-time inference loop")
    parser.add_argument("--stub", action="store_true",
                        help="Use random noise stubs (no trained model)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained predictor checkpoint")
    parser.add_argument("--tokenizer_checkpoint", type=str, default=None,
                        help="Path to trained tokenizer checkpoint")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--profile", action="store_true",
                        help="Print per-step timing breakdown every 30 frames")
    parser.add_argument("--headless", action="store_true",
                        help="Skip display (for headless profiling)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=argmax, >0=stochastic)")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_stub = args.stub

    if use_stub:
        # Stub mode — random noise pipeline
        tok_encoder = StubEncoder()
        predictor = StubPredictor()
        tok_decoder = StubDecoder(resolution=args.resolution)
        scene_state = StubSceneState()
        gru = StubGRUState()
        tok_vq = None
        context_frames = 1
        d_model = 512
    elif args.checkpoint:
        assert args.tokenizer_checkpoint, "Provide --tokenizer_checkpoint with --checkpoint"
        print(f"Loading models from {args.checkpoint}...")
        (tok_encoder, tok_vq, tok_decoder, predictor, scene_state, gru,
         context_frames) = _load_models(
            args.checkpoint, args.tokenizer_checkpoint, device,
        )
        d_model = predictor.d_model
        print(f"  Predictor loaded. Context frames: {context_frames}")
    else:
        parser.error("Provide --stub or --checkpoint + --tokenizer_checkpoint")

    display = None if args.headless else FrameDisplay(resolution=args.resolution)

    # State
    # For real model: maintain a rolling buffer of token indices for context
    token_buffer: list[torch.LongTensor] = []  # each entry: (1, 256)
    gru_hidden = gru.reset(batch_size=1) if use_stub else gru.reset(1).to(device)
    current_frame = torch.rand(1, 3, args.resolution, args.resolution, device=device)

    step_times: deque[float] = deque(maxlen=120)
    running = True
    step = 0

    print("FRAME inference loop started.")
    print(f"  Mode: {'stub' if use_stub else 'real model'}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Device: {device}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Headless: {args.headless}")
    print("  Press ESC to toggle mouse grab, close window to exit.\n")

    try:
        with torch.no_grad():
            while running:
                t_start = time.perf_counter()

                # 1. Keyboard capture
                t0 = time.perf_counter()
                if display is not None:
                    action = capture_action().to(device)
                else:
                    action = torch.zeros(72, device=device)
                    action[7] = 1.0  # noop
                    action[8] = 1.0  # mouse center
                t_keyboard = time.perf_counter() - t0

                if use_stub:
                    # --- Stub pipeline (unchanged) ---
                    t0 = time.perf_counter()
                    token_indices = tok_encoder(current_frame)
                    t_encode = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    scene_token = scene_state.update(torch.zeros(1, 512))
                    t_scene = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    logits = predictor(
                        token_indices, action.unsqueeze(0),
                        scene_token, gru_hidden,
                    )
                    predicted_indices = logits.argmax(dim=-1)
                    t_predict = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    _, gru_hidden = gru.forward(torch.zeros(1, 512), gru_hidden)
                    t_gru = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    next_frame = tok_decoder(predicted_indices)
                    t_decode = time.perf_counter() - t0

                else:
                    # --- Real model pipeline ---
                    # 2. Encode current frame -> tokens
                    t0 = time.perf_counter()
                    z = tok_encoder(current_frame)
                    z_q, _, indices = tok_vq(z)  # indices: (1, 16, 16)
                    cur_tokens = indices.reshape(1, -1)  # (1, 256)
                    token_buffer.append(cur_tokens)
                    # Keep only last context_frames
                    if len(token_buffer) > context_frames:
                        token_buffer.pop(0)
                    t_encode = time.perf_counter() - t0

                    # 3. Scene state update
                    t0 = time.perf_counter()
                    # Use mean of token embeddings as scene state input
                    tok_emb = predictor.token_embed(cur_tokens)  # (1, 256, D)
                    scene_input = tok_emb.mean(dim=1)  # (1, D)
                    scene_token = scene_state.update(scene_input)
                    t_scene = time.perf_counter() - t0

                    # 4. Predict next frame tokens
                    t0 = time.perf_counter()
                    # Stack context tokens: (1, n_context, 256)
                    context = torch.stack(token_buffer, dim=1)
                    logits, info = predictor(
                        context, action.unsqueeze(0),
                        scene_token, gru_hidden,
                    )
                    # logits: (1, 256, codebook_size)

                    # Sample or argmax
                    if args.temperature > 0:
                        probs = torch.softmax(logits / args.temperature, dim=-1)
                        predicted_indices = torch.multinomial(
                            probs.reshape(-1, logits.shape[-1]), 1,
                        ).reshape(1, -1)  # (1, 256)
                    else:
                        predicted_indices = logits.argmax(dim=-1)  # (1, 256)
                    t_predict = time.perf_counter() - t0

                    # 5. GRU update
                    t0 = time.perf_counter()
                    _, gru_hidden = gru(info["hidden_mean"], gru_hidden)
                    t_gru = time.perf_counter() - t0

                    # 6. Decode predicted tokens -> pixels
                    t0 = time.perf_counter()
                    pred_2d = predicted_indices.reshape(1, 16, 16)
                    z_q_pred = tok_vq.lookup(pred_2d)  # (1, 256, 16, 16)
                    next_frame = tok_decoder(z_q_pred)  # (1, 3, 128, 128)
                    t_decode = time.perf_counter() - t0

                # 7. Display
                t0 = time.perf_counter()
                if display is not None:
                    running = display.show(next_frame.squeeze(0))
                t_display = time.perf_counter() - t0

                current_frame = next_frame
                t_total = time.perf_counter() - t_start
                step_times.append(t_total)

                if args.profile and step % 30 == 0:
                    fps = (len(step_times) / sum(step_times)) if step_times else 0
                    print(
                        f"Step {step:5d}: {fps:6.1f} fps | "
                        f"kbd={t_keyboard * 1000:5.1f}ms "
                        f"enc={t_encode * 1000:5.1f}ms "
                        f"scene={t_scene * 1000:5.1f}ms "
                        f"pred={t_predict * 1000:5.1f}ms "
                        f"gru={t_gru * 1000:5.1f}ms "
                        f"dec={t_decode * 1000:5.1f}ms "
                        f"disp={t_display * 1000:5.1f}ms "
                        f"total={t_total * 1000:5.1f}ms"
                    )

                step += 1

                # Headless: run 300 frames then exit
                if args.headless and step >= 300:
                    avg_fps = len(step_times) / sum(step_times) if step_times else 0
                    print(f"\nHeadless profiling complete: {avg_fps:.1f} avg fps "
                          f"over {step} frames")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if display is not None:
            display.close()


if __name__ == "__main__":
    main()
