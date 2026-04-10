"""FRAME real-time inference loop.

Pipeline per frame:
  keyboard -> encode -> scene state update -> predict -> GRU update -> decode -> display

Run with --stub for profiling on random noise (no trained model needed).
Target: >=30fps on stubs.
"""

import argparse
import time
from collections import deque

import torch

from inference.display import FrameDisplay
from inference.keyboard import capture_action, get_action_name
from inference.stub import (
    StubDecoder,
    StubEncoder,
    StubGRUState,
    StubPredictor,
    StubSceneState,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAME real-time inference loop")
    parser.add_argument("--stub", action="store_true",
                        help="Use random noise stubs (no trained model)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--profile", action="store_true",
                        help="Print per-step timing breakdown every 30 frames")
    parser.add_argument("--headless", action="store_true",
                        help="Skip display (for headless profiling)")
    args = parser.parse_args()

    # Initialize components
    if args.stub:
        encoder = StubEncoder()
        predictor = StubPredictor()
        decoder = StubDecoder(resolution=args.resolution)
        scene_state = StubSceneState()
        gru_state = StubGRUState()
    elif args.checkpoint:
        raise NotImplementedError(
            "Real model loading not yet implemented. Use --stub."
        )
    else:
        parser.error("Provide --stub or --checkpoint")

    display = None if args.headless else FrameDisplay(resolution=args.resolution)

    # Initialize state
    current_frame = torch.rand(1, 3, args.resolution, args.resolution)
    gru_hidden = gru_state.reset(batch_size=1)

    step_times: deque[float] = deque(maxlen=120)
    running = True
    step = 0

    print("FRAME inference loop started.")
    if args.stub:
        print("  Mode: stub (random noise)")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Device: {args.device}")
    print(f"  Headless: {args.headless}")
    print("  Press ESC to toggle mouse grab, close window to exit.\n")

    try:
        while running:
            t_start = time.perf_counter()

            # 1. Keyboard capture
            t0 = time.perf_counter()
            if display is not None:
                action = capture_action()
            else:
                # Headless: random action
                action = torch.zeros(72)
                action[7] = 1.0  # noop
                action[8] = 1.0  # mouse center-ish
            t_keyboard = time.perf_counter() - t0

            # 2. Encode current frame -> tokens
            t0 = time.perf_counter()
            token_indices = encoder(current_frame)
            t_encode = time.perf_counter() - t0

            # 3. Scene state update
            t0 = time.perf_counter()
            scene_token = scene_state.update(
                torch.zeros(1, 512)  # stub: dummy hidden
            )
            t_scene = time.perf_counter() - t0

            # 4. Predict next frame tokens
            t0 = time.perf_counter()
            logits = predictor(
                token_indices, action.unsqueeze(0), scene_token, gru_hidden
            )
            predicted_indices = logits.argmax(dim=-1)
            t_predict = time.perf_counter() - t0

            # 5. GRU update
            t0 = time.perf_counter()
            _, gru_hidden = gru_state.forward(
                torch.zeros(1, 512), gru_hidden
            )
            t_gru = time.perf_counter() - t0

            # 6. Decode tokens -> pixels
            t0 = time.perf_counter()
            next_frame = decoder(predicted_indices)
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
